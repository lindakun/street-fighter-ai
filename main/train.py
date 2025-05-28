# Copyright 2023 LIN Yi. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys
import random
import numpy as np
import retro
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from street_fighter_custom_wrapper import StreetFighterCustomWrapper

NUM_ENV = 16
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.current_phase = 0
        self.phases = [
            {'stages': [1, 2], 'characters': ['Ryu', 'Ken']},
            {'stages': [1, 2, 3, 4], 'characters': ['Ryu', 'Ken', 'Chun-Li', 'Guile']},
            {'stages': [1, 2, 3, 4, 5, 6], 'characters': ['Ryu', 'Ken', 'Chun-Li', 'Guile', 'E.Honda', 'Blanka']},
            {'stages': list(range(1, 13)), 'characters': ['Ryu', 'Ken', 'Chun-Li', 'Guile', 'E.Honda', 'Blanka', 
                                                        'Zangief', 'Dhalsim', 'M.Bison', 'Sagat', 'Vega', 'Balrog']}
        ]
        self.phase_thresholds = [1000000, 2000000, 3000000]  # 每个阶段的训练步数阈值

    def _on_step(self):
        if self.num_timesteps >= self.phase_thresholds[self.current_phase]:
            self.current_phase = min(self.current_phase + 1, len(self.phases) - 1)
            print(f"Advancing to curriculum phase {self.current_phase + 1}")
        return True

def linear_schedule(initial_value, final_value=0.0):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def get_available_states():
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    return retro.list_states(game)

def make_env(game, state, seed=0, curriculum_phase=0):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED, 
            obs_type=retro.Observations.IMAGE    
        )
        env = StreetFighterCustomWrapper(env)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init

def main():
    # 设置游戏环境
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    available_states = get_available_states()
    
    # 初始化课程学习回调
    curriculum_callback = CurriculumCallback()
    
    # 创建环境
    env = SubprocVecEnv([
        make_env(game, state=random.choice(available_states), seed=i) 
        for i in range(NUM_ENV)
    ])

    # 设置学习率调度器
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
    clip_range_schedule = linear_schedule(0.15, 0.025)

    # 创建模型
    model = PPO(
        "CnnPolicy", 
        env,
        device="cuda", 
        verbose=1,
        n_steps=512,
        batch_size=512,
        n_epochs=4,
        gamma=0.94,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log="logs"
    )

    # 设置保存目录
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)

    # 设置检查点回调
    checkpoint_interval = 31250
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval, 
        save_path=save_dir, 
        name_prefix="ppo_sf2"
    )

    # 将训练日志写入文件
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file
    
        model.learn(
            total_timesteps=int(100000000),
            callback=[checkpoint_callback, curriculum_callback]
        )
        env.close()

    # 恢复标准输出
    sys.stdout = original_stdout

    # 保存最终模型
    model.save(os.path.join(save_dir, "ppo_sf2_final.zip"))

if __name__ == "__main__":
    main()
