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

import retro
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from street_fighter_custom_wrapper import StreetFighterCustomWrapper

RESET_ROUND = True # Reset the round when fight is over. 
RENDERING = False
MODEL_PATH = r"trained_models/ppo_ryu_2000000_steps"

def get_available_states():
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    return retro.list_states(game)

def make_env(game, state, reset_round=True, rendering=False):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED, 
            obs_type=retro.Observations.IMAGE
        )
        env = StreetFighterCustomWrapper(env, reset_round=reset_round, rendering=rendering)
        env = Monitor(env)
        return env
    return _init

def evaluate_model(model_path, num_episodes=5, rendering=False):
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    available_states = get_available_states()
    
    # 评估结果统计
    total_rewards = []
    stage_clears = 0
    character_wins = {}
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 对每个状态进行评估
    for state in available_states:
        print(f"\n评估状态: {state}")
        env = make_env(game, state, reset_round=True, rendering=rendering)()
        
        # 评估当前状态
        mean_reward, std_reward = evaluate_policy(
            model, 
            env, 
            n_eval_episodes=num_episodes,
            deterministic=True,
            return_episode_rewards=True
        )
        
        # 记录结果
        total_rewards.extend(mean_reward)
        
        # 统计关卡通过情况
        for episode_reward in mean_reward:
            if episode_reward > 0:  # 假设正奖励表示胜利
                stage_clears += 1
                
        # 统计角色胜率
        character = state.split('.')[-2]  # 假设状态名称格式为 "xxx.LevelX.CharacterVsCharacter"
        if character not in character_wins:
            character_wins[character] = 0
        character_wins[character] += sum(1 for r in mean_reward if r > 0)
        
        env.close()
    
    # 输出评估结果
    print("\n评估结果汇总:")
    print(f"平均奖励: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    print(f"关卡通过率: {stage_clears / (len(available_states) * num_episodes) * 100:.2f}%")
    print("\n角色胜率:")
    for character, wins in character_wins.items():
        win_rate = wins / (len(available_states) * num_episodes) * 100
        print(f"{character}: {win_rate:.2f}%")

if __name__ == "__main__":
    MODEL_PATH = "trained_models/ppo_sf2_final.zip"
    evaluate_model(MODEL_PATH, num_episodes=5, rendering=True)
