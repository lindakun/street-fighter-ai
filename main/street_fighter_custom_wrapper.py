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

import math
import time
import collections
import numpy as np
import gym
import retro

# Custom environment wrapper
class StreetFighterCustomWrapper(gym.Wrapper):
    def __init__(self, env, reset_round=True, rendering=False):
        super(StreetFighterCustomWrapper, self).__init__(env)
        self.env = env

        # 帧堆叠设置
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)
        self.num_step_frames = 6

        # 奖励系数
        self.reward_coeff = 3.0
        self.stage_clear_bonus = 1000.0
        self.final_boss_bonus = 2000.0

        # 游戏状态追踪
        self.total_timesteps = 0
        self.current_stage = 1
        self.max_stages = 12  # 街头霸王II的关卡数
        self.is_final_boss = False

        # 生命值设置
        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        # 观察空间设置 - 增加角色信息通道
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255, 
            shape=(100, 128, 4),  # 增加一个通道用于角色信息
            dtype=np.uint8
        )
        
        self.reset_round = reset_round
        self.rendering = rendering

        # 角色信息映射
        self.character_info = {
            'Ryu': 0,
            'Ken': 1,
            'Chun-Li': 2,
            'Guile': 3,
            'E.Honda': 4,
            'Blanka': 5,
            'Zangief': 6,
            'Dhalsim': 7,
            'M.Bison': 8,
            'Sagat': 9,
            'Vega': 10,
            'Balrog': 11
        }
        
        # 当前角色信息
        self.current_character = None
        self.character_one_hot = np.zeros(len(self.character_info))

    def _get_character_info(self, info):
        # 从游戏信息中提取当前角色信息
        # 这里需要根据实际游戏内存地址来获取角色信息
        # 示例实现，实际需要根据游戏ROM的具体情况调整
        character_id = info.get('character_id', 0)
        self.character_one_hot = np.zeros(len(self.character_info))
        self.character_one_hot[character_id] = 1
        return self.character_one_hot

    def _stack_observation(self):
        # 将帧堆叠和角色信息组合
        frame_stack = np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)
        character_channel = np.full((100, 128, 1), self.character_one_hot[0] * 255, dtype=np.uint8)
        return np.concatenate([frame_stack, character_channel], axis=-1)

    def reset(self):
        observation = self.env.reset()
        
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp
        self.total_timesteps = 0
        
        # 清除帧堆叠并添加初始观察
        self.frame_stack.clear()
        for _ in range(self.num_frames):
            self.frame_stack.append(observation[::2, ::2, :])

        # 获取初始角色信息
        info = self.env.get_state()
        self._get_character_info(info)
        
        return self._stack_observation()

    def step(self, action):
        custom_done = False
        stage_cleared = False

        obs, _reward, _done, info = self.env.step(action)
        self.frame_stack.append(obs[::2, ::2, :])

        if self.rendering:
            self.env.render()
            time.sleep(0.01)

        for _ in range(self.num_step_frames - 1):
            obs, _reward, _done, info = self.env.step(action)
            self.frame_stack.append(obs[::2, ::2, :])
            if self.rendering:
                self.env.render()
                time.sleep(0.01)

        curr_player_health = info['agent_hp']
        curr_oppont_health = info['enemy_hp']
        
        self.total_timesteps += self.num_step_frames
        
        # 更新角色信息
        self._get_character_info(info)
        
        # 计算基础奖励
        if curr_player_health < 0:
            custom_reward = -math.pow(self.full_hp, (curr_oppont_health + 1) / (self.full_hp + 1))
            custom_done = True
        elif curr_oppont_health < 0:
            # 检查是否是最终Boss
            if self.current_stage == self.max_stages:
                custom_reward = self.final_boss_bonus
                stage_cleared = True
            else:
                custom_reward = self.stage_clear_bonus
                stage_cleared = True
                self.current_stage += 1
            custom_done = True
        else:
            custom_reward = self.reward_coeff * (self.prev_oppont_health - curr_oppont_health) - \
                          (self.prev_player_health - curr_player_health)
            self.prev_player_health = curr_player_health
            self.prev_oppont_health = curr_oppont_health
            custom_done = False

        if not self.reset_round:
            custom_done = False

        # 归一化奖励
        normalized_reward = 0.001 * custom_reward

        return self._stack_observation(), normalized_reward, custom_done, {
            **info,
            'stage_cleared': stage_cleared,
            'current_stage': self.current_stage,
            'character_info': self.character_one_hot
        }
    