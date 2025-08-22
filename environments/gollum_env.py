# minimind/environments/gollum_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GollumEnv(gym.Env):
    """
    咕噜的“无头世界”脚本，一个用于强化学习训练的、高效的数学宇宙。
    它遵循 Gymnasium (OpenAI Gym) 的标准接口。
    """
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # --- 1. 定义动作空间 (Action Space) ---
        # 咕噜可以做什么？我们定义3个离散动作：
        # 0: 休息 (缓慢恢复能量)
        # 1: 接近玩具 (可能增加满意度)
        # 2: 远离玩家 (可能增加安全感)
        self.action_space = spaces.Discrete(3)

        # --- 2. 定义观察空间 (Observation Space) ---
        # 咕噜能感知到什么？这必须与阶段一的LTC模型输入维度(5)匹配
        # [光照强度, 噪音水平, 玩家距离, 玩具距离, 自身能量水平]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

        # --- 3. 模拟世界中的实体 (初始状态) ---
        self.player_position = np.array([5.0, 5.0])
        self.toy_position = np.array([1.0, 1.0])
        self.gollum_position = np.array([0.0, 0.0])

        # --- 4. 咕噜的内在状态 (由环境管理) ---
        # [能量, 安全感, 满意度, 信任度]
        self._internal_states = np.array([0.8, 0.5, 0.5, 0.5]) 
        
        self.current_step = 0
        self.max_steps_per_episode = 500 # 设置一个最长回合步数，防止无限循环
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        """感知抽象层：将世界状态“翻译”成咕噜的感知信号"""
        light_intensity = 0.8
        noise_level = 0.2
        
        player_dist = np.linalg.norm(self.gollum_position - self.player_position)
        toy_dist = np.linalg.norm(self.gollum_position - self.toy_position)
        
        # 将距离归一化到0-1范围 (距离越近，信号值越大)
        normalized_player_dist = np.clip(1.0 - player_dist / 10.0, 0, 1)
        normalized_toy_dist = np.clip(1.0 - toy_dist / 10.0, 0, 1)

        # 返回5维的观察向量
        return np.array([
            light_intensity,
            noise_level,
            normalized_player_dist,
            normalized_toy_dist,
            self._internal_states[0] # 内感受：咕噜能感知自己的能量
        ], dtype=np.float32)

    def _get_info(self):
        """提供用于调试的额外信息"""
        return {"gollum_position": self.gollum_position, "internal_states": self._internal_states}
        
    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        super().reset(seed=seed)
        
        self.gollum_position = np.array([0.0, 0.0])
        self._internal_states = np.array([0.8, 0.5, 0.5, 0.5])
        self.current_step = 0
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """
        环境的核心：执行一步模拟。
        接收动作，更新状态，计算奖励。
        """
        self.current_step += 1
        
        # 1. 根据动作更新咕噜的位置
        if action == 0: # 休息
            pass
        elif action == 1: # 接近玩具
            direction = (self.toy_position - self.gollum_position)
            self.gollum_position += direction * 0.2
        elif action == 2: # 远离玩家 (为了简化，假设是朝原点反方向移动)
            self.gollum_position -= self.player_position * 0.1
        
        # 2. 更新内在状态（世界的“心理物理学定律”）
        energy_change = 0.01 if action == 0 else -0.02
        self._internal_states[0] += energy_change
        
        player_dist = np.linalg.norm(self.gollum_position - self.player_position)
        self._internal_states[1] = np.clip(player_dist / 10.0, 0.2, 0.9)

        toy_dist = np.linalg.norm(self.gollum_position - self.toy_position)
        self._internal_states[2] = np.clip(1.0 - toy_dist / 5.0, 0.1, 0.9)
        
        self._internal_states = np.clip(self._internal_states, 0, 1)

        # 3. 计算奖励函数 (咕噜的“幸福感”)
        reward = (self._internal_states[0] * 0.2 + 
                  self._internal_states[1] * 0.4 + 
                  self._internal_states[2] * 0.4)
        
        # 4. 判断回合是否结束
        terminated = self._internal_states[0] <= 0.01 # 能量耗尽
        truncated = self.current_step >= self.max_steps_per_episode # 达到最大步数

        # 5. 获取新的观察和信息
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info