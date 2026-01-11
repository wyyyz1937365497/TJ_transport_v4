"""
Gymnasium环境包装器 - 将SUMO环境包装成标准的Gymnasium Env
兼容Stable-Baselines3的VecEnv
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
# 使用比赛标准环境（Frenet坐标系 + 正确的奖励函数）
from .competition_env import CompetitionSumoEnv as SumoEnvironment
from ..constants import (
    MAX_VEHICLES,
    FEATURES_PER_VEHICLE,
    ANGLE_SCALE,
    LANE_INDEX_SCALE,
    POSITION_SCALE,
    DEFAULT_STEP_LENGTH,
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_ACCEL,
    DEFAULT_MAX_DECEL
)


class GymSumoEnv(gym.Env):
    """
    Gymnasium兼容的SUMO环境

    将自定义的SumoEnvironment包装成标准Gymnasium接口，
    支持Stable-Baselines3的VecEnv并行环境。

    特性：
    - 标准的observation_space和action_space
    - 兼容Gymnasium API (reset, step)
    - 自动处理SUMO生命周期
    - 支持并行环境（每个环境独立端口）
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: Dict[str, Any],
        port: int = 8813,
        seed: Optional[int] = None
    ):
        """
        初始化环境

        Args:
            config: SUMO配置字典
            port: SUMO端口（每个环境应使用不同端口）
            seed: 随机种子
        """
        super().__init__()

        # 创建底层SUMO环境
        self.sumo_env = SumoEnvironment(
            config=config,
            use_gui=False,
            port=port,
            disable_port_retry=True  # 并行环境禁用端口重试
        )

        # 配置
        self.config = config
        self.port = port
        self.max_steps = config.get('max_steps', DEFAULT_MAX_STEPS)
        self.step_length = config.get('step_length', DEFAULT_STEP_LENGTH)

        # 设置种子
        if seed is not None:
            self._set_seed(seed)

        # 定义观测空间和动作空间
        self._define_spaces()

    def _define_spaces(self):
        """定义观测空间和动作空间"""
        # 计算总特征维度
        total_features = (
            MAX_VEHICLES * FEATURES_PER_VEHICLE +  # 车辆状态特征
            32 +                                     # 全局统计特征（比赛专用）
            1                                        # 车辆数量
        )

        # 使用扁平化的Box观测空间以兼容Stable-Baselines3
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32
        )

        # 动作空间：控制多个车辆的加速度和换道
        # 扁平化为 MAX_VEHICLES * 2 维向量以兼容 SB3
        self.action_space = gym.spaces.Box(
            low=np.array([DEFAULT_MAX_DECEL, 0.0] * MAX_VEHICLES, dtype=np.float32),
            high=np.array([DEFAULT_MAX_ACCEL, 1.0] * MAX_VEHICLES, dtype=np.float32),
            dtype=np.float32
        )

        self.max_vehicles = MAX_VEHICLES

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境

        Returns:
            (observation, info)
        """
        if seed is not None:
            self._set_seed(seed)

        # 重置SUMO环境
        observation = self.sumo_env.reset()

        # 转换为标准格式
        obs = self._format_observation(observation)
        info = self._get_info(observation)

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步

        Args:
            action: 动作数组 [max_vehicles, 2]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # 解析动作
        vehicle_actions = self._parse_actions(action)

        # 执行一步
        observation, reward, done, info = self.sumo_env.step(vehicle_actions)

        # 转换为标准格式
        obs = self._format_observation(observation)

        # 判断是否终止
        terminated = done
        truncated = self.sumo_env.current_step >= self.max_steps

        # 合并info
        info = self._get_info(observation)
        info['truncated'] = truncated

        return obs, reward, terminated, truncated, info

    def _parse_actions(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """
        解析动作向量为车辆控制字典

        Args:
            action: 扁平化的 [max_vehicles * 2] 数组

        Returns:
            {vehicle_id: [acceleration, lane_change]}
        """
        actions = {}
        obs = self.sumo_env._get_observation()

        # 获取当前ICV列表
        icv_ids = list(obs.get('icv_ids', set()))

        # 将扁平化的动作向量重新整形为 [max_vehicles, 2]
        action_reshaped = action.reshape(self.max_vehicles, 2)

        # 为每个ICV分配动作
        for i, veh_id in enumerate(icv_ids[:self.max_vehicles]):
            if i < len(action_reshaped):
                actions[veh_id] = action_reshaped[i]

        return actions

    def _format_observation(self, observation: Dict) -> np.ndarray:
        """
        将SUMO观测格式化为标准Gymnasium格式（使用Frenet坐标系）

        Args:
            observation: SUMO原始观测（来自CompetitionSumoEnv，包含Frenet坐标）

        Returns:
            扁平化的观测数组，用于Stable-Baselines3
        """
        vehicle_states = observation.get('vehicle_states', {})
        global_stats = observation.get('global_stats', np.zeros(32))
        icv_ids = observation.get('icv_ids', set())

        # 车辆状态向量化（Frenet坐标系9维特征）
        vehicle_features = []
        for veh_id, state in vehicle_states.items():
            # 提取Frenet坐标系特征并归一化
            features = [
                state.get('s', 0.0) / 1000.0,              # 纵向位置（归一化到0-1）
                state.get('d', 0.0) / 10.0,               # 横向偏移（归一化，假设车道宽度~3-4m）
                state.get('vs', 0.0) / 30.0,              # 纵向速度（归一化到0-30m/s）
                state.get('vd', 0.0) / 10.0,              # 横向速度（归一化）
                state.get('speed', 0.0) / 30.0,           # 总速度（归一化到0-30m/s）
                state.get('acceleration', 0.0) / 3.0,     # 加速度（归一化到-3~3m/s²）
                state.get('lane_index', 0.0) / 10.0,      # 车道索引（归一化）
                state.get('angle', 0.0) / 360.0,          # 航向角（归一化到0-360度）
                1.0 if veh_id in icv_ids else 0.0         # is_icv标志
            ]
            vehicle_features.extend(features)

        # Padding到固定大小
        max_features = self.max_vehicles * FEATURES_PER_VEHICLE
        if len(vehicle_features) < max_features:
            vehicle_features.extend([0.0] * (max_features - len(vehicle_features)))
        else:
            vehicle_features = vehicle_features[:max_features]

        # 拼接所有特征为一个扁平数组
        flat_obs = np.array(vehicle_features, dtype=np.float32)
        flat_obs = np.concatenate([
            flat_obs,                              # 车辆状态特征 (MAX_VEHICLES * FEATURES_PER_VEHICLE)
            global_stats.flatten(),                # 全局统计特征 (32)
            [len(vehicle_states)]                  # 车辆数量 (1)
        ]).astype(np.float32)

        return flat_obs

    def _get_info(self, observation: Dict) -> Dict:
        """获取额外信息"""
        return {
            'vehicle_ids': observation.get('vehicle_ids', []),
            'icv_ids': list(observation.get('icv_ids', set())),
            'step': self.sumo_env.current_step
        }

    def _set_seed(self, seed: int):
        """设置随机种子"""
        self.np_random = np.random.default_rng(seed)

    def render(self):
        """
        渲染环境

        注意：并行环境中不支持SUMO GUI渲染。
        如需可视化，请使用单线程环境并设置use_gui=True。
        """
        raise NotImplementedError(
            "SUMO并行环境不支持渲染。如需可视化，请使用单线程环境。"
        )

    def close(self):
        """关闭环境"""
        if hasattr(self, 'sumo_env'):
            self.sumo_env.close()


def make_gym_env(
    config: Dict[str, Any],
    port: int = 8813,
    seed: Optional[int] = None
) -> GymSumoEnv:
    """
    创建Gymnasium环境的工厂函数

    用于SubprocVecEnv

    Args:
        config: SUMO配置
        port: SUMO端口
        seed: 随机种子

    Returns:
        GymSumoEnv实例
    """
    return GymSumoEnv(config=config, port=port, seed=seed)
