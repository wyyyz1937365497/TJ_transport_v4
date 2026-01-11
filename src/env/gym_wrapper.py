"""
Gymnasium环境包装器 - 将SUMO环境包装成标准的Gymnasium Env
兼容Stable-Baselines3的VecEnv
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .sumo_env import SumoEnvironment


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
        self.max_steps = config.get('max_steps', 3600)
        self.step_length = config.get('step_length', 0.1)

        # 设置种子
        if seed is not None:
            self._set_seed(seed)

        # 定义观测空间和动作空间
        self._define_spaces()

    def _define_spaces(self):
        """定义观测空间和动作空间"""
        # 观测空间：包含车辆状态和全局统计
        # 这是一个简化的实现，实际可以根据需求调整
        self.observation_space = gym.spaces.Dict({
            # 车辆状态（如果有车辆的话）
            # 这里使用Box表示可以处理变长车辆列表
            # 实际观测会在reset/step中动态生成
            'vehicle_states': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(256,),  # 固定大小，用padding处理
                dtype=np.float32
            ),
            # 全局统计特征（16维）
            'global_stats': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(16,),
                dtype=np.float32
            ),
            'num_vehicles': gym.spaces.Box(
                low=0,
                high=1000,
                shape=(1,),
                dtype=np.int32
            )
        })

        # 动作空间：控制多个车辆的加速度和换道
        # 简化版：固定最多控制32辆车，每辆2个动作（加速度、换道）
        max_vehicles = 32
        self.action_space = gym.spaces.Box(
            low=np.array([[-3.0, 0.0]] * max_vehicles),  # 最小加速度, 不换道
            high=np.array([[2.0, 1.0]] * max_vehicles),   # 最大加速度, 换道
            dtype=np.float32
        )

        self.max_vehicles = max_vehicles

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
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

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
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
            action: [max_vehicles, 2] 数组

        Returns:
            {vehicle_id: [acceleration, lane_change]}
        """
        actions = {}
        obs = self.sumo_env._get_observation()

        # 获取当前ICV列表
        icv_ids = list(obs.get('icv_ids', set()))

        # 为每个ICV分配动作
        for i, veh_id in enumerate(icv_ids[:self.max_vehicles]):
            if i < len(action):
                actions[veh_id] = action[i]

        return actions

    def _format_observation(self, observation: Dict) -> Dict:
        """
        将SUMO观测格式化为标准Gymnasium格式

        Args:
            observation: SUMO原始观测

        Returns:
            标准化观测字典
        """
        vehicle_states = observation.get('vehicle_states', {})
        global_stats = observation.get('global_stats', np.zeros(16))

        # 车辆状态向量化
        vehicle_features = []
        for veh_id, state in vehicle_states.items():
            # 提取关键特征
            features = [
                state.get('speed', 0.0),
                state.get('acceleration', 0.0),
                state.get('angle', 0.0) / 360.0,  # 归一化角度
                state.get('lane_index', 0.0) / 10.0,  # 归一化车道
                state.get('position', 0.0) / 1000.0,  # 归一化位置
            ]
            vehicle_features.extend(features)

        # Padding到固定大小
        max_features = self.max_vehicles * 5
        if len(vehicle_features) < max_features:
            vehicle_features.extend([0.0] * (max_features - len(vehicle_features)))
        else:
            vehicle_features = vehicle_features[:max_features]

        return {
            'vehicle_states': np.array(vehicle_features, dtype=np.float32),
            'global_stats': np.array(global_stats, dtype=np.float32),
            'num_vehicles': np.array([len(vehicle_states)], dtype=np.int32)
        }

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
        """渲染（SUMO不支持，保留接口）"""
        pass

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
