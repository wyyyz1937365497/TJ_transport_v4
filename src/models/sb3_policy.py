"""
自定义PPO策略 - 集成现有TrafficController架构
使用Stable-Baselines3的自定义策略功能
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import gymnasium as gym
import numpy as np

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

from .traffic_controller import TrafficController
from .gnn import GraphBuilder


class TrafficControllerFeatureExtractor(BaseFeaturesExtractor):
    """
    使用TrafficController作为特征提取器

    将现有的TrafficController网络包装成SB3兼容的特征提取器。
    """

    def __init__(
        self,
        observation_space: gym.Space,
        config: Dict[str, Any],
        device: torch.device
    ):
        """
        初始化特征提取器

        Args:
            observation_space: 观测空间
            config: 配置字典
            device: 计算设备
        """
        super().__init__(observation_space, features_dim=512)  # 固定输出维度

        self.config = config
        self._device_internal = device

        # 创建TrafficController（但只使用前向部分）
        # 我们将用GNN+世界模型作为特征提取器
        self.traffic_controller = TrafficController(config=config).to(self._device_internal)

        # 创建图构建器
        self.graph_builder = GraphBuilder(
            interaction_radius=config.get('interaction_radius', 100.0),
            max_neighbors=config.get('max_neighbors', 8),
            lane_change_distance=config.get('lane_change_distance', 50.0)
        )

        # 冻结安全层（特征提取阶段不需要）
        for param in self.traffic_controller.safety_shield.parameters():
            param.requires_grad = False

        # 输出投影层（将GNN+世界模型输出映射到固定维度）
        self.output_projection = nn.Sequential(
            nn.Linear(256 + 256, 512),  # GNN输出 + 世界模型输出
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512)
        ).to(self._device_internal)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播

        Args:
            observations: 观测字典，包含扁平化的车辆状态

        Returns:
            特征向量 [batch_size, 512]
        """
        batch_size = observations['vehicle_states'].shape[0]
        device = observations['vehicle_states'].device

        # 收集每个样本的特征
        all_features = []

        for i in range(batch_size):
            # 提取单个样本的观测
            vehicle_states_flat = observations['vehicle_states'][i]  # [256]
            global_stats = observations['global_stats'][i]  # [16]
            num_vehicles = int(observations['num_vehicles'][i].item())

            # 从扁平化的观测重建车辆状态字典
            vehicle_states = self._reconstruct_vehicle_states(
                vehicle_states_flat, num_vehicles, device
            )

            # 构建图（假设所有车辆都是ICV用于特征提取）
            icv_ids = set(vehicle_states.keys())
            graph_data = self.graph_builder.build_graph(vehicle_states, icv_ids)

            # 移动到正确的设备
            graph_data = graph_data.to(device)

            # 通过GNN提取特征
            with torch.set_grad_enabled(self.training):
                if graph_data.x.size(0) > 0:
                    # 有车辆时使用GNN
                    gnn_output = self.traffic_controller.risk_gnn(
                        node_features=graph_data.x,
                        edge_index=graph_data.edge_index,
                        edge_features=graph_data.edge_attr,
                        batch=None  # 单图模式
                    )

                    # 获取全局嵌入
                    graph_embedding = gnn_output['global_embedding'].squeeze(0)

                    # 世界模型（简化版：不使用预测，直接使用GNN输出）
                    # 由于世界模型需要特定输入格式，这里我们简化处理
                    combined = torch.cat([
                        graph_embedding,
                        global_stats  # 使用全局统计信息补充
                    ])
                else:
                    # 无车辆时使用零向量
                    combined = torch.cat([
                        torch.zeros(256, device=device),
                        global_stats
                    ])

                # 投影到固定维度
                features = self.output_projection(combined)
                all_features.append(features)

        # 堆叠成批次
        features_batch = torch.stack(all_features, dim=0)

        return features_batch

    def _reconstruct_vehicle_states(
        self,
        flat_features: torch.Tensor,
        num_vehicles: int,
        device: torch.device
    ) -> Dict[str, Dict]:
        """
        从扁平化的特征重建车辆状态字典

        Args:
            flat_features: [256] 扁平化特征
            num_vehicles: 车辆数量
            device: 设备

        Returns:
            {vehicle_id: {state_dict}}
        """
        vehicle_states = {}

        # 根据 gym_wrapper.py 中的特征提取逻辑重建
        # 每辆车有5个特征: speed, acceleration, angle/360, lane_index/10, position/1000
        features_per_vehicle = 5

        for i in range(min(num_vehicles, 32)):  # 最多32辆车
            start_idx = i * features_per_vehicle
            end_idx = start_idx + features_per_vehicle

            if end_idx > len(flat_features):
                break

            # 提取特征
            speed = float(flat_features[start_idx])
            acceleration = float(flat_features[start_idx + 1])
            angle = float(flat_features[start_idx + 2]) * 360.0
            lane_index = float(flat_features[start_idx + 3]) * 10.0
            position = float(flat_features[start_idx + 4]) * 1000.0

            # 构建车辆状态（简化版，只包含GNN需要的字段）
            vehicle_states[f"veh_{i}"] = {
                'id': f"veh_{i}",
                'x': position,
                'y': 0.0,  # 简化：只有一维位置
                'z': 0.0,
                'vx': speed,
                'vy': 0.0,
                'ax': acceleration,
                'ay': 0.0,
                'lane_index': int(lane_index),
                'lane_id': int(lane_index),  # GraphBuilder需要
                'speed': speed,
                'acceleration': acceleration,
                'angle': angle,
                'position': position
            }

        return vehicle_states


class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    自定义Actor-Critic策略

    集成现有的TrafficController作为特征提取器，
    然后添加标准的actor和critic头。

    架构：
    1. Feature Extractor: TrafficController (GNN + World Model)
    2. Actor Head: 策略网络（输出动作分布）
    3. Critic Head: 价值网络（输出状态价值）
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: callable,
        config: Dict[str, Any],
        **kwargs
    ):
        """
        初始化策略

        Args:
            observation_space: 观测空间
            action_space: 动作空间
            lr_schedule: 学习率调度
            config: 配置字典
            **kwargs: SB3其他参数
        """
        # 保存配置
        self.config = config
        self._device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # 调用父类初始化
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **kwargs
        )

        # 替换特征提取器
        # 移除默认的特征提取器，使用我们的TrafficController
        self.features_extractor = TrafficControllerFeatureExtractor(
            observation_space=observation_space,
            config=config,
            device=self.device
        )

        # 更新features_dim
        self.features_dim = 512

        # 重新构建action_net和value_net（使用正确的输入维度）
        # Actor网络：输出动作均值
        self.action_net = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space.shape[0])  # 输出动作维度
        ).to(self.device)

        # Critic网络：输出状态价值
        self.value_net = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)

    def _build_mlp_extractor(self) -> None:
        """设置mlp_extractor属性（避免AttributeError）"""
        # 创建一个假的mlp_extractor占位符
        # 实际特征提取由TrafficController完成
        class DummyExtractor(nn.Module):
            def __init__(self, features_dim):
                super().__init__()
                self.latent_dim_pi = features_dim
                self.latent_dim_vf = features_dim

        self.mlp_extractor = DummyExtractor(self.features_dim)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            obs: 观测
            deterministic: 是否使用确定性策略

        Returns:
            (actions, values, log_prob)
        """
        # 特征提取
        features = self.extract_features(obs)

        # 计算动作均值
        action_mean = self.action_net(features)

        # 计算动作标准差（可学习参数）
        log_std = self.log_std
        action_std = torch.ones_like(action_mean) * torch.exp(log_std)

        # 创建分布
        distribution = self.action_dist_cls(action_mean, action_std)

        # 采样动作
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.get_actions()

        # 计算log概率
        log_prob = distribution.log_prob(actions)

        # 计算状态价值
        values = self.value_net(features)

        return actions, values.flatten(), log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作（用于PPO训练）

        Args:
            obs: 观测
            actions: 动作

        Returns:
            (values, log_prob, entropy)
        """
        # 特征提取
        features = self.extract_features(obs)

        # 计算动作均值
        action_mean = self.action_net(features)

        # 计算动作标准差
        action_std = torch.ones_like(action_mean) * torch.exp(self.log_std)

        # 创建分布
        distribution = self.action_dist_cls(action_mean, action_std)

        # 计算log概率和熵
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # 计算状态价值
        values = self.value_net(features)

        return values.flatten(), log_prob, distribution.entropy()

    def get_distribution(self, obs: torch.Tensor):
        """获取动作分布"""
        features = self.extract_features(obs)
        action_mean = self.action_net(features)
        action_std = torch.ones_like(action_mean) * torch.exp(self.log_std)
        return self.action_dist_cls(action_mean, action_std)


def create_custom_policy(
    config: Dict[str, Any]
) -> type:
    """
    创建自定义策略类的工厂函数

    Args:
        config: 配置字典

    Returns:
        策略类
    """
    # 使用partial绑定config
    from functools import partial

    class PolicyWithConfig(CustomActorCriticPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, config=config, **kwargs)

    return PolicyWithConfig
