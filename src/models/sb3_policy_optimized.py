"""
优化的PPO策略 - 高性能特征提取器
通过批量化处理和减少CPU/GPU传输来提升性能
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import gymnasium as gym
import numpy as np

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution

from .traffic_controller import TrafficController
from ..constants import (
    DEFAULT_INTERACTION_RADIUS,
    DEFAULT_MAX_NEIGHBORS,
    DEFAULT_LANE_CHANGE_DISTANCE,
    GNN_OUTPUT_DIM,
    WORLD_MODEL_HIDDEN_DIM,
    FEATURE_EXTRACTOR_OUTPUT_DIM,
    MAX_VEHICLES,
    FEATURES_PER_VEHICLE,
    ANGLE_SCALE,
    LANE_INDEX_SCALE,
    POSITION_SCALE
)


class OptimizedTrafficControllerFeatureExtractor(BaseFeaturesExtractor):
    """
    优化的特征提取器 - 批量化处理，减少CPU/GPU传输

    关键优化：
    1. 移除Python循环，使用批量矩阵运算
    2. 简化图构建，使用预定义的边连接模式
    3. 所有计算在GPU上完成（除了必要的重建）
    """

    def __init__(
        self,
        observation_space: gym.Space,
        config: Dict[str, Any],
        device: torch.device
    ):
        super().__init__(observation_space, features_dim=FEATURE_EXTRACTOR_OUTPUT_DIM)

        self.config = config
        self._device_internal = device

        # 创建TrafficController
        self.traffic_controller = TrafficController(config=config).to(self._device_internal)

        # 冻结安全层
        for param in self.traffic_controller.safety_shield.parameters():
            param.requires_grad = False

        # 获取世界模型隐藏维度
        world_model_hidden_dim = config.get('world_model', {}).get('hidden_dim', WORLD_MODEL_HIDDEN_DIM)

        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(GNN_OUTPUT_DIM + world_model_hidden_dim, FEATURE_EXTRACTOR_OUTPUT_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(FEATURE_EXTRACTOR_OUTPUT_DIM, FEATURE_EXTRACTOR_OUTPUT_DIM)
        ).to(self._device_internal)

        # 创建简化的特征编码器（替代图构建）
        # 直接从扁平化的车辆状态编码为特征
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(FEATURES_PER_VEHICLE, 64),  # 5个特征 -> 64维
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        ).to(self._device_internal)

        # 全局统计编码器
        self.global_stats_encoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        ).to(self._device_internal)

        # 组合编码器（输出GNN输入维度）
        self.combiner = nn.Sequential(
            nn.Linear(128 + 128 + MAX_VEHICLES * 128, GNN_OUTPUT_DIM),
            nn.ReLU(),
            nn.LayerNorm(GNN_OUTPUT_DIM)
        ).to(self._device_internal)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        批量化前向传播 - 无Python循环

        Args:
            observations: Dict with keys 'vehicle_states', 'global_stats', 'num_vehicles'
                      vehicle_states: [batch_size, 160] (32辆 * 5特征)
                      global_stats: [batch_size, 16]
                      num_vehicles: [batch_size, 1]

        Returns:
            features: [batch_size, 512]
        """
        device = observations['vehicle_states'].device
        batch_size = observations['vehicle_states'].shape[0]

        # 1. 批量重建车辆状态（使用向量化操作）
        # vehicle_states: [batch_size, MAX_VEHICLES * FEATURES_PER_VEHICLE]
        vehicle_states_flat = observations['vehicle_states']  # [B, 160]

        # 重塑为 [batch_size, MAX_VEHICLES, FEATURES_PER_VEHICLE]
        vehicle_states_reshaped = vehicle_states_flat.view(batch_size, MAX_VEHICLES, FEATURES_PER_VEHICLE)

        # 反归一化特征（在GPU上完成）
        vehicle_states_denorm = vehicle_states_reshaped.clone()
        vehicle_states_denorm[:, :, 2] *= ANGLE_SCALE  # 角度
        vehicle_states_denorm[:, :, 3] *= LANE_INDEX_SCALE  # 车道
        vehicle_states_denorm[:, :, 4] *= POSITION_SCALE  # 位置

        # 2. 批量编码每个车辆的特征（使用CNN或MLP）
        # vehicle_states_denorm: [batch_size, MAX_VEHICLES, 5]
        batch_vehicles = vehicle_states_denorm.view(batch_size * MAX_VEHICLES, FEATURES_PER_VEHICLE)
        encoded_vehicles = self.vehicle_encoder(batch_vehicles)  # [B*32, 128]
        encoded_vehicles = encoded_vehicles.view(batch_size, MAX_VEHICLES, 128)  # [B, 32, 128]

        # 3. 编码全局统计
        global_stats = observations['global_stats']  # [B, 16]
        encoded_global = self.global_stats_encoder(global_stats)  # [B, 128]

        # 4. 扁平化并组合所有特征
        # [B, 32, 128] -> [B, 4096]
        encoded_vehicles_flat = encoded_vehicles.view(batch_size, -1)

        # 拼接: [B, 4096] + [B, 128] = [B, 4224]
        combined_features = torch.cat([encoded_vehicles_flat, encoded_global], dim=1)

        # 通过组合器得到GNN输出
        gnn_output = self.combiner(combined_features)  # [B, 256]

        # 5. 使用简化的世界模型（可选：如果性能还是瓶颈，可以跳过）
        # 为了简化，我们创建一个伪世界模型输出
        world_features = self._simplified_world_model(gnn_output, encoded_global)

        # 6. 组合并投影
        final_combined = torch.cat([gnn_output, world_features], dim=1)  # [B, 384]
        features = self.output_projection(final_combined)  # [B, 512]

        return features

    def _simplified_world_model(self, gnn_output: torch.Tensor, global_stats: torch.Tensor) -> torch.Tensor:
        """
        简化的世界模型 - 纯GPU计算，无需LSTM

        使用MLP替代LSTM以提高性能
        """
        # 简单的线性变换来模拟世界模型的输出
        world_model_simplified = nn.Sequential(
            nn.Linear(GNN_OUTPUT_DIM + 128, 256),
            nn.ReLU(),
            nn.Linear(256, WORLD_MODEL_HIDDEN_DIM)
        ).to(gnn_output.device)

        combined = torch.cat([gnn_output, global_stats], dim=1)
        return world_model_simplified(combined)


class OptimizedActorCriticPolicy(ActorCriticPolicy):
    """
    优化的Actor-Critic策略

    主要优化：
    1. 使用优化的特征提取器（无Python循环）
    2. 批量化处理
    3. 减少CPU/GPU传输
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: callable,
        config: Dict[str, Any],
        **kwargs
    ):
        self.config = config
        self._device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # 调用父类初始化
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **kwargs
        )

        # 使用优化的特征提取器
        self.features_extractor = OptimizedTrafficControllerFeatureExtractor(
            observation_space=observation_space,
            config=config,
            device=self.device
        )

        # 更新features_dim
        self.features_dim = FEATURE_EXTRACTOR_OUTPUT_DIM

        # 重新初始化log_std
        action_dim = self.action_space.shape[0]
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32).to(self.device))

        # 设置动作分布类
        self.action_dist_cls = DiagGaussianDistribution

        # 重新构建action_net和value_net
        self.action_net = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space.shape[0])
        ).to(self.device)

        self.value_net = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)

    def _build_mlp_extractor(self) -> None:
        """设置mlp_extractor属性"""
        class FeatureExtractorInterface(nn.Module):
            def __init__(self, features_dim: int):
                super().__init__()
                self.latent_dim_pi = features_dim
                self.latent_dim_vf = features_dim

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                return features

        self.mlp_extractor = FeatureExtractorInterface(self.features_dim)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ):
        features = self.extract_features(obs)
        action_mean = self.action_net(features)

        action_dim = self.action_space.shape[0]
        distribution = self.action_dist_cls(action_dim)
        distribution = distribution.proba_distribution(action_mean, self.log_std)

        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.get_actions(distribution.distribution)

        log_prob = distribution.log_prob(actions)
        values = self.value_net(features)

        return actions, values.flatten(), log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ):
        features = self.extract_features(obs)
        action_mean = self.action_net(features)

        action_dim = self.action_space.shape[0]
        distribution = self.action_dist_cls(action_dim)
        distribution = distribution.proba_distribution(action_mean, self.log_std)

        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(features)

        return values.flatten(), log_prob, entropy

    def get_distribution(self, obs: torch.Tensor):
        features = self.extract_features(obs)
        action_mean = self.action_net(features)

        action_dim = self.action_space.shape[0]
        distribution = self.action_dist_cls(action_dim)
        return distribution.proba_distribution(action_mean, self.log_std)


def create_optimized_policy(config: Dict[str, Any]) -> type:
    """创建优化的策略类"""
    from functools import partial

    class OptimizedPolicyWithConfig(OptimizedActorCriticPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, config=config, **kwargs)

    return OptimizedPolicyWithConfig


# 别名：保持与原版兼容
create_custom_policy = create_optimized_policy
