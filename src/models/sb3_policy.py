"""
超简化PPO策略 - 无复杂依赖，纯GPU计算
专为高性能训练设计，移除所有可能导致依赖问题的组件
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import gymnasium as gym

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution

# 常量定义（内联避免导入问题）
MAX_VEHICLES = 32
FEATURES_PER_VEHICLE = 5
FEATURE_EXTRACTOR_OUTPUT_DIM = 512
ACTION_DIM = 64  # MAX_VEHICLES * 2


class SimpleFeatureExtractor(BaseFeaturesExtractor):
    """
    超简化特征提取器 - 无复杂依赖

    关键优化：
    1. 纯GPU张量运算
    2. 批量处理，无Python循环
    3. 无图构建，无复杂模型
    4. 直接从观测提取特征
    """

    def __init__(
        self,
        observation_space: gym.Space,
        config: Dict[str, Any],
        device: torch.device
    ):
        # 不调用父类的__init__，避免使用FlattenFeaturesExtractor
        nn.Module.__init__(self)

        self.observation_space = observation_space
        self._features_dim = FEATURE_EXTRACTOR_OUTPUT_DIM  # 使用私有属性
        self._device = device

        # 检查观测空间是Dict
        assert isinstance(observation_space, gym.spaces.Dict), \
            f"观测空间必须是Dict，当前是: {type(observation_space)}"

        # 1. 车辆特征编码器
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(MAX_VEHICLES * FEATURES_PER_VEHICLE, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        ).to(device)

        # 2. 全局统计编码器
        self.global_encoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        ).to(device)

        # 3. 组合编码器
        self.combiner = nn.Sequential(
            nn.Linear(256 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(512)
        ).to(device)

    @property
    def features_dim(self):
        """重写features_dim属性"""
        return self._features_dim

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播 - 完全批量化的GPU计算

        Args:
            observations: Dict with 'vehicle_states', 'global_stats', 'num_vehicles'

        Returns:
            features: [batch_size, 512]
        """
        device = observations['vehicle_states'].device

        # 批量编码车辆状态
        vehicle_features = self.vehicle_encoder(observations['vehicle_states'])  # [B, 256]

        # 批量编码全局统计
        global_features = self.global_encoder(observations['global_stats'])  # [B, 128]

        # 组合特征
        combined = torch.cat([vehicle_features, global_features], dim=1)  # [B, 384]
        features = self.combiner(combined)  # [B, 512]

        return features


class SimpleActorCriticPolicy(ActorCriticPolicy):
    """
    超简化Actor-Critic策略

    特点：
    1. 无复杂依赖
    2. 批量GPU计算
    3. 高性能
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
        # 安全的设备检测：如果请求cuda但不可用，则回退到cpu
        requested_device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        if requested_device == 'cuda' and not torch.cuda.is_available():
            print(f"⚠️  警告: 配置要求使用CUDA但PyTorch未安装CUDA支持，回退到CPU")
            requested_device = 'cpu'
        self._device = torch.device(requested_device)

        # 调用父类初始化
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **kwargs
        )

        # 使用简化的特征提取器
        self.features_extractor = SimpleFeatureExtractor(
            observation_space=observation_space,
            config=config,
            device=self._device
        )

        # 为value function使用相同的特征提取器
        self.vf_features_extractor = self.features_extractor

        # 从特征提取器获取 features_dim
        self.features_dim = self.features_extractor.features_dim

        # 重新初始化log_std
        action_dim = self.action_space.shape[0]
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32).to(self._device))

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
        ).to(self._device)

        self.value_net = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self._device)

        # 构建mlp_extractor接口（SB3需要）
        self._build_mlp_extractor()

    def _build_mlp_extractor(self) -> None:
        """设置mlp_extractor属性"""
        class FeatureExtractorInterface(nn.Module):
            def __init__(self, features_dim: int):
                super().__init__()
                self.latent_dim_pi = features_dim
                self.latent_dim_vf = features_dim

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                return features

            def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
                """Actor网络的特征提取"""
                return features

            def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
                """Critic网络的特征提取"""
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


def create_custom_policy(config: Dict[str, Any]) -> type:
    """创建简化策略类"""
    from functools import partial

    class SimplePolicyWithConfig(SimpleActorCriticPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, config=config, **kwargs)

    return SimplePolicyWithConfig
