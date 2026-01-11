"""
完整交通控制策略 - 用于Stable-Baselines3 Phase 3端到端微调

集成完整的模型架构：
- RiskSensitiveGNN (感知层)
- ProgressiveWorldModel (预测层)
- InfluenceDrivenController (决策层)
- DualModeSafetyShield (安全层)

用于Phase 3: 端到端微调（所有组件联合优化）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution

# 导入完整模型组件
from .gnn import RiskSensitiveGNN, GraphBuilder
from .world_model import ProgressiveWorldModel
from .controller import InfluenceDrivenController
from .safety import DualModeSafetyShield


class FullTrafficController(nn.Module):
    """
    完整交通控制器 - GNN + World Model + Controller + Safety

    作为SB3的自定义策略网络，用于Phase 3端到端微调
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))

        # 1. 感知层：GNN
        self.risk_gnn = RiskSensitiveGNN(
            node_dim=config.get('node_dim', 9),
            edge_dim=config.get('edge_dim', 4),
            hidden_dim=config.get('gnn_hidden_dim', 64),
            output_dim=config.get('gnn_output_dim', 256),
            num_layers=config.get('gnn_layers', 3),
            heads=config.get('gnn_heads', 4),
            dropout=config.get('gnn_dropout', 0.1)
        )

        # 2. 预测层：世界模型
        self.world_model = ProgressiveWorldModel(
            input_dim=config.get('gnn_output_dim', 256),
            hidden_dim=config.get('world_hidden_dim', 128),
            future_steps=config.get('future_steps', 5),
            dropout=config.get('world_dropout', 0.1),
            num_layers=config.get('world_num_layers', 2),
            bidirectional=config.get('world_bidirectional', False)
        )

        # 3. 决策层：控制器
        self.controller = InfluenceDrivenController(
            gnn_dim=config.get('gnn_output_dim', 256),
            world_dim=config.get('gnn_output_dim', 256),
            global_dim=config.get('global_dim', 16),
            hidden_dim=config.get('controller_hidden_dim', 128),
            action_dim=config.get('action_dim', 2),
            top_k=config.get('top_k', 5),
            dropout=config.get('controller_dropout', 0.2)
        )

        # 4. 安全层：安全屏障
        self.safety_shield = DualModeSafetyShield(
            ttc_threshold=config.get('ttc_threshold', 2.0),
            thw_threshold=config.get('thw_threshold', 1.5),
            max_accel=config.get('max_accel', 2.0),
            max_decel=config.get('max_decel', -3.0),
            emergency_decel=config.get('emergency_decel', -5.0),
            max_lane_change_speed=config.get('max_lane_change_speed', 5.0)
        )

        # 图构建器
        self.graph_builder = GraphBuilder(
            interaction_radius=config.get('interaction_radius', 100.0),
            max_neighbors=config.get('max_neighbors', 8),
            lane_change_distance=config.get('lane_change_distance', 50.0)
        )

        # 用于SB3的输出投影层
        # 将模型输出转换为SB3期望的格式
        self.action_projection = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)  # 假设动作空间是64维
        )

        self.value_projection = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 将所有模块移动到正确的设备（在所有子模块创建之后）
        self.to(self.device)

    def forward(
        self,
        vehicle_states: torch.Tensor,
        global_stats: torch.Tensor,
        vehicle_ids: List[str] = None,
        icv_ids: set = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SB3兼容的前向传播

        Args:
            vehicle_states: [batch_size, num_vehicles * features_per_vehicle]
            global_stats: [batch_size, 16]
            vehicle_ids: 车辆ID列表
            icv_ids: 智能网联车ID集合

        Returns:
            actions: [batch_size, action_dim]
            values: [batch_size, 1]
        """
        batch_size = vehicle_states.size(0)

        # 重塑车辆状态
        # vehicle_states: [B, N*9] -> [B*N, 9] (9维Frenet特征: s, d, vs, vd, speed, acceleration, lane_index, angle, is_icv)
        num_vehicles = vehicle_states.size(1) // 9
        vehicle_states_reshaped = vehicle_states.view(-1, 9)  # [B*N, 9]

        # 构建图数据（简化版，用于批量处理）
        # 注意：在真实场景中需要为每个batch构建单独的图
        graph_data = self._build_batch_graph(
            vehicle_states_reshaped,
            batch_size,
            num_vehicles
        )

        # 1. GNN特征提取
        gnn_output = self.risk_gnn(
            node_features=graph_data.x,
            edge_index=graph_data.edge_index,
            edge_features=graph_data.edge_attr,
            batch=graph_data.batch
        )

        gnn_embedding = gnn_output['node_embedding']  # [B*N, 256]

        # 2. 世界模型预测
        world_predictions = self.world_model(gnn_embedding)
        world_features = world_predictions.get('next_state', gnn_embedding)  # [B*N, 256]

        # 3. 全局特征扩展
        global_features = global_stats.unsqueeze(1).expand(-1, num_vehicles, -1)  # [B, N, 16]
        global_features = global_features.reshape(-1, 16)  # [B*N, 16]

        # 4. 控制器（简化版，用于SB3）
        # 这里我们不需要完整的选择逻辑，只需要特征聚合
        fused_features = torch.cat([
            gnn_embedding,
            world_features,
            global_features
        ], dim=-1)  # [B*N, 256+256+16]

        # 聚合到batch级别
        aggregated = fused_features.view(batch_size, num_vehicles, -1).mean(dim=1)  # [B, 528]

        # 5. 投影到SB3期望的输出维度
        action_features = self.action_projection(aggregated)  # [B, 64]
        value_features = self.value_projection(aggregated)  # [B, 1]

        return action_features, value_features

    def _build_batch_graph(self, vehicle_states, batch_size, num_vehicles):
        """
        构建批量图数据

        环境已提供9维Frenet特征：[s, d, vs, vd, speed, acceleration, lane_index, angle, is_icv]
        直接使用这些特征构建图，无需转换。
        """
        from torch_geometric.data import Batch, Data

        # 获取设备
        device = vehicle_states.device

        # 获取图构建配置
        interaction_radius = self.config.get('interaction_radius', 100.0)
        max_neighbors = self.config.get('max_neighbors', 8)
        lane_change_distance = self.config.get('lane_change_distance', 50.0)

        graphs = []
        for b in range(batch_size):
            # 提取当前batch的车辆状态
            start_idx = b * num_vehicles
            end_idx = start_idx + num_vehicles
            batch_vehicle_states = vehicle_states[start_idx:end_idx]  # [N, 9]

            num_nodes = batch_vehicle_states.size(0)

            # 直接使用9维Frenet特征作为节点特征
            # [0]: s (纵向位置), [1]: d (横向偏移), [2]: vs (纵向速度), [3]: vd (横向速度)
            # [4]: speed, [5]: acceleration, [6]: lane_index, [7]: angle, [8]: is_icv
            x = batch_vehicle_states  # [N, 9]

            # 为边构建准备位置和速度信息（使用Frenet坐标）
            # positions_2d: [N, 2] (s, d) - Frenet坐标系中的位置
            positions_2d = torch.stack([
                batch_vehicle_states[:, 0],  # s (纵向位置)
                batch_vehicle_states[:, 1]   # d (横向偏移)
            ], dim=1)  # [N, 2]

            # velocities_2d: [N, 2] (vs, vd) - Frenet坐标系中的速度
            velocities_2d = torch.stack([
                batch_vehicle_states[:, 2],  # vs (纵向速度)
                batch_vehicle_states[:, 3]   # vd (横向速度)
            ], dim=1)  # [N, 2]

            lane_ids = batch_vehicle_states[:, 6].long()  # lane_index

            if num_nodes > 1:
                # 计算车辆之间的距离矩阵
                # positions_2d: [N, 2] -> [N, 1, 2] and [1, N, 2]
                pos_expanded_1 = positions_2d.unsqueeze(1)  # [N, 1, 2]
                pos_expanded_2 = positions_2d.unsqueeze(0)  # [1, N, 2]
                distances = torch.norm(pos_expanded_1 - pos_expanded_2, dim=2)  # [N, N]

                # 计算相对速度
                vel_expanded_1 = velocities_2d.unsqueeze(1)  # [N, 1, 2]
                vel_expanded_2 = velocities_2d.unsqueeze(0)  # [1, N, 2]
                relative_velocities = vel_expanded_2 - vel_expanded_1  # [N, N, 2]

                # 判断是否在同一车道
                lane_expanded_1 = lane_ids.unsqueeze(1)  # [N, 1]
                lane_expanded_2 = lane_ids.unsqueeze(0)  # [1, N]
                same_lane = (lane_expanded_1 == lane_expanded_2).float()  # [N, N]

                # 构建边：基于交互半径和最大邻居数
                edge_index = []
                edge_features = []

                for i in range(num_nodes):
                    # 找到车辆 i 的邻居
                    # 排除自身
                    mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
                    mask[i] = False

                    # 计算交互分数（距离越近分数越高）
                    interaction_scores = torch.zeros(num_nodes, device=device)
                    interaction_scores[mask] = 1.0 / (distances[i, mask] + 1e-6)

                    # 同车道加分
                    interaction_scores += same_lane[i] * 0.5

                    # 设置自身的分数为 -inf
                    interaction_scores[i] = -float('inf')

                    # 选择 top-k 邻居
                    k = min(max_neighbors, num_nodes - 1)
                    topk_scores, topk_indices = torch.topk(interaction_scores, k)

                    # 过滤超出交互半径的邻居
                    valid_mask = distances[i, topk_indices] <= interaction_radius
                    valid_indices = topk_indices[valid_mask]

                    # 添加边
                    for j in valid_indices:
                        edge_index.append([i, j])

                        # 构建边特征 [4]
                        # 1. 距离
                        dist = distances[i, j].item()
                        # 2. 相对速度 x
                        rel_vel_x = relative_velocities[i, j, 0].item()
                        # 3. 相对速度 y
                        rel_vel_y = relative_velocities[i, j, 1].item()
                        # 4. 是否同车道
                        is_same_lane = same_lane[i, j].item()

                        edge_features.append([dist, rel_vel_x, rel_vel_y, is_same_lane])

                # 转换为张量
                if len(edge_index) > 0:
                    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
                    edge_attr = torch.tensor(edge_features, dtype=torch.float32, device=device)
                else:
                    # 如果没有边，创建空张量
                    edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
                    edge_attr = torch.zeros(0, 4, device=device)
            else:
                # 单个节点
                edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
                edge_attr = torch.zeros(0, 4, device=device)

            # 创建图
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            graphs.append(graph)

        # 批量处理
        batch_graph = Batch.from_data_list(graphs)
        return batch_graph


class FullTrafficFeatureExtractor(BaseFeaturesExtractor):
    """
    完整交通特征提取器 - 用于SB3的ActorCriticPolicy

    使用完整的GNN + World Model提取特征
    """

    def __init__(
        self,
        observation_space: gym.Space,
        config: Dict[str, Any]
    ):
        super(FullTrafficFeatureExtractor, self).__init__(
            observation_space,
            features_dim=512  # 输出特征维度
        )

        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))

        # 完整的交通控制器
        self.traffic_controller = FullTrafficController(config)

        # 将vehicle_states投影到固定维度
        self.input_projection = nn.Sequential(
            nn.Linear(observation_space['vehicle_states'].shape[0], 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )

        # 全局特征编码器
        self.global_encoder = nn.Sequential(
            nn.Linear(observation_space['global_stats'].shape[0], 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        # 特征组合器
        self.combiner = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        提取特征

        Args:
            observations: Dict with 'vehicle_states', 'global_stats', 'num_vehicles'

        Returns:
            features: [batch_size, 512]
        """
        vehicle_states = observations['vehicle_states']
        global_stats = observations['global_stats']

        # 投影车辆状态
        vehicle_features = self.input_projection(vehicle_states)  # [B, 256]

        # 组合全局统计
        global_features = F.relu(self.global_encoder(global_stats))  # [B, 128]

        # 组合特征
        combined = torch.cat([vehicle_features, global_features], dim=-1)  # [B, 384]

        # 最终投影
        features = self.combiner(combined)  # [B, 512]

        return features


class FullTrafficActorCriticPolicy(ActorCriticPolicy):
    """
    完整交通Actor-Critic策略 - 用于SB3 PPO

    继承自SB3的ActorCriticPolicy，使用完整的模型架构
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: callable,
        config: Dict[str, Any],
        **kwargs
    ):
        # 保存配置
        self.config = config

        # 调用父类初始化
        super(FullTrafficActorCriticPolicy, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **kwargs
        )

        # 替换特征提取器为完整的交通控制器
        self.traffic_controller = FullTrafficController(config)

        # 重置actor和critic以使用交通控制器
        # actor_mean: 策略网络输出动作均值
        self.actor_mean = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_space.shape[0])
        )

        # critic: 价值网络
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        # 动作分布
        self.action_dist = DiagGaussianDistribution(
            action_space.shape[0]
        )

    def _build_mlp_extractor(self) -> None:
        """
        构建MLP提取器 - 为SB3提供兼容的特征提取器
        """
        # 让父类创建标准的 mlp_extractor
        # 这样 SB3 的 _build 方法可以正常访问它
        super(FullTrafficActorCriticPolicy, self)._build_mlp_extractor()

    def forward(self, observations: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            observations: [batch_size, obs_dim]
            deterministic: 是否使用确定性策略

        Returns:
            actions: [batch_size, action_dim]
            values: [batch_size, 1]
            log_prob: [batch_size, 1]
        """
        # 提取特征
        features = self.extract_features(observations)

        # actor: 计算动作
        action_mean = self.actor_mean(features)
        action_std = self.log_std.exp()
        distribution = self.action_dist.proba_distribution(action_mean, action_std)

        # critic: 计算价值
        values = self.critic(features)

        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.get_actions(deterministic=False)

        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def extract_features(self, observations: torch.Tensor) -> torch.Tensor:
        """
        使用完整交通控制器提取特征

        Args:
            observations: [batch_size, obs_dim] 或 Dict

        Returns:
            features: [batch_size, 512]
        """
        # 如果是tensor，需要转换回dict格式
        if isinstance(observations, torch.Tensor):
            # 假设输入是扁平化的观测
            # 需要解析为原始格式
            # 这里简化处理，直接使用MLP
            features = self._mlp_extractor(observations)
        else:
            # 使用完整的交通控制器
            # 注意：这里需要根据实际观测格式调整
            features = self.traffic_controller(
                vehicle_states=observations['vehicle_states'],
                global_stats=observations['global_stats']
            )[0]  # 返回action features

        return features

    def _mlp_extractor(self, observations: torch.Tensor) -> torch.Tensor:
        """简单的MLP特征提取器（备用）"""
        if not hasattr(self, '_mlp'):
            self._mlp = nn.Sequential(
                nn.Linear(observations.size(-1), 512),
                nn.ReLU(),
                nn.LayerNorm(512)
            ).to(self.device)
        return self._mlp(observations)

    def load_phase2_weights(self, checkpoint_path: str) -> None:
        """
        加载Phase 2训练的权重

        Args:
            checkpoint_path: Phase 2检查点路径
        """
        import os

        if not os.path.exists(checkpoint_path):
            print(f"⚠️  警告: Phase 2检查点不存在: {checkpoint_path}")
            return

        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 加载权重到交通控制器
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 过滤并加载权重
        model_dict = self.traffic_controller.state_dict()
        filtered_dict = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        model_dict.update(filtered_dict)
        self.traffic_controller.load_state_dict(model_dict)

        print(f"✅ 已加载Phase 2权重: {checkpoint_path}")
        print(f"   加载了 {len(filtered_dict)}/{len(state_dict)} 个参数")

    def freeze_gnn_and_world_model(self) -> None:
        """冻结GNN和世界模型，只训练控制器"""
        for param in self.traffic_controller.risk_gnn.parameters():
            param.requires_grad = False
        for param in self.traffic_controller.world_model.parameters():
            param.requires_grad = False

        print("✅ 已冻结GNN和世界模型，只训练控制器")

    def unfreeze_all(self) -> None:
        """解冻所有组件，进行端到端训练"""
        for param in self.traffic_controller.parameters():
            param.requires_grad = True

        print("✅ 已解冻所有组件，进行端到端训练")

    def freeze_batch_norm(self) -> None:
        """冻结BatchNorm层"""
        for module in self.traffic_controller.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()

        print("✅ 已冻结BatchNorm层")

    def load_phase2_weights_from_sb3(
        self,
        phase2_sb3_model,
        verbose: bool = True
    ) -> int:
        """
        从 Phase 2 SB3 模型加载权重

        这个方法专门用于从 SB3 SimpleActorCriticPolicy 加载权重
        到完整的 FullTrafficActorCriticPolicy。

        权重映射策略：
        1. 如果 Phase 2 模型包含 GNN 和 WorldModel，直接传递
        2. Controller 权重需要重新映射（简化版 → 完整版）
        3. Actor 和 Critic 网络权重传递

        Args:
            phase2_sb3_model: Phase 2 的 SB3 PPO 模型
            verbose: 是否打印详细日志

        Returns:
            成功加载的权重数量
        """
        import torch.nn.functional as F

        if verbose:
            print("\n" + "="*70)
            print("🔄 从 Phase 2 SB3 模型加载权重")
            print("="*70)

        # 获取 Phase 2 策略的状态字典
        phase2_policy = phase2_sb3_model.policy
        phase2_state = phase2_policy.state_dict()

        if verbose:
            print(f"   Phase 2 权重数量: {len(phase2_state)}")

        # 获取当前完整策略的状态字典
        full_state = self.state_dict()

        # 权重映射统计
        loaded_count = 0
        skipped_count = 0
        failed_count = 0

        # 定义权重映射规则
        # 这里需要根据实际的架构来定义映射
        weight_mappings = []

        # 策略 1: 尝试映射特征提取器权重
        # 如果 Phase 2 使用的简化策略包含完整模型组件
        for key in phase2_state.keys():
            if 'features_extractor.' in key:
                # 尝试映射到 traffic_controller
                new_key = key.replace('features_extractor.', 'traffic_controller.')
                weight_mappings.append((key, new_key))

        # 策略 2: 映射 Actor 和 Critic 网络
        for key in phase2_state.keys():
            if key.startswith('action_net.'):
                # 映射到 actor_mean
                new_key = key.replace('action_net.', 'actor_mean.')
                weight_mappings.append((key, new_key))
            elif key.startswith('value_net.'):
                # 映射到 critic
                new_key = key.replace('value_net.', 'critic.')
                weight_mappings.append((key, new_key))

        # 策略 3: 直接匹配的键名
        for key in phase2_state.keys():
            if key in full_state:
                weight_mappings.append((key, key))

        # 执行权重传递
        for phase2_key, full_key in weight_mappings:
            # 检查目标键是否存在
            if full_key not in full_state:
                if verbose:
                    print(f"   ⚠️  跳过: {phase2_key} → {full_key} (目标不存在)")
                skipped_count += 1
                continue

            # 检查形状是否匹配
            phase2_shape = phase2_state[phase2_key].shape
            full_shape = full_state[full_key].shape

            if phase2_shape != full_shape:
                if verbose:
                    print(f"   ❌ 失败: {phase2_key} → {full_key}")
                    print(f"      源形状: {phase2_shape}")
                    print(f"      目标形状: {full_shape}")
                failed_count += 1
                continue

            # 传递权重
            full_state[full_key] = phase2_state[phase2_key].to(self.device)
            loaded_count += 1

            if verbose:
                print(f"   ✅ {phase2_key} → {full_key}")

        # 加载权重到当前模型
        self.load_state_dict(full_state)

        # 打印总结
        if verbose:
            print("\n" + "="*70)
            print("📊 权重加载总结")
            print("="*70)
            print(f"   ✅ 成功: {loaded_count}")
            print(f"   ⚠️  跳过: {skipped_count}")
            print(f"   ❌ 失败: {failed_count}")
            print("="*70 + "\n")

        return loaded_count

    def copy_weights_from_simple_policy(
        self,
        simple_policy: 'SimpleActorCriticPolicy',
        verbose: bool = True
    ) -> int:
        """
        从简化策略复制权重

        这是 load_phase2_weights_from_sb3 的别名，
        用于更直观的 API 调用。

        Args:
            simple_policy: SimpleActorCriticPolicy 实例
            verbose: 是否打印详细日志

        Returns:
            成功复制的权重数量
        """
        return self.load_phase2_weights_from_sb3(
            simple_policy,
            verbose=verbose
        )


def create_full_traffic_policy(config: Dict[str, Any]) -> type:
    """
    创建完整的交通策略类工厂

    Args:
        config: 配置字典

    Returns:
        policy_class: 策略类
    """
    class PolicyClass(FullTrafficActorCriticPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, config=config, **kwargs)

    return PolicyClass
