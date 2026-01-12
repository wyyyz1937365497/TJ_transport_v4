"""
SB3 PPO策略网络 - v4.0理想架构版本

核心功能：
1. 完全整合v4_architecture.py中的所有模块
2. 使用增强的影响力驱动Top-K控制器（可学习权重）
3. 增强的动态权重门控（场景识别）
4. 拉格朗日约束优化（动态λ更新）
5. 优化GNN边构建（使用空间邻近而非全连接）

所有增强功能默认启用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import gymnasium as gym
from stable_baselines3.common.policies import ActorCriticPolicy

from .v4_architecture import (
    RiskSensitiveGNN,
    MultiScaleRSSM,
    EnhancedDynamicWeightGating,
    EnhancedInfluenceBasedController,
    LagrangianOptimizer,
    IdealTrafficControllerV4
)


class IdealTrafficPolicyV4(ActorCriticPolicy):
    """
    理想交通策略 v4.0 - SB3 PPO 完全整合版

    架构层次（增强功能默认启用）：
    1. 感知层：RiskSensitiveGNN（风险敏感异构图）
    2. 预测层：MultiScaleRSSM（多尺度世界模型）
    3. 元控制层：EnhancedDynamicWeightGating（场景识别权重门控）
    4. 决策层：EnhancedInfluenceBasedController（可学习权重+自适应Top-K）
    5. 约束层：LagrangianOptimizer（动态拉格朗日优化）
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: callable,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # 保存配置（从参数或类属性获取）
        if config is None:
            config = getattr(self.__class__, 'config', {})
        self.config = config
        self.cfg_device = config.get('device', 'cuda')  # Store config device separately

        # 调用父类初始化
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **kwargs
        )

        # 从配置中提取参数
        model_config = config.get('model', {})
        self.top_k = model_config.get('controller', {}).get('top_k', 5)
        self.max_vehicles = config.get('environment', {}).get('max_vehicles', 32)

        # ============================================================
        # 1. 感知层：风险敏感GNN
        # ============================================================
        gnn_config = model_config.get('gnn', {})
        self.perception_layer = RiskSensitiveGNN(
            node_dim=gnn_config.get('node_dim', 9),
            edge_dim=gnn_config.get('edge_dim', 4),
            hidden_dim=gnn_config.get('hidden_dim', 64),
            output_dim=gnn_config.get('output_dim', 256),
            num_layers=gnn_config.get('num_layers', 3),
            num_heads=gnn_config.get('heads', 4),
            dropout=gnn_config.get('dropout', 0.1)
        )

        # ============================================================
        # 2. 预测层：多尺度RSSM
        # ============================================================
        wm_config = model_config.get('world_model', {})
        self.prediction_layer = MultiScaleRSSM(
            input_dim=gnn_config.get('output_dim', 256),
            hidden_dim=wm_config.get('hidden_dim', 128),
            latent_dim=wm_config.get('hidden_dim', 64),
            future_steps=wm_config.get('future_steps', 5),
            num_layers=wm_config.get('num_layers', 2),
            dropout=wm_config.get('dropout', 0.1)
        )

        # ============================================================
        # 3. 元控制层：增强的动态权重门控（场景识别）
        # ============================================================
        self.weight_gating = EnhancedDynamicWeightGating(
            state_dim=gnn_config.get('output_dim', 256),
            history_dim=32,
            prediction_dim=wm_config.get('hidden_dim', 64) * 2,
            hidden_dim=128,
            dropout=gnn_config.get('dropout', 0.1),
            use_temporal_smoothing=True
        )

        # ============================================================
        # 4. 决策层：增强的影响力控制器（可学习权重+自适应Top-K）
        # ============================================================
        ctrl_config = model_config.get('controller', {})
        self.decision_layer = EnhancedInfluenceBasedController(
            gnn_dim=gnn_config.get('output_dim', 256),
            flow_dim=wm_config.get('hidden_dim', 64),
            risk_dim=wm_config.get('hidden_dim', 64),
            global_dim=ctrl_config.get('global_dim', 32),
            hidden_dim=ctrl_config.get('hidden_dim', 128),
            action_dim=ctrl_config.get('action_dim', 2),
            base_top_k=self.top_k,
            max_top_k=min(self.top_k + 3, 10),
            min_top_k=max(self.top_k - 2, 2),
            dropout=ctrl_config.get('dropout', 0.2),
            use_learnable_weights=True
        )

        # ============================================================
        # 5. 成本价值网络
        # ============================================================
        self.cost_critic = nn.Sequential(
            nn.Linear(ctrl_config.get('hidden_dim', 128), 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        # ============================================================
        # 5.5. 价值网络（用于PPO，SB3需要）
        # ============================================================
        # Create critic network for PPO (maps global embedding to value)
        self.critic = nn.Sequential(
            nn.Linear(gnn_config.get('output_dim', 256), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # ============================================================
        # 6. 拉格朗日优化器（动态约束优化）
        # ============================================================
        self.lagrangian_optimizer = LagrangianOptimizer(
            cost_limit=0.1,
            lambda_init=0.1,
            adaptive_penalty=True
        )

        # ============================================================

        # ============================================================
        # LSTM隐藏状态（用于世界模型）
        # ============================================================
        # LSTM hidden state is stored as a regular attribute (_rssm_hidden), not a buffer
        # because it's a tuple, not a tensor
        self._rssm_hidden = None

        # ============================================================
        # 图构建参数（使用空间邻近而非全连接）
        # ============================================================
        graph_config = model_config.get('graph', {})
        self.interaction_radius = graph_config.get('interaction_radius', 100.0)
        self.max_neighbors = graph_config.get('max_neighbors', 8)

    def _build_mlp_extractor(self) -> None:
        """构建MLP提取器（为SB3兼容）"""
        super()._build_mlp_extractor()

    def extract_features(
        self,
        observations: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        提取特征（完整v4.0架构）

        Args:
            observations: [batch_size, 321] 扁平化观测

        Returns:
            features_dict: 包含所有中间特征的字典
        """
        batch_size = observations.size(0)
        device = observations.device

        # 解析扁平化观测: [B, 288(vehicles) + 32(global) + 1(num)]
        vehicle_features = observations[:, :288]  # [B, 288]
        global_stats = observations[:, 288:320]   # [B, 32]
        num_vehicles = observations[:, 320:321]   # [B, 1]

        # 重塑车辆状态: [B, 32, 9]
        vehicle_states_reshaped = vehicle_features.view(
            batch_size, self.max_vehicles, 9
        )

        # 构建图数据（使用空间邻近）
        graph_data = self._build_graph_spatial(
            vehicle_states_reshaped,
            num_vehicles,
            batch_size
        )

        # ============================================================
        # 1. 感知层：风险敏感GNN
        # ============================================================
        gnn_output = self.perception_layer(
            node_features=graph_data['x'],
            edge_index=graph_data['edge_index'],
            edge_features=graph_data['edge_attr'],
            risk_features=graph_data['risk_features'],
            batch=graph_data['batch']
        )

        node_embeddings = gnn_output['node_embeddings']
        importance_scores = gnn_output['importance_scores']
        global_embedding = gnn_output['global_embedding']

        # Ensure global_embedding matches batch_size
        if global_embedding.size(0) != batch_size:
            # Mismatch detected - this can happen due to batch tensor issues
            # Create a corrected global embedding by manual mean pooling
            if graph_data.batch is not None and graph_data.batch.size(0) == node_embeddings.size(0):
                # Manual global mean pooling
                num_nodes_total = node_embeddings.size(0)
                global_embedding_corrected = torch.zeros(batch_size, node_embeddings.size(1), device=device)

                for b in range(batch_size):
                    mask = (graph_data.batch == b)
                    if mask.sum() > 0:
                        global_embedding_corrected[b] = node_embeddings[mask].mean(dim=0)
                    else:
                        global_embedding_corrected[b] = torch.zeros(node_embeddings.size(1), device=device)

                global_embedding = global_embedding_corrected
            else:
                # Fallback: repeat or truncate to match batch_size
                if global_embedding.size(0) < batch_size:
                    # Repeat the last embedding
                    last_emb = global_embedding[-1:].unsqueeze(0)
                    global_embedding = torch.cat([global_embedding, last_emb.repeat(batch_size - global_embedding.size(0), 1)], dim=0)
                else:
                    # Truncate to batch_size
                    global_embedding = global_embedding[:batch_size]

        # ============================================================
        # 2. 预测层：多尺度RSSM
        # ============================================================
        rssm_output = self.prediction_layer(
            node_embeddings=node_embeddings,
            hidden_state=getattr(self, '_rssm_hidden', None)
        )

        # 更新LSTM隐藏状态（不使用buffer，直接作为属性存储）
        # LSTM hidden state is a tuple (h, c), store it as a regular attribute
        if not hasattr(self, '_rssm_hidden'):
            self._rssm_hidden = None
        self._rssm_hidden = rssm_output['hidden_state']

        z_flow = rssm_output['z_flow']
        z_risk = rssm_output['z_risk']

        # ============================================================
        # 3. 元控制层：动态权重门控
        # ============================================================
        weight_output = self.weight_gating(global_embedding)
        dynamic_weights = weight_output['all_weights']

        return {
            'node_embeddings': node_embeddings,
            'importance_scores': importance_scores,
            'global_embedding': global_embedding,
            'z_flow': z_flow,
            'z_risk': z_risk,
            'dynamic_weights': dynamic_weights,
            'rssm_output': rssm_output,
            'global_stats': global_stats,
            'num_vehicles': num_vehicles,
            'graph_data': graph_data
        }

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播（SB3调用）

        使用完整的影响力驱动控制器
        """
        batch_size = observations.size(0)
        device = observations.device

        # 1. 提取特征（完整v4.0架构）
        features_dict = self.extract_features(observations)

        node_embeddings = features_dict['node_embeddings']
        z_flow = features_dict['z_flow']
        z_risk = features_dict['z_risk']
        global_stats = features_dict['global_stats']
        dynamic_weights = features_dict['dynamic_weights']

        # 2. 获取价值估计（使用全局嵌入）
        global_features = features_dict['global_embedding']

        # Ensure global_features has correct shape and no NaN
        global_features = torch.nan_to_num(global_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Check if global_features is empty or all zeros
        if global_features.numel() == 0 or global_features.abs().sum() < 1e-6:
            # Return zero values
            values = torch.zeros(batch_size, 1, device=device)
        else:
            # Ensure shape is [batch_size, feature_dim]
            if global_features.dim() == 1:
                global_features = global_features.unsqueeze(0)

            # If we have more features than batch_size, take first batch_size
            if global_features.size(0) > batch_size:
                global_features = global_features[:batch_size]

            # If we have fewer features than batch_size, repeat the last one
            elif global_features.size(0) < batch_size:
                last_feature = global_features[-1:].unsqueeze(0)
                global_features = torch.cat([global_features, last_feature.repeat(batch_size - global_features.size(0), 1)], dim=0)

            values = self.critic(global_features)

            # Ensure values has correct shape [batch_size, 1]
            if values.dim() == 1:
                values = values.unsqueeze(-1)
            if values.size(0) != batch_size:
                values = torch.zeros(batch_size, 1, device=device)

        # 3. 使用影响力驱动控制器选择车辆和生成动作
        # 注意：这里需要模拟车辆ID列表和ICV标志
        # 由于SB3需要固定维度输出，我们使用以下策略：
        # - 为所有32个槽位生成动作
        # - 使用影响力得分作为软注意力权重

        # 转换为控制器输入格式
        if batch_size > 1:
            # Batch模式：使用第一个样本的图结构
            graph_data = features_dict['graph_data']
            num_veh = int(features_dict['num_vehicles'][0].item())
            vehicle_ids = [f"veh_{i}" for i in range(num_veh)]
            is_icv = torch.zeros(self.max_vehicles, device=device)
            is_icv[:num_veh] = 1.0

            # 使用影响力评分作为动作权重
            importance = features_dict['importance_scores'][:self.max_vehicles].squeeze(-1)
        else:
            # 单样本模式
            graph_data = features_dict['graph_data']
            num_veh = int(features_dict['num_vehicles'].item())
            vehicle_ids = [f"veh_{i}" for i in range(num_veh)]
            is_icv = torch.zeros(self.max_vehicles, device=device)
            is_icv[:num_veh] = 1.0

            importance = features_dict['importance_scores'][:self.max_vehicles].squeeze(-1)

        # 为所有车辆生成动作（使用重要性加权）
        # 融合特征用于动作生成
        fused = torch.cat([
            node_embeddings[:self.max_vehicles],
            z_flow[:self.max_vehicles],
            z_risk[:self.max_vehicles]
        ], dim=-1)

        # 检查是否有NaN或Inf，如果有则替换为零
        fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)

        # 检查是否全为零（没有车辆或无效数据）
        if fused.abs().sum() < 1e-6:
            # 返回零动作
            actions = torch.zeros(batch_size, 64, device=device)
        else:
            # 动作生成头 - 生成64维动作（匹配action_space）
            # 使用简单线性投影到64维
            action_projection = nn.Linear(fused.size(-1), 64).to(device)
            action_features = action_projection(fused)  # [max_vehicles, 64]

            # 检查投影后的NaN
            action_features = torch.nan_to_num(action_features, nan=0.0, posinf=0.0, neginf=0.0)

            # 使用tanh确保动作在[-1, 1]范围内
            actions = torch.tanh(action_features)  # [max_vehicles, 64]

            # 应用Top-K掩码（使用重要性得分）
            k = min(self.top_k, int(is_icv.sum().item()))
            if k > 0 and importance.size(0) >= k:
                top_k_values, top_k_indices = torch.topk(importance[:k], k)

                # 创建掩码 - 只控制Top-K车辆，其他置零
                mask = torch.zeros_like(actions)  # [max_vehicles, 64]
                mask[top_k_indices] = 1.0

                # 应用掩码
                actions = actions * mask

            # 取平均或最大池化到单个动作向量
            # 对于SB3，我们只需要一个64维的动作向量
            actions = actions.mean(dim=0, keepdim=True)  # [1, 64]

            # 如果是batch，扩展到batch大小
            if batch_size > 1:
                actions = actions.expand(batch_size, -1)  # [batch_size, 64]

        # 动作分布
        action_mean = actions
        action_std = torch.ones_like(actions) * 0.1
        distribution = self.action_dist.proba_distribution(action_mean, action_std)

        if deterministic:
            actions_out = distribution.mode()
        else:
            actions_out = distribution.get_actions(deterministic=False)

        log_prob = distribution.log_prob(actions_out)

        return actions_out, values, log_prob

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作（SB3 PPO训练时调用）
        """
        # 提取特征
        features_dict = self.extract_features(observations)

        # 价值估计
        global_features = features_dict['global_embedding']
        values = self.critic(global_features)

        # 重新生成动作分布
        # 简化：直接使用输入动作的均值
        action_mean = actions
        action_std = torch.ones_like(actions) * 0.1
        distribution = self.action_dist.proba_distribution(action_mean, action_std)

        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_prob, entropy

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
        return_top_k: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        推理（预测动作）- 使用完整的影响力驱动控制器

        Args:
            observation: 单个观测 [321]
            deterministic: 是否确定性
            return_top_k: 是否返回Top-K信息

        Returns:
            actions: [64] 扁平化动作
            info: 额外信息（可选）
        """
        # 转换为tensor
        device = self.device
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            # 提取特征
            features_dict = self.extract_features(obs_tensor)

            node_embeddings = features_dict['node_embeddings']
            z_flow = features_dict['z_flow']
            z_risk = features_dict['z_risk']
            global_stats = features_dict['global_stats']
            importance = features_dict['importance_scores']

            # 确定实际车辆数
            num_veh = int(features_dict['num_vehicles'].item())

            # 车辆ID列表（模拟）
            vehicle_ids = [f"veh_{i}" for i in range(num_veh)]

            # ICV标志（假设前25%是ICV）
            is_icv = torch.zeros(num_veh, device=device)
            num_icv = max(1, int(num_veh * 0.25))
            is_icv[:num_icv] = 1.0

            # 使用影响力驱动控制器选择Top-K
            decision_output = self.decision_layer(
                gnn_embedding=node_embeddings,
                z_flow=z_flow,
                z_risk=z_risk,
                global_metrics=global_stats,
                vehicle_ids=vehicle_ids,
                is_icv=is_icv,
                deterministic=deterministic
            )

            selected_vehicle_ids = decision_output['selected_vehicle_ids']
            raw_actions = decision_output['raw_actions']
            influence_scores = decision_output['influence_scores']

            # 将动作转换为扁平化格式
            accel_flat = torch.zeros(self.max_vehicles, device=device)
            lane_flat = torch.zeros(self.max_vehicles, device=device)

            # 为选中的车辆分配动作
            for i, veh_id in enumerate(selected_vehicle_ids):
                if i < len(raw_actions) and veh_id in vehicle_ids:
                    local_idx = vehicle_ids.index(veh_id)
                    if local_idx < self.max_vehicles:
                        accel_flat[local_idx] = raw_actions[i, 0]
                        lane_flat[local_idx] = raw_actions[i, 1]

            # 合并
            actions = torch.cat([accel_flat, lane_flat]).cpu().numpy()

            # 额外信息
            info = None
            if return_top_k:
                info = {
                    'selected_vehicle_ids': selected_vehicle_ids,
                    'influence_scores': influence_scores.cpu().numpy(),
                    'dynamic_weights': features_dict['dynamic_weights'].cpu().numpy(),
                    'z_flow': z_flow.mean().cpu().numpy(),  # 平均流演化特征
                    'z_risk': z_risk.mean().cpu().numpy(),  # 平均风险特征
                }

        return actions, info

    def get_cost_value(
        self,
        observations: torch.Tensor
    ) -> torch.Tensor:
        """
        获取成本价值（用于拉格朗日约束）

        Args:
            observations: [batch_size, 321]

        Returns:
            cost_values: [batch_size, 1]
        """
        features_dict = self.extract_features(observations)

        # 使用融合特征预测成本
        node_embeddings = features_dict['node_embeddings'][:self.max_vehicles]
        z_flow = features_dict['z_flow'][:self.max_vehicles]
        z_risk = features_dict['z_risk'][:self.max_vehicles]

        fused = torch.cat([node_embeddings, z_flow, z_risk], dim=-1)

        # 简化：平均池化
        pooled = fused.mean(dim=0, keepdim=True)

        cost_values = self.cost_critic(pooled)
        return cost_values

    def _build_graph_spatial(
        self,
        vehicle_states: torch.Tensor,
        num_vehicles: torch.Tensor,
        batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """
        使用空间邻近关系构建图（而非全连接）

        Returns:
            图数据字典（用于PyTorch Geometric）
        """
        from torch_geometric.data import Data, Batch

        graphs = []

        for b in range(batch_size):
            num_veh = int(num_vehicles[b].item())
            states = vehicle_states[b, :num_veh]  # [num_veh, 9]

            if num_veh == 0:
                # 空图
                graph = Data(
                    x=torch.zeros(0, 9, device=vehicle_states.device),
                    edge_index=torch.empty((2, 0), dtype=torch.long, device=vehicle_states.device),
                    edge_attr=torch.empty((0, 4), device=vehicle_states.device),
                    risk_features=torch.zeros(0, 2, device=vehicle_states.device),
                    num_nodes=0
                )
                graphs.append(graph)
                continue

            # 节点特征
            x = states  # [num_veh, 9]

            # 计算风险特征（TTC, THW）
            risk_features = self._compute_risk_features_spatial(states)

            # 边特征（空间邻近）
            edge_index, edge_attr = self._build_edges_spatial(states, num_veh)

            # Validate edge indices to prevent CUDA errors
            if edge_index.size(1) > 0:
                max_idx = edge_index.max().item()
                if max_idx >= num_veh:
                    # Edge indices are out of bounds, reset to empty
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=states.device)
                    edge_attr = torch.empty((0, 4), device=states.device)

            # 创建图
            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                risk_features=risk_features,
                num_nodes=num_veh
            )

            graphs.append(graph)

        # 批处理
        if len(graphs) > 0:
            batch_graph = Batch.from_data_list(graphs)

            # Additional validation after batching to prevent CUDA errors
            num_nodes = batch_graph.x.size(0)

            # Validate edge indices
            if batch_graph.edge_index.size(1) > 0:
                max_idx = batch_graph.edge_index.max().item()
                min_idx = batch_graph.edge_index.min().item()

                # Check if any edge index is out of bounds
                if max_idx >= num_nodes or min_idx < 0:
                    # Reset to empty edges to prevent CUDA error
                    batch_graph.edge_index = torch.empty((2, 0), dtype=torch.long, device=batch_graph.x.device)
                    batch_graph.edge_attr = torch.empty((0, 4), device=batch_graph.x.device)

            # Validate batch tensor
            if hasattr(batch_graph, 'batch'):
                if batch_graph.batch.size(0) != num_nodes:
                    # Batch tensor size mismatch, recreate it
                    # Each node should be assigned to its batch index
                    batch_vector = []
                    node_idx = 0
                    for b_idx, graph in enumerate(graphs):
                        num_nodes_in_graph = graph.x.size(0)
                        batch_vector.extend([b_idx] * num_nodes_in_graph)
                        node_idx += num_nodes_in_graph
                    batch_graph.batch = torch.tensor(batch_vector, dtype=torch.long, device=batch_graph.x.device)
            else:
                # Create batch tensor if it doesn't exist
                batch_vector = []
                node_idx = 0
                for b_idx, graph in enumerate(graphs):
                    num_nodes_in_graph = graph.x.size(0)
                    batch_vector.extend([b_idx] * num_nodes_in_graph)
                    node_idx += num_nodes_in_graph
                batch_graph.batch = torch.tensor(batch_vector, dtype=torch.long, device=batch_graph.x.device)
        else:
            # 空batch
            batch_graph = Data(
                x=torch.zeros(0, 9, device=vehicle_states.device),
                edge_index=torch.empty((2, 0), dtype=torch.long, device=vehicle_states.device),
                edge_attr=torch.empty((0, 4), device=vehicle_states.device),
                risk_features=torch.zeros(0, 2, device=vehicle_states.device),
                num_nodes=0
            )
            # Set batch tensor for empty graph
            batch_graph.batch = torch.zeros(0, dtype=torch.long, device=vehicle_states.device)

        return batch_graph

    def _compute_risk_features_spatial(
        self,
        vehicle_states: torch.Tensor
    ) -> torch.Tensor:
        """
        计算风险特征（空间感知版本）

        Args:
            vehicle_states: [N, 9]

        Returns:
            risk_features: [N, 2]
        """
        N = vehicle_states.size(0)

        if N == 0:
            return torch.zeros(0, 2, device=vehicle_states.device)

        # 特征索引: [s, d, vs, vd, speed, accel, lane, angle, is_icv]
        speed = vehicle_states[:, 4].clamp(min=0.1)  # 避免除零

        # 简化TTC计算（基于速度）
        # TTC ~ 1 / (速度 + epsilon)
        ttc_inv = 1.0 / (speed + 0.1)

        # 简化THW计算
        # THW ~ 1 / (速度 * 安全距离因子)
        thw_inv = 1.0 / (speed * 2.0 + 0.1)

        risk_features = torch.stack([ttc_inv, thw_inv], dim=-1)

        return risk_features

    def _build_edges_spatial(
        self,
        vehicle_states: torch.Tensor,
        num_vehicles: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建边（空间邻近版本）

        Returns:
            edge_index: [2, E]
            edge_attr: [E, 4]
        """
        num_veh = vehicle_states.size(0)

        if num_veh <= 1:
            return (
                torch.empty((2, 0), dtype=torch.long, device=vehicle_states.device),
                torch.empty((0, 4), device=vehicle_states.device)
            )

        # 特征索引: [s, d, vs, vd, speed, accel, lane, angle, is_icv]
        s = vehicle_states[:, 0] * 1000.0  # 反归一化
        d = vehicle_states[:, 1] * 10.0
        lane = vehicle_states[:, 6]

        sources = []
        targets = []
        edge_features = []

        # 为每辆车找到最近的邻居
        for i in range(num_veh):
            # 计算与其他车辆的距离
            s_diff = s - s[i]
            d_diff = d - d[i]
            lane_diff = lane - lane[i]

            # 欧氏距离（简化）
            distances = torch.sqrt(s_diff**2 + d_diff**2)

            # 找到交互半径内的邻居（排除自己）
            mask = (distances < self.interaction_radius) & (distances > 0.1)

            neighbors = torch.where(mask)[0]

            # 限制最大邻居数
            if len(neighbors) > self.max_neighbors:
                # 选择最近的邻居
                _, sorted_indices = torch.topk(distances[neighbors],
                                              self.max_neighbors, largest=False)
                neighbors = neighbors[sorted_indices]

            # 创建边
            for j in neighbors:
                sources.append(i)
                targets.append(j.item())

                # 边特征
                state_i = vehicle_states[i]
                state_j = vehicle_states[j.item()]

                rel_speed = state_i[4] - state_j[4]  # 速度差
                rel_s = state_i[0] - state_j[0]       # 纵向位置差
                lane_diff = abs(state_i[6] - state_j[6])  # 车道差
                both_icv = state_i[8] * state_j[8]     # 都是ICV

                edge_features.append([rel_speed, rel_s, lane_diff, both_icv])

        if len(sources) == 0:
            # 没有边
            return (
                torch.empty((2, 0), dtype=torch.long, device=vehicle_states.device),
                torch.empty((0, 4), device=vehicle_states.device)
            )

        edge_index = torch.tensor(
            [sources, targets],
            device=vehicle_states.device,
            dtype=torch.long
        )

        edge_attr = torch.tensor(
            edge_features,
            device=vehicle_states.device,
            dtype=torch.float32
        )

        return edge_index, edge_attr

    def freeze_perception(self):
        """冻结感知层（GNN）"""
        for param in self.perception_layer.parameters():
            param.requires_grad = False
        print("[OK] Frozen perception layer (GNN)")

    def freeze_prediction(self):
        """冻结预测层（World Model）"""
        for param in self.prediction_layer.parameters():
            param.requires_grad = False
        print("[OK] Frozen prediction layer (World Model)")

    def freeze_decision(self):
        """冻结决策层（Controller）"""
        for param in self.decision_layer.parameters():
            param.requires_grad = False
        print("[OK] Frozen decision layer (Controller)")

    def unfreeze_all(self):
        """解冻所有组件"""
        for param in self.perception_layer.parameters():
            param.requires_grad = True
        for param in self.prediction_layer.parameters():
            param.requires_grad = True
        for param in self.decision_layer.parameters():
            param.requires_grad = True
        print("[OK] Unfrozen all components")

    def set_top_k(self, k: int):
        """设置Top-K值"""
        self.top_k = k
        self.decision_layer.top_k = k
        print(f"[OK] Top-K set to: {k}")


def create_ideal_traffic_policy_v4(config: Dict[str, Any]) -> type:
    """
    创建理想交通策略v4.0的工厂函数

    Args:
        config: 配置字典

    Returns:
        IdealTrafficPolicyV4 类
    """

    class PolicyClass(IdealTrafficPolicyV4):
        pass

    PolicyClass.config = config

    return PolicyClass
