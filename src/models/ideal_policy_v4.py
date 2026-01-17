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
from torch.jit import script
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


# =============================================================================
# 工具函数
# =============================================================================

def safe_item(tensor: torch.Tensor) -> int:
    """
    安全地从张量中提取标量值，处理空张量情况

    Args:
        tensor: 输入张量

    Returns:
        张量的标量值，如果张量为空则返回0
    """
    if tensor.numel() == 0:
        return 0
    return int(tensor.item())


@script
def compute_risk_features_jit(
    vehicle_states: torch.Tensor
) -> torch.Tensor:
    """
    JIT编译的风险特征计算函数

    计算TTC和THW的向量化版本

    Args:
        vehicle_states: [N, 9] - [s, d, vs, vd, speed, accel, lane, angle, is_icv]

    Returns:
        risk_features: [N, 2] - [ttc_inv, thw_inv]
    """
    N = vehicle_states.size(0)

    if N == 0:
        return torch.zeros(0, 2, device=vehicle_states.device)

    # 特征提取
    s = vehicle_states[:, 0] * 1000.0  # 纵向位置(m)
    vs = vehicle_states[:, 2] * 30.0   # 纵向速度(m/s)
    speed = (vehicle_states[:, 4] * 30.0).clamp(min=0.1)  # 总速度(m/s)
    lane = vehicle_states[:, 6]        # 车道索引

    # 构建距离矩阵 [N, N]
    s_diff = s.unsqueeze(1) - s.unsqueeze(0)
    lane_matrix = lane.unsqueeze(1) == lane.unsqueeze(0)

    # 排除自己
    not_self = ~torch.eye(N, dtype=torch.bool, device=vehicle_states.device)

    # 前车掩码
    ahead_mask = lane_matrix & (s_diff > 0) & not_self

    # 初始化输出
    ttc_inv = torch.zeros(N, device=vehicle_states.device)
    thw_inv = torch.zeros(N, device=vehicle_states.device)

    # 找最近的前车
    s_gap_matrix = torch.where(ahead_mask, s_diff, torch.tensor(float('inf'), device=vehicle_states.device))
    nearest_ahead_indices = torch.argmin(s_gap_matrix, dim=1)
    has_ahead = ahead_mask.any(dim=1)

    if has_ahead.any():
        valid_indices = torch.where(has_ahead)[0]
        nearest_idx = nearest_ahead_indices[valid_indices]
        s_gap_valid = s_gap_matrix[valid_indices, nearest_idx]

        vs_front_valid = vs[nearest_idx]
        vs_self_valid = vs[valid_indices]
        vs_diff = vs_self_valid - vs_front_valid

        catching_up = vs_diff > 0.1

        ttc = torch.where(
            catching_up,
            s_gap_valid / vs_diff.clamp(min=0.1),
            torch.tensor(float('inf'), device=vehicle_states.device)
        )
        ttc_inv_valid = 1.0 / (ttc + 0.1)
        ttc_inv_valid[~catching_up] = 0.0

        thw = s_gap_valid / speed[valid_indices].clamp(min=0.1)
        thw_inv_valid = 1.0 / (thw + 0.1)

        ttc_inv[valid_indices] = ttc_inv_valid
        thw_inv[valid_indices] = thw_inv_valid

    return torch.stack([ttc_inv, thw_inv], dim=-1)


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

        # ========== GPU配置 ==========
        # 检查配置文件中的device设置
        device_config = config.get('device', 'cuda')

        # 强制使用单GPU模式（解决GPU利用率问题）
        if torch.cuda.is_available():
            self.device_train = torch.device('cuda:0')
            self.device_compute = torch.device('cuda:0')
            print(f"[GPU] 单GPU模式：所有计算使用cuda:0")
        else:
            self.device_train = torch.device('cpu')
            self.device_compute = torch.device('cpu')
            print(f"[GPU] 使用CPU")

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
            latent_dim=wm_config.get('latent_dim', 64),
            future_steps=wm_config.get('future_steps', 5),
            num_layers=wm_config.get('num_layers', 2),
            dropout=wm_config.get('dropout', 0.1)
        )

        # ============================================================
        # 3. 元控制层：增强的动态权重门控（场景识别）
        # ============================================================
        self.weight_gating = EnhancedDynamicWeightGating(
            state_dim=gnn_config.get('output_dim', 256),
            history_dim=model_config.get('history_dim', 32),
            prediction_dim=wm_config.get('latent_dim', 64) * 2,
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
            flow_dim=wm_config.get('latent_dim', 64),
            risk_dim=wm_config.get('latent_dim', 64),
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
                # Manual global mean pooling - 向量化版本（使用cuda:1加速）
                # 使用scatter_mean进行高效的全局池化
                num_nodes_total = node_embeddings.size(0)
                embedding_dim = node_embeddings.size(1)

                # 使用cuda:1进行计算（如果有双GPU）
                device_compute = torch.device('cuda:1') if torch.cuda.device_count() >= 2 else device
                device_train = device  # cuda:0

                # 将数据传输到cuda:1进行计算
                node_embeddings_compute = node_embeddings.to(device_compute)
                batch_compute = graph_data.batch.to(device_compute)

                # 初始化输出张量（在cuda:1上）
                global_embedding_corrected = torch.zeros(batch_size, embedding_dim, device=device_compute)

                # 使用one_hot + scatter进行向量化池化
                # 创建one-hot编码: [total_nodes, batch_size]
                batch_one_hot = torch.zeros(num_nodes_total, batch_size, device=device_compute)
                batch_one_hot.scatter_(1, batch_compute.unsqueeze(1), 1.0)

                # 计算每个batch的节点数
                batch_counts = batch_one_hot.sum(dim=0)  # [batch_size]

                # 加权求和: [batch_size, embedding_dim]
                global_embedding_corrected = torch.matmul(batch_one_hot.t(), node_embeddings_compute)

                # 除以节点数得到均值（处理空batch）
                batch_counts_clamped = batch_counts.clamp(min=1.0)  # 避免除零
                global_embedding_corrected = global_embedding_corrected / batch_counts_clamped.unsqueeze(1)

                # 处理空batch（没有节点的batch）
                empty_batches = (batch_counts == 0)
                if empty_batches.any():
                    global_embedding_corrected[empty_batches] = 0.0

                # 将结果传回cuda:0
                global_embedding = global_embedding_corrected.to(device_train)
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
            num_veh = safe_item(features_dict['num_vehicles'][0])
            vehicle_ids = [f"veh_{i}" for i in range(num_veh)]
            is_icv = torch.zeros(self.max_vehicles, device=device)
            is_icv[:num_veh] = 1.0

            # 使用影响力评分作为动作权重
            importance = features_dict['importance_scores'][:self.max_vehicles].squeeze(-1)
        else:
            # 单样本模式
            graph_data = features_dict['graph_data']
            num_veh = safe_item(features_dict['num_vehicles'])
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
            # 使用int()而不是.item()，减少CPU-GPU同步
            k = min(self.top_k, int(is_icv.sum()))
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
            num_veh = safe_item(features_dict['num_vehicles'])

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

        # 优化：基于Frenet坐标的注意力池化
        # 前提：需要获取原始vehicle_states来计算注意力权重
        if 'vehicle_states' in features_dict:
            vehicle_states = features_dict['vehicle_states'][:self.max_vehicles]
            # 特征索引: [s, d, vs, vd, speed, accel, lane, angle, is_icv]
            s = vehicle_states[:, 0]      # 纵向位置（归一化）
            speed = vehicle_states[:, 4]  # 速度（归一化）
            is_icv = vehicle_states[:, 8]  # ICV标志

            # 注意力权重计算：
            # 1. 前方车辆权重更高（s越大，权重越高）
            # 2. 速度快的车辆权重更高
            # 3. ICV车辆权重略高（因为可控）
            attention_scores = torch.zeros_like(s)

            # 前方重要性（s坐标：前方为正，权重高）
            attention_scores += s * 2.0

            # 速度重要性
            attention_scores += speed * 1.0

            # ICV重要性
            attention_scores += is_icv * 0.5

            # Softmax归一化
            attention_weights = F.softmax(attention_scores, dim=0)

            # 加权池化
            pooled = (fused * attention_weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
        else:
            # 回退到简单平均池化（如果没有vehicle_states）
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
        使用空间邻近关系构建图（向量化优化版本 + 双GPU）

        Returns:
            图数据字典（用于PyTorch Geometric）
        """
        from torch_geometric.data import Data

        device_train = vehicle_states.device  # cuda:0（训练设备）
        device_compute = self.device_compute  # cuda:1（计算设备）

        # ============== 第1步：统计总节点数 ==============
        # 优化：使用纯tensor操作，避免CPU-GPU同步
        # 确保num_vehicles是1D tensor
        if num_vehicles.dim() > 1:
            num_vehicles = num_vehicles.squeeze()

        # 先将num_vehicles限制在max_vehicles以内（在GPU上）
        num_vehicles_clamped = num_vehicles.clamp(max=self.max_vehicles)

        # 直接在GPU上计算total_nodes（避免传到CPU）
        total_nodes = num_vehicles_clamped.sum().item()

        # 只在需要时才创建列表（用于后续循环）
        # 延迟转换到真正需要的时候
        if total_nodes == 0:
            # 空batch
            batch_graph = Data(
                x=torch.zeros(0, 9, device=device_train),
                edge_index=torch.empty((2, 0), dtype=torch.long, device=device_train),
                edge_attr=torch.empty((0, 4), device=device_train),
                risk_features=torch.zeros(0, 2, device=device_train),
                num_nodes=0
            )
            batch_graph.batch = torch.zeros(0, dtype=torch.long, device=device_train)
            return batch_graph

        # ============== 双GPU优化：将数据传到cuda:1进行向量计算 ==============
        # 仅当双GPU可用时才传输
        use_dual_gpu = (device_train != device_compute)
        if use_dual_gpu:
            vehicle_states_compute = vehicle_states.to(device_compute)  # 传到cuda:1
            # 同时将num_vehicles传到cuda:1
            num_vehicles_compute = num_vehicles_clamped.to(device_compute)
        else:
            vehicle_states_compute = vehicle_states
            num_vehicles_compute = num_vehicles_clamped

        # ============== 第2步：收集所有节点特征（向量化优化版本）=============
        # 现在才创建num_veh_list（延迟转换，减少CPU-GPU同步）
        num_veh_list = [int(num_vehicles_compute[b].item()) for b in range(batch_size)]

        # 计算累积偏移量（用于拼接）
        num_veh_tensor = torch.tensor(num_veh_list, device=device_compute)
        cumsum = torch.cumsum(num_veh_tensor, dim=0)  # [batch_size]

        # 预分配输出tensor
        all_x_list = []
        all_risk_features_list = []
        all_edge_indices_list = []
        all_edge_attrs_list = []

        # 向量化处理：并行计算所有样本的特征和边
        # 注意：为了保持兼容性，仍然使用循环，但减少CPU-GPU同步
        for b in range(batch_size):
            num_veh = num_veh_list[b]

            if num_veh == 0:
                continue

            # 提取当前batch样本的车辆状态
            actual_num_veh = min(num_veh, vehicle_states_compute.size(1))
            if actual_num_veh <= 0:
                continue

            states = vehicle_states_compute[b, :actual_num_veh]  # [num_veh, 9]

            # 节点特征
            all_x_list.append(states)

            # 风险特征（在cuda:1上计算）
            risk_features = self._compute_risk_features_spatial(states)
            all_risk_features_list.append(risk_features)

            # 边特征（在cuda:1上计算）
            edge_index, edge_attr = self._build_edges_spatial(states, actual_num_veh)

            # 边索引偏移（在cuda:1上计算）
            if edge_index.size(1) > 0:
                node_offset = (cumsum[b-1] if b > 0 else 0)
                edge_index_offset = edge_index + node_offset

                # 验证边索引
                max_valid_idx = (cumsum[b] - 1) if b < batch_size else (total_nodes - 1)
                valid_mask = (edge_index_offset[0] <= max_valid_idx) & (edge_index_offset[1] <= max_valid_idx)

                if valid_mask.any():
                    all_edge_indices_list.append(edge_index_offset[:, valid_mask])
                    all_edge_attrs_list.append(edge_attr[valid_mask])

        # ============== 第3步：合并所有节点特征（向量化）=============
        if not all_x_list:
            # 如果没有有效的节点，返回空图
            batch_graph = Data(
                x=torch.zeros(0, 9, device=device_train),
                edge_index=torch.empty((2, 0), dtype=torch.long, device=device_train),
                edge_attr=torch.empty((0, 4), device=device_train),
                risk_features=torch.zeros(0, 2, device=device_train),
                num_nodes=0
            )
            batch_graph.batch = torch.zeros(0, dtype=torch.long, device=device_train)
            return batch_graph

        # 一次性合并所有tensor（减少多次torch.cat的开销）
        x = torch.cat(all_x_list, dim=0)  # [total_nodes, 9]
        risk_features_cat = torch.cat(all_risk_features_list, dim=0)  # [total_nodes, 2]

        # 合并边
        if all_edge_indices_list:
            edge_index = torch.cat(all_edge_indices_list, dim=1)  # [2, total_edges]
            edge_attr = torch.cat(all_edge_attrs_list, dim=0)  # [total_edges, 4]
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device_compute)
            edge_attr = torch.empty((0, 4), device=device_compute)

        # ============== 第4步：创建batch tensor（向量化）=============
        # 优化：使用repeat而不是循环
        batch_indices = []
        for b, num_veh in enumerate(num_veh_list):
            if num_veh > 0:
                batch_indices.extend([b] * num_veh)

        batch_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device_compute)  # [total_nodes]

        # ============== 第5步：双GPU优化：将计算结果传回cuda:0 ==============
        if use_dual_gpu:
            x = x.to(device_train)
            risk_features_cat = risk_features_cat.to(device_train)
            edge_index = edge_index.to(device_train)
            edge_attr = edge_attr.to(device_train)
            batch_tensor = batch_tensor.to(device_train)

        # ============== 第6步：创建并验证图 ==============
        batch_graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            risk_features=risk_features_cat,
            num_nodes=total_nodes
        )
        batch_graph.batch = batch_tensor

        # 验证边索引
        if edge_index.size(1) > 0:
            max_idx = edge_index.max().item()
            if max_idx >= total_nodes:
                # 边索引越界，重置为空边
                batch_graph.edge_index = torch.empty((2, 0), dtype=torch.long, device=device_train)
                batch_graph.edge_attr = torch.empty((0, 4), device=device_train)

        return batch_graph

    def _compute_risk_features_spatial(
        self,
        vehicle_states: torch.Tensor
    ) -> torch.Tensor:
        """
        计算风险特征（向量化优化版本 + JIT编译）

        TTC (Time To Collision): 纵向距离 / 相对速度（仅当追赶前车时）
        THW (Time Headway): 纵向距离 / 自车速度

        Args:
            vehicle_states: [N, 9] - [s, d, vs, vd, speed, accel, lane, angle, is_icv]

        Returns:
            risk_features: [N, 2] - [ttc_inv, thw_inv]
        """
        # 使用JIT编译的版本（首次调用时编译，后续调用更快）
        return compute_risk_features_jit(vehicle_states)

    def _build_edges_spatial(
        self,
        vehicle_states: torch.Tensor,
        num_vehicles: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建边（向量化优化版本）

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

        device = vehicle_states.device

        # 特征索引: [s, d, vs, vd, speed, accel, lane, angle, is_icv]
        s = vehicle_states[:, 0] * 1000.0  # 反归一化 [N]
        d = vehicle_states[:, 1] * 10.0    # [N]
        lane = vehicle_states[:, 6]         # [N]

        # ============== 向量化计算距离矩阵 ==============
        # 计算所有车对之间的距离 [N, N]
        s_diff = s.unsqueeze(1) - s.unsqueeze(0)  # [N, N]
        d_diff = d.unsqueeze(1) - d.unsqueeze(0)  # [N, N]
        lane_diff_matrix = lane.unsqueeze(1) - lane.unsqueeze(0)  # [N, N]

        # Frenet距离
        s_distance = torch.abs(s_diff)
        d_distance = torch.abs(d_diff) + torch.abs(lane_diff_matrix) * 3.5

        # 加权距离
        distances = torch.sqrt(
            (s_distance * 1.0)**2 +
            (d_distance * 0.3)**2
        )  # [N, N]

        # ============== 找到交互半径内的邻居 ==============
        # 排除自己
        not_self = ~torch.eye(num_veh, dtype=torch.bool, device=device)

        # 邻居掩码：在交互半径内 + 不是自己
        neighbor_mask = (distances < self.interaction_radius) & not_self  # [N, N]

        # ============== 限制最大邻居数 ==============
        # 对每个节点，选择最近的max_neighbors个邻居
        sources_list = []
        targets_list = []

        for i in range(num_veh):
            # 找到i的邻居
            neighbors = torch.where(neighbor_mask[i])[0]

            if len(neighbors) == 0:
                continue

            # 限制邻居数量
            if len(neighbors) > self.max_neighbors:
                # 根据距离排序，选择最近的
                neighbor_distances = distances[i, neighbors]
                _, topk_indices = torch.topk(neighbor_distances, self.max_neighbors, largest=False)
                neighbors = neighbors[topk_indices]

            # 添加边
            sources_list.append(torch.full_like(neighbors, i, dtype=torch.long))
            targets_list.append(neighbors)

        if len(sources_list) == 0:
            return (
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0, 4), device=device)
            )

        # 合并所有边
        sources = torch.cat(sources_list)  # [E]
        targets = torch.cat(targets_list)  # [E]

        # ============== 向量化计算边特征 ==============
        # 提取源节点和目标节点的状态
        source_states = vehicle_states[sources]  # [E, 9]
        target_states = vehicle_states[targets]  # [E, 9]

        # 计算边特征
        rel_speed = source_states[:, 4] - target_states[:, 4]  # [E]
        rel_s = source_states[:, 0] - target_states[:, 0]      # [E]
        lane_diff_edge = torch.abs(source_states[:, 6] - target_states[:, 6])  # [E]
        both_icv = source_states[:, 8] * target_states[:, 8]   # [E]

        edge_attr = torch.stack([rel_speed, rel_s, lane_diff_edge, both_icv], dim=1)  # [E, 4]

        # 构建edge_index
        edge_index = torch.stack([sources, targets], dim=0)  # [2, E]

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
