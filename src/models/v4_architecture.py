"""
理想架构 v4.0 - 基于约束的分层多智能体世界模型（增强版）

核心设计理念：
1. 感知层：风险敏感异构 GNN (Risk-Sensitive Hetero-GNN)
2. 预测层：潜在状态空间模型 (RSSM) + 多尺度解耦
3. 决策层：影响力驱动 + 动态权重约束 + 稀疏控制
4. 约束层：双模态安全屏障
5. 增强功能：拉格朗日约束优化、可学习Top-K权重、场景识别

对应赛题三大挑战：
- 挑战一（结构化表征）: 风险敏感异构图
- 挑战二（前瞻性预测）: 多尺度世界模型
- 挑战三（协同决策）: Top-K影响力 + 成本约束

增强功能（通过use_enhanced_features控制）：
- 强化拉格朗日约束机制（动态λ更新）
- 可学习的Top-K权重（α、β参数）
- 增强的动态权重门控（场景识别）
- 自适应Top-K值调整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
import numpy as np


# =============================================================================
# 第一层：风险敏感异构GNN (感知层)
# =============================================================================

class RiskSensitiveGNN(nn.Module):
    """
    风险敏感异构图神经网络

    核心创新：
    1. 节点特征融入TTC（碰撞时间）和THW（车头时距）
    2. Biased Attention：高风险边强制高权重
    3. 层次化聚合：局部 -> 区域 -> 全局
    """

    def __init__(
        self,
        node_dim: int = 9,
        edge_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # 节点特征编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 边特征编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 多层GNN（使用PyTorch Geometric）
        try:
            from torch_geometric.nn import GATConv
            self.gnn_layers = nn.ModuleList([
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                        edge_dim=hidden_dim, dropout=dropout, concat=True)
                for _ in range(num_layers)
            ])
            self.use_pyg = True
        except ImportError:
            # Fallback: 手动实现注意力GNN
            print("[WARNING] PyTorch Geometric not installed, using simplified GNN")
            self.gnn_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                for _ in range(num_layers)
            ])
            self.use_pyg = False

        # 风险感知注意力偏置
        self.risk_bias = nn.Sequential(
            nn.Linear(2, hidden_dim),  # TTC, THW -> 偏置
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 层次化池化
        # GAT with concat=True outputs hidden_dim, not hidden_dim * num_heads
        self.local_pool = nn.Linear(hidden_dim, hidden_dim)
        self.global_pool = nn.Linear(hidden_dim, output_dim)

        # 关键性评分头（用于Top-K选择）
        self.importance_scorer = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        node_features: torch.Tensor,  # [N, node_dim]
        edge_index: torch.Tensor,      # [2, E]
        edge_features: torch.Tensor,   # [E, edge_dim]
        risk_features: torch.Tensor,   # [N, 2] - TTC倒数, THW倒数
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Returns:
            node_embeddings: [N, output_dim] 节点嵌入
            importance_scores: [N, 1] 节点重要性
            global_embedding: [B, output_dim] 全局嵌入（如果提供batch）
        """
        # 1. 编码节点和边特征
        h = self.node_encoder(node_features)  # [N, hidden_dim]
        e = self.edge_encoder(edge_features)   # [E, hidden_dim]

        # 2. 提取风险偏置
        risk_bias = self.risk_bias(risk_features)  # [N, 1]

        # 3. GNN层传播
        if self.use_pyg:
            for gnn_layer in self.gnn_layers:
                # PyG版本
                h = gnn_layer(h, edge_index, e)
                h = F.relu(h)
        else:
            # 简化版本
            for layer in self.gnn_layers:
                # 聚合邻居特征
                row, col = edge_index
                neighbor_features = h[col]  # [E, hidden_dim]

                # 拼接当前节点和邻居
                combined = torch.cat([h[row], neighbor_features], dim=-1)
                h_new = layer(combined)

                # 更新
                h = h_new + h  # 残差连接

        # 4. 应用风险偏置
        h = h + risk_bias  # 广播风险偏置

        # 5. 计算节点嵌入
        if self.use_pyg:
            h_pooled = self.local_pool(h)
        else:
            h_pooled = h

        node_embeddings = self.global_pool(h_pooled)

        # 6. 计算重要性得分
        importance_scores = self.importance_scorer(node_embeddings)

        # 7. 全局池化（如果提供了batch）
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            global_embedding = global_mean_pool(node_embeddings, batch)
        else:
            global_embedding = node_embeddings.mean(dim=0, keepdim=True)

        return {
            'node_embeddings': node_embeddings,
            'importance_scores': importance_scores,
            'global_embedding': global_embedding,
            'risk_features': risk_features
        }


# =============================================================================
# 第二层：多尺度RSSM世界模型 (预测层)
# =============================================================================

class MultiScaleRSSM(nn.Module):
    """
    多尺度潜在状态空间模型

    核心创新：
    1. 强制解耦 z = [z_flow, z_risk]
    2. z_flow: 流演化预测（速度、密度）
    3. z_risk: 风险演化预测（冲突概率）
    """

    def __init__(
        self,
        input_dim: int = 256,    # GNN输出
        hidden_dim: int = 128,    # LSTM隐藏维度
        latent_dim: int = 64,     # 潜在状态维度
        future_steps: int = 5,    # 预测未来步数
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.future_steps = future_steps
        self.num_layers = num_layers

        # 共享编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM核心
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 解耦预测头
        self.flow_predictor = nn.Sequential(  # z_flow: 流演化
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.risk_predictor = nn.Sequential(  # z_risk: 风险演化
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # 速度预测器（用于监督学习）
        self.speed_predictor = nn.Linear(hidden_dim, 1)

        # 位置预测器（用于监督学习）
        self.position_predictor = nn.Linear(hidden_dim, 2)

        # 冲突预测器（用于风险监督）
        self.conflict_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,  # [B, N, input_dim] 或 [N, input_dim]
        hidden_state: Optional[Tuple] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Returns:
            z_flow: [B*N, latent_dim] 流演化特征
            z_risk: [B*N, latent_dim] 风险演化特征
            next_states: 预测的下一状态
            conflict_probs: 冲突概率
            new_hidden: LSTM隐藏状态
        """
        # 确保输入是3D
        if node_embeddings.dim() == 2:
            node_embeddings = node_embeddings.unsqueeze(0)  # [1, N, D]

        batch_size, num_nodes, _ = node_embeddings.shape

        # 处理空车辆的情况
        if num_nodes == 0:
            device = node_embeddings.device
            return {
                'z_flow': torch.zeros(batch_size * num_nodes, self.latent_dim, device=device),
                'z_risk': torch.zeros(batch_size * num_nodes, self.latent_dim, device=device),
                'speed_pred': torch.zeros(batch_size * num_nodes, 1, device=device),
                'position_pred': torch.zeros(batch_size * num_nodes, 2, device=device),
                'conflict_prob': torch.zeros(batch_size * num_nodes, 1, device=device),
                'hidden_state': hidden_state
            }

        # 检查hidden_state的batch维度是否匹配
        if hidden_state is not None:
            # hidden_state是tuple (h, c)，每个shape是 [num_layers, batch_size, hidden_dim]
            h_hidden, c_hidden = hidden_state
            expected_batch_size = num_nodes  # LSTM处理的是序列维度，即num_nodes
            if h_hidden.size(1) != expected_batch_size:
                # Batch size不匹配，重置hidden_state
                hidden_state = None

        # 编码
        h = self.encoder(node_embeddings)  # [B, N, hidden_dim]

        # LSTM处理（展平为序列）
        h_flat = h.view(batch_size, num_nodes, -1)
        h_flat = h_flat.permute(1, 0, 2)  # [N, B, hidden_dim]

        lstm_out, new_hidden = self.lstm(h_flat, hidden_state)

        # 恢复形状
        lstm_out = lstm_out.permute(1, 0, 2)  # [B, N, hidden_dim]
        lstm_out = lstm_out.reshape(batch_size * num_nodes, -1)

        # 解耦预测
        z_flow = self.flow_predictor(lstm_out)    # 流演化
        z_risk = self.risk_predictor(lstm_out)    # 风险演化

        # 附加预测头
        speed_pred = self.speed_predictor(lstm_out)
        pos_pred = self.position_predictor(lstm_out)
        conflict_prob = self.conflict_predictor(lstm_out)

        return {
            'z_flow': z_flow,
            'z_risk': z_risk,
            'speed_pred': speed_pred,
            'position_pred': pos_pred,
            'conflict_prob': conflict_prob,
            'hidden_state': new_hidden
        }


# =============================================================================
# 第三层：动态权重门控网络 (决策层辅助)
# =============================================================================

class DynamicWeightGating(nn.Module):
    """
    动态权重门控网络

    根据当前交通状态动态调整效率、稳定性、成本的权重

    输入：全局交通状态
    输出：w_eff, w_stab, w_cost (Softmax归一化)
    """

    def __init__(
        self,
        state_dim: int = 64,  # 全局状态维度
        hidden_dim: int = 32
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # 3个权重
        )

    def forward(self, global_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            global_state: [B, state_dim] 全局交通状态

        Returns:
            weights: [B, 3] Softmax归一化的权重 [w_eff, w_stab, w_cost]
        """
        logits = self.mlp(global_state)
        weights = F.softmax(logits, dim=-1)

        return {
            'w_eff': weights[..., 0:1],
            'w_stab': weights[..., 1:2],
            'w_cost': weights[..., 2:3],
            'all_weights': weights
        }


# =============================================================================
# 第三层：影响力驱动Top-K控制器 (决策层核心)
# =============================================================================

class InfluenceBasedController(nn.Module):
    """
    基于影响力的Top-K控制器

    核心设计：
    1. 影响力评分 = α·Importance(GNN) + β·Impact(Predicted)
    2. Top-K选择：只控制K辆关键车
    3. 动作生成：加速度 + 换道概率
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        flow_dim: int = 64,
        risk_dim: int = 64,
        global_dim: int = 32,
        hidden_dim: int = 128,
        action_dim: int = 2,
        top_k: int = 5,
        dropout: float = 0.2
    ):
        super().__init__()

        self.gnn_dim = gnn_dim
        self.flow_dim = flow_dim
        self.risk_dim = risk_dim
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.top_k = top_k

        # 全局上下文编码器
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )

        # 特征融合层
        fusion_input = gnn_dim + flow_dim + risk_dim + 32
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input, 384),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(384),
            nn.Linear(384, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # 影响力评分网络（改进版）
        self.influence_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 归一化到[0,1]
        )

        # 价值网络（三头：Q, Cost, Advantage）
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.cost_value_network = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.advantage_network = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # 动作生成网络
        self.action_generator = nn.ModuleDict({
            'acceleration': nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Tanh()  # [-1,1] -> 映射到[-3,2] m/s²
            ),
            'lane_change': nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # [0,1] -> 换道概率
            )
        })

    def forward(
        self,
        gnn_embedding: torch.Tensor,
        z_flow: torch.Tensor,
        z_risk: torch.Tensor,
        global_metrics: torch.Tensor,
        vehicle_ids: List[str],
        is_icv: torch.Tensor,
        alpha: float = 0.6,
        beta: float = 0.4,
        deterministic: bool = False
    ) -> Dict[str, Any]:
        """
        前向传播（带Top-K选择）

        Returns:
            selected_vehicle_ids: 选中的K辆车
            raw_actions: 对应的动作
            influence_scores: 影响力得分
            ...
        """
        device = gnn_embedding.device
        batch_size = gnn_embedding.size(0)

        # 1. 编码全局上下文
        global_context = self.global_encoder(global_metrics)  # [B, 32]

        # 2. 扩展全局特征到每个车辆
        if batch_size > 1:
            global_context = global_context.expand(-1, gnn_embedding.size(1), -1)
            global_context = global_context.reshape(-1, 32)
        else:
            global_context = global_context.squeeze(0)

        # 3. 特征融合
        fused = torch.cat([
            gnn_embedding.reshape(-1, self.gnn_dim),
            z_flow.reshape(-1, self.flow_dim),
            z_risk.reshape(-1, self.risk_dim),
            global_context
        ], dim=-1)

        fused_features = self.fusion_layer(fused)  # [N, hidden_dim]

        # 4. 过滤ICV车辆
        icv_mask = is_icv.bool()
        icv_indices = torch.where(icv_mask)[0]

        if len(icv_indices) == 0:
            return {
                'selected_vehicle_ids': [],
                'raw_actions': torch.zeros(0, self.action_dim, device=device),
                'influence_scores': torch.zeros(0, device=device),
                'value_estimates': torch.zeros(0, device=device),
                'cost_estimates': torch.zeros(0, device=device),
                'advantage_estimates': torch.zeros(0, device=device)
            }

        # 5. 计算影响力得分
        icv_features = fused_features[icv_mask]
        influence_scores = self.influence_scorer(icv_features).squeeze(-1)

        # 6. Top-K选择
        k = min(self.top_k, len(icv_indices))
        top_k_scores, top_k_local_indices = torch.topk(
            influence_scores, k, largest=True, sorted=True
        )

        selected_indices = icv_indices[top_k_local_indices]
        selected_vehicle_ids = [vehicle_ids[i] for i in selected_indices.cpu().numpy()]

        # 7. 为选中的车辆生成动作
        selected_features = fused_features[selected_indices]

        accel_actions = self.action_generator['acceleration'](selected_features)
        lane_actions = self.action_generator['lane_change'](selected_features)

        raw_actions = torch.cat([accel_actions, lane_actions], dim=-1)

        # 8. 价值估计
        value_estimates = self.value_network(fused_features).squeeze(-1)
        cost_estimates = self.cost_value_network(fused_features).squeeze(-1)
        advantage_estimates = self.advantage_network(fused_features).squeeze(-1)

        return {
            'selected_vehicle_ids': selected_vehicle_ids,
            'selected_indices': selected_indices.cpu().numpy().tolist(),
            'raw_actions': raw_actions,
            'influence_scores': influence_scores,
            'top_k_scores': top_k_scores,
            'value_estimates': value_estimates,
            'cost_estimates': cost_estimates,
            'advantage_estimates': advantage_estimates
        }


# =============================================================================
# 第四层：双模态安全屏障 (约束层)
# =============================================================================

class SafetyBarrier(nn.Module):
    """
    双模态安全屏障

    Level 1: 规则卫士 - 简单约束检查
    Level 2: 紧急避险 - TTC<2s 强制制动
    """

    def __init__(
        self,
        max_accel: float = 2.0,
        max_decel: float = -3.0,
        emergency_decel: float = -5.0,
        safe_ttc_threshold: float = 2.0,  # 秒
        min_speed: float = 0.0
    ):
        super().__init__()

        self.max_accel = max_accel
        self.max_decel = max_decel
        self.emergency_decel = emergency_decel
        self.safe_ttc_threshold = safe_ttc_threshold
        self.min_speed = min_speed

    def forward(
        self,
        actions: torch.Tensor,  # [N, 2] - [accel, lane_change]
        vehicle_states: List[Dict[str, Any]],
        ttc_values: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        应用安全屏障

        Returns:
            safe_actions: [N, 2] 修正后的动作
            barrier_info: 安全屏障触发信息
        """
        safe_actions = actions.clone()
        barrier_triggered = {
            'level1_count': 0,
            'level2_count': 0,
            'emergency_vehicles': []
        }

        for i, (action, state) in enumerate(zip(actions, vehicle_states)):
            # Level 1: 基本约束检查
            accel = action[0].item()

            # 限制加速度范围
            if accel > self.max_accel:
                safe_actions[i, 0] = self.max_accel
                barrier_triggered['level1_count'] += 1
            elif accel < self.max_decel:
                safe_actions[i, 0] = self.max_decel
                barrier_triggered['level1_count'] += 1

            # 确保速度不会变为负数
            current_speed = state.get('speed', 0.0)
            predicted_speed = current_speed + accel * 0.1  # 0.1秒步长
            if predicted_speed < self.min_speed:
                safe_actions[i, 0] = max(self.max_decel, -current_speed / 0.1)

            # Level 2: TTC紧急检查
            if ttc_values is not None and ttc_values[i] < self.safe_ttc_threshold:
                # 强制制动
                safe_actions[i, 0] = self.emergency_decel
                barrier_triggered['level2_count'] += 1
                barrier_triggered['emergency_vehicles'].append(i)

        return safe_actions, barrier_triggered


# =============================================================================
# 完整的v4.0架构
# =============================================================================

class IdealTrafficControllerV4(nn.Module):
    """
    理想交通控制器 v4.0

    完整架构：
    1. Risk-Sensitive GNN (感知)
    2. Multi-Scale RSSM (预测)
    3. Dynamic Weight Gating (元控制)
    4. Influence-Based Controller (决策)
    5. Safety Barrier (约束)
    """

    def __init__(
        self,
        node_dim: int = 9,
        edge_dim: int = 4,
        global_dim: int = 32,
        gnn_hidden_dim: int = 64,
        gnn_output_dim: int = 256,
        rssm_hidden_dim: int = 128,
        rssm_latent_dim: int = 64,
        controller_hidden_dim: int = 128,
        top_k: int = 5,
        dropout: float = 0.2,
        device: str = 'cuda'
    ):
        super().__init__()

        self.device = torch.device(device)
        self.top_k = top_k

        # 1. 感知层：风险敏感GNN
        self.perception_layer = RiskSensitiveGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            dropout=dropout
        )

        # 2. 预测层：多尺度RSSM
        self.prediction_layer = MultiScaleRSSM(
            input_dim=gnn_output_dim,
            hidden_dim=rssm_hidden_dim,
            latent_dim=rssm_latent_dim,
            dropout=dropout
        )

        # 3. 决策层：影响力控制器
        self.decision_layer = InfluenceBasedController(
            gnn_dim=gnn_output_dim,
            flow_dim=rssm_latent_dim,
            risk_dim=rssm_latent_dim,
            global_dim=global_dim,
            hidden_dim=controller_hidden_dim,
            top_k=top_k,
            dropout=dropout
        )

        # 4. 动态权重门控
        self.weight_gating = DynamicWeightGating(
            state_dim=gnn_output_dim
        )

        # 5. 安全屏障
        self.safety_barrier = SafetyBarrier()

        # 初始化隐藏状态
        self.register_buffer('rssm_hidden', None)

    def forward(
        self,
        observation: Dict[str, Any],
        deterministic: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4
    ) -> Dict[str, Any]:
        """
        完整前向传播

        Returns:
            包含所有中间结果和最终动作的字典
        """
        # 1. 感知层：风险敏感GNN
        gnn_output = self.perception_layer(
            node_features=observation['node_features'],
            edge_index=observation['edge_index'],
            edge_features=observation['edge_features'],
            risk_features=observation['risk_features'],
            batch=observation.get('batch')
        )

        node_embeddings = gnn_output['node_embeddings']
        importance_scores = gnn_output['importance_scores']
        global_embedding = gnn_output['global_embedding']

        # 2. 预测层：多尺度RSSM
        rssm_output = self.prediction_layer(
            node_embeddings=node_embeddings,
            hidden_state=None  # 或传入之前的隐藏状态
        )

        z_flow = rssm_output['z_flow']
        z_risk = rssm_output['z_risk']

        # 3. 动态权重门控
        weight_output = self.weight_gating(global_embedding)
        dynamic_weights = weight_output['all_weights']

        # 4. 决策层：Top-K控制器
        decision_output = self.decision_layer(
            gnn_embedding=node_embeddings,
            z_flow=z_flow,
            z_risk=z_risk,
            global_metrics=observation['global_metrics'],
            vehicle_ids=observation['vehicle_ids'],
            is_icv=observation['is_icv'],
            alpha=alpha,
            beta=beta,
            deterministic=deterministic
        )

        # 5. 安全屏障（如果有车辆状态）
        safe_actions = decision_output['raw_actions']
        barrier_info = {}

        if 'vehicle_states' in observation:
            safe_actions, barrier_info = self.safety_barrier(
                actions=decision_output['raw_actions'],
                vehicle_states=observation['vehicle_states'],
                ttc_values=observation.get('ttc_values')
            )

        return {
            # 最终输出
            'selected_vehicle_ids': decision_output['selected_vehicle_ids'],
            'safe_actions': safe_actions,
            'influence_scores': decision_output['influence_scores'],

            # 中间结果（用于分析）
            'gnn_output': gnn_output,
            'rssm_output': rssm_output,
            'dynamic_weights': dynamic_weights,
            'decision_output': decision_output,
            'barrier_info': barrier_info
        }

    def compute_loss(
        self,
        batch: Dict[str, Any],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失

        损失组成：
        1. 预测损失：轨迹预测 + 风险预测
        2. 价值损失：Q值 + 成本值
        3. 影响力损失：鼓励稀疏选择
        """
        # 前向传播
        output = self.forward(batch, deterministic=False)

        losses = {}

        # 1. 预测损失
        speed_pred = output['rssm_output']['speed_pred']
        speed_target = targets.get('speed_target')
        if speed_target is not None:
            losses['speed_mse'] = F.mse_loss(speed_pred, speed_target)

        # 2. 风险预测损失
        conflict_prob = output['rssm_output']['conflict_prob']
        conflict_target = targets.get('conflict_target')
        if conflict_target is not None:
            losses['conflict_bce'] = F.binary_cross_entropy(
                conflict_prob.squeeze(-1),
                conflict_target.float()
            )

        # 3. 价值损失（如果提供）
        value_estimates = output['decision_output']['value_estimates']
        value_target = targets.get('value_target')
        if value_target is not None:
            losses['value_loss'] = F.mse_loss(
                value_estimates,
                value_target
            )

        # 4. 成本价值损失
        cost_estimates = output['decision_output']['cost_estimates']
        cost_target = targets.get('cost_target')
        if cost_target is not None:
            losses['cost_value_loss'] = F.mse_loss(
                cost_estimates,
                cost_target
            )

        # 5. 稀疏性损失（鼓励只选择少数车辆）
        influence_scores = output['influence_scores']
        selected_count = len(output['selected_vehicle_ids'])
        total_count = len(batch['vehicle_ids'])
        sparsity_loss = (selected_count / max(total_count, 1)) - 0.15  # 目标15%
        losses['sparsity'] = torch.tensor(sparsity_loss, device=self.device)

        return losses


# =============================================================================
# 增强功能模块（可选启用）
# =============================================================================

class LagrangianOptimizer:
    """
    拉格朗日优化器（增强版）

    实现了基于真实成本违反的动态约束机制
    """

    def __init__(
        self,
        cost_limit: float = 0.1,
        lambda_init: float = 0.1,
        lambda_max: float = 10.0,
        lambda_lr: float = 0.01,
        adaptive_penalty: bool = True,
        violation_tolerance: float = 0.01
    ):
        self.cost_limit = cost_limit
        self.lambda_init = lambda_init
        self.lambda_max = lambda_max
        self.lambda_lr = lambda_lr
        self.adaptive_penalty = adaptive_penalty
        self.violation_tolerance = violation_tolerance

        self.lambda_param = lambda_init
        self.cost_history = []
        self.violation_history = []

    def update_lambda(self, cost_violation: float, smooth_update: bool = True) -> float:
        """更新拉格朗日乘子"""
        self.cost_history.append(cost_violation + self.cost_limit)
        self.violation_history.append(cost_violation)

        if self.adaptive_penalty:
            if len(self.violation_history) > 10:
                recent_violations = self.violation_history[-10:]
                if all(v > 0 for v in recent_violations):
                    adaptive_lr = self.lambda_lr * 2.0
                elif all(v < -self.violation_tolerance for v in recent_violations):
                    adaptive_lr = self.lambda_lr * 0.5
                else:
                    adaptive_lr = self.lambda_lr
            else:
                adaptive_lr = self.lambda_lr
        else:
            adaptive_lr = self.lambda_lr

        if smooth_update:
            delta = adaptive_lr * max(0, cost_violation)
            self.lambda_param = np.clip(self.lambda_param + delta, 0.0, self.lambda_max)
        else:
            if cost_violation > 0:
                self.lambda_param = np.clip(self.lambda_param * 1.1, 0.0, self.lambda_max)
            elif cost_violation < -self.violation_tolerance:
                self.lambda_param = np.clip(self.lambda_param * 0.95, 0.0, self.lambda_max)

        return self.lambda_param

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'lambda_param': self.lambda_param,
            'avg_violation': np.mean(self.violation_history) if self.violation_history else 0.0,
            'violation_rate': np.mean([v > 0 for v in self.violation_history]) if self.violation_history else 0.0
        }


class EnhancedDynamicWeightGating(nn.Module):
    """
    增强的动态权重门控网络

    核心改进：
    1. 场景识别（平峰、早高峰、晚高峰、拥堵）
    2. 多输入融合（全局状态、历史统计、预测信息）
    3. 时间平滑（避免权重突变）
    """

    def __init__(
        self,
        state_dim: int = 256,
        history_dim: int = 32,
        prediction_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_temporal_smoothing: bool = True,
        smoothing_factor: float = 0.7
    ):
        super().__init__()

        self.use_temporal_smoothing = use_temporal_smoothing
        self.smoothing_factor = smoothing_factor
        self.scene_names = ['平峰', '早高峰', '晚高峰', '拥堵']

        # 全局状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 历史统计编码器
        self.history_encoder = nn.Sequential(
            nn.Linear(history_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # 预测信息编码器
        self.prediction_encoder = nn.Sequential(
            nn.Linear(prediction_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # 场景分类器
        total_dim = hidden_dim + 32 + 32
        self.scene_classifier = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Softmax(dim=-1)
        )

        # 场景特定基线权重
        self.scene_baselines = nn.Parameter(
            torch.tensor([
                [0.6, 0.3, 0.1],  # 平峰
                [0.5, 0.4, 0.1],  # 早高峰
                [0.5, 0.4, 0.1],  # 晚高峰
                [0.3, 0.5, 0.2],  # 拥堵
            ])
        )

        # 场景自适应调整
        self.adapter = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Tanh()
        )

        self.register_parameter('adaptation_strength', nn.Parameter(torch.tensor(0.2)))
        self.register_buffer('prev_weights', None)

    def forward(
        self,
        global_state: torch.Tensor,
        history_stats: Optional[torch.Tensor] = None,
        prediction_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播"""
        batch_size = global_state.size(0)
        device = global_state.device

        # 编码各输入源
        state_encoded = self.state_encoder(global_state)

        if history_stats is None:
            history_encoded = torch.zeros(batch_size, 32, device=device)
        else:
            history_encoded = self.history_encoder(history_stats)

        if prediction_features is None:
            prediction_encoded = torch.zeros(batch_size, 32, device=device)
        else:
            prediction_encoded = self.prediction_encoder(prediction_features)

        # 融合
        fused = torch.cat([state_encoded, history_encoded, prediction_encoded], dim=-1)

        # 场景识别
        scene_probs = self.scene_classifier(fused)

        # 场景基线权重
        baseline_weights = torch.matmul(scene_probs, self.scene_baselines)

        # 自适应调整
        adaptation = self.adapter(fused)
        adaptation_strength = torch.sigmoid(self.adaptation_strength)

        adjusted_weights = baseline_weights + adaptation_strength * adaptation
        adjusted_weights = F.softplus(adjusted_weights)
        weights = adjusted_weights / adjusted_weights.sum(dim=-1, keepdim=True)

        # 时间平滑
        if self.use_temporal_smoothing and self.prev_weights is not None:
            # Check if batch sizes match
            if self.prev_weights.size(0) == weights.size(0):
                # Batch sizes match, apply smoothing
                weights = (
                    self.smoothing_factor * self.prev_weights +
                    (1 - self.smoothing_factor) * weights
                )
            else:
                # Batch sizes don't match, reset prev_weights to current weights
                # This handles variable batch sizes during training
                pass

        self.prev_weights = weights.detach().clone()

        return {
            'w_eff': weights[..., 0:1],
            'w_stab': weights[..., 1:2],
            'w_cost': weights[..., 2:3],
            'all_weights': weights,
            'scene_probs': scene_probs
        }


class EnhancedInfluenceBasedController(nn.Module):
    """
    增强的影响力控制器

    核心改进：
    1. 可学习的α、β权重
    2. 多维度影响力评分
    3. 自适应Top-K值
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        flow_dim: int = 64,
        risk_dim: int = 64,
        global_dim: int = 32,
        hidden_dim: int = 128,
        action_dim: int = 2,
        base_top_k: int = 5,
        max_top_k: int = 10,
        min_top_k: int = 2,
        dropout: float = 0.2,
        use_learnable_weights: bool = True
    ):
        super().__init__()

        self.gnn_dim = gnn_dim
        self.flow_dim = flow_dim
        self.risk_dim = risk_dim
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.base_top_k = base_top_k
        self.max_top_k = max_top_k
        self.min_top_k = min_top_k
        self.use_learnable_weights = use_learnable_weights

        # 可学习的权重参数
        if use_learnable_weights:
            self.register_parameter('alpha', nn.Parameter(torch.tensor(0.6)))
            self.register_parameter('beta', nn.Parameter(torch.tensor(0.4)))

            # 场景自适应
            self.congestion_adapter = nn.Sequential(
                nn.Linear(global_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
                nn.Sigmoid()
            )
        else:
            self.alpha = 0.6
            self.beta = 0.4

        # 多维度影响力评分
        self.gnn_importance_net = nn.Sequential(
            nn.Linear(gnn_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.future_impact_net = nn.Sequential(
            nn.Linear(flow_dim + risk_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 自适应Top-K预测器
        self.top_k_predictor = nn.Sequential(
            nn.Linear(global_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 特征融合
        fusion_input = gnn_dim + flow_dim + risk_dim
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input + 32, 384),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(384),
            nn.Linear(384, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # 动作生成
        self.action_generator = nn.ModuleDict({
            'acceleration': nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Tanh()
            ),
            'lane_change': nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        })

        # 价值网络
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.cost_value_network = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        gnn_embedding: torch.Tensor,
        z_flow: torch.Tensor,
        z_risk: torch.Tensor,
        global_metrics: torch.Tensor,
        vehicle_ids: List[str],
        is_icv: torch.Tensor,
        deterministic: bool = False
    ) -> Dict[str, Any]:
        """前向传播"""
        device = gnn_embedding.device

        # 计算可学习的权重
        if self.use_learnable_weights:
            alpha_learned = torch.sigmoid(self.alpha)
            beta_learned = torch.sigmoid(self.beta)
            total = alpha_learned + beta_learned + 1e-6
            alpha_final = (alpha_learned / total).item()
            beta_final = (beta_learned / total).item()
        else:
            alpha_final = self.alpha
            beta_final = self.beta

        # 计算影响力得分
        gnn_importance = self.gnn_importance_net(gnn_embedding).squeeze(-1)
        future_features = torch.cat([z_flow, z_risk], dim=-1)
        future_impact = self.future_impact_net(future_features).squeeze(-1)

        influence_scores = alpha_final * gnn_importance + beta_final * future_impact

        # 自适应Top-K
        top_k_ratio = self.top_k_predictor(global_metrics).item()
        adaptive_k = int(
            self.min_top_k +
            (self.max_top_k - self.min_top_k) * top_k_ratio
        )
        adaptive_k = min(adaptive_k, self.base_top_k)

        # 全局上下文
        global_context = self.global_encoder(global_metrics)
        if global_context.dim() == 1:
            global_context = global_context.unsqueeze(0)

        # 特征融合
        fused = torch.cat([
            gnn_embedding.reshape(-1, self.gnn_dim),
            z_flow.reshape(-1, self.flow_dim),
            z_risk.reshape(-1, self.risk_dim),
            global_context.expand(gnn_embedding.size(0), -1).reshape(-1, 32)
        ], dim=-1)

        fused_features = self.fusion_layer(fused)

        # 过滤ICV
        icv_mask = is_icv.bool()
        icv_indices = torch.where(icv_mask)[0]

        if len(icv_indices) == 0:
            return {
                'selected_vehicle_ids': [],
                'raw_actions': torch.zeros(0, self.action_dim, device=device),
                'influence_scores': torch.zeros(0, device=device),
                'adaptive_k': 0,
                'alpha_final': alpha_final,
                'beta_final': beta_final,
                'value_estimates': torch.zeros(0, device=device),
                'cost_estimates': torch.zeros(0, device=device)
            }

        # Top-K选择
        icv_influence = influence_scores[icv_indices]
        k = min(adaptive_k, len(icv_indices))
        top_k_scores, top_k_local_indices = torch.topk(icv_influence, k, largest=True, sorted=True)

        selected_indices = icv_indices[top_k_local_indices]
        selected_vehicle_ids = [vehicle_ids[i] for i in selected_indices.cpu().numpy()]

        # 生成动作
        selected_features = fused_features[selected_indices]

        if deterministic:
            accel_actions = self.action_generator['acceleration'](selected_features)
            lane_actions = self.action_generator['lane_change'](selected_features)
        else:
            accel_mean = self.action_generator['acceleration'](selected_features)
            lane_mean = self.action_generator['lane_change'](selected_features)
            accel_actions = accel_mean + torch.randn_like(accel_mean) * 0.1
            lane_actions = torch.clamp(lane_mean + torch.randn_like(lane_mean) * 0.05, 0.0, 1.0)

        raw_actions = torch.cat([accel_actions, lane_actions], dim=-1)

        # 价值估计
        value_estimates = self.value_network(fused_features).squeeze(-1)
        cost_estimates = self.cost_value_network(fused_features).squeeze(-1)

        return {
            'selected_vehicle_ids': selected_vehicle_ids,
            'selected_indices': selected_indices.cpu().numpy().tolist(),
            'raw_actions': raw_actions,
            'influence_scores': influence_scores,
            'adaptive_k': adaptive_k,
            'alpha_final': alpha_final,
            'beta_final': beta_final,
            'top_k_scores': top_k_scores,
            'value_estimates': value_estimates,
            'cost_estimates': cost_estimates
        }

    def get_alpha_beta(self) -> Tuple[float, float]:
        """获取当前的α和β值"""
        if self.use_learnable_weights:
            alpha = torch.sigmoid(self.alpha).item()
            beta = torch.sigmoid(self.beta).item()
            total = alpha + beta + 1e-6
            return alpha / total, beta / total
        else:
            return self.alpha, self.beta


class IdealTrafficControllerV4(nn.Module):
    """
    理想交通控制器 v4.0（增强版默认启用）

    完整架构：
    1. Risk-Sensitive GNN (感知)
    2. Multi-Scale RSSM (预测)
    3. Enhanced Dynamic Weight Gating (元控制，带场景识别)
    4. Enhanced Influence-Based Controller (决策，可学习权重)
    5. Lagrangian Optimizer (约束，动态λ更新)
    6. Safety Barrier (安全)

    增强功能默认启用：
    - 强化拉格朗日约束（动态λ更新）
    - 可学习的Top-K权重（α、β参数）
    - 增强的动态权重门控（场景识别）
    - 自适应Top-K值调整
    """

    def __init__(
        self,
        node_dim: int = 9,
        edge_dim: int = 4,
        global_dim: int = 32,
        gnn_hidden_dim: int = 64,
        gnn_output_dim: int = 256,
        rssm_hidden_dim: int = 128,
        rssm_latent_dim: int = 64,
        controller_hidden_dim: int = 128,
        history_dim: int = 32,
        top_k: int = 5,
        dropout: float = 0.2,
        device: str = 'cuda'
    ):
        super().__init__()

        self.device = torch.device(device)
        self.top_k = top_k

        # 1. 感知层：风险敏感GNN
        self.perception_layer = RiskSensitiveGNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            dropout=dropout
        )

        # 2. 预测层：多尺度RSSM
        self.prediction_layer = MultiScaleRSSM(
            input_dim=gnn_output_dim,
            hidden_dim=rssm_hidden_dim,
            latent_dim=rssm_latent_dim,
            dropout=dropout
        )

        # 3. 决策层：增强的影响力控制器（可学习权重+自适应Top-K）
        self.decision_layer = EnhancedInfluenceBasedController(
            gnn_dim=gnn_output_dim,
            flow_dim=rssm_latent_dim,
            risk_dim=rssm_latent_dim,
            global_dim=global_dim,
            hidden_dim=controller_hidden_dim,
            base_top_k=top_k,
            max_top_k=min(top_k + 3, 10),
            min_top_k=max(top_k - 2, 2),
            dropout=dropout,
            use_learnable_weights=True  # 默认启用可学习权重
        )

        # 4. 动态权重门控：增强版（场景识别+时间平滑）
        self.weight_gating = EnhancedDynamicWeightGating(
            state_dim=gnn_output_dim,
            history_dim=history_dim,
            prediction_dim=rssm_latent_dim * 2,
            hidden_dim=128,
            dropout=dropout,
            use_temporal_smoothing=True
        )

        # 5. 拉格朗日优化器：动态约束优化
        self.lagrangian_optimizer = LagrangianOptimizer(
            cost_limit=0.1,
            lambda_init=0.1,
            adaptive_penalty=True
        )

        # 6. 安全屏障
        self.safety_barrier = SafetyBarrier()

        # 初始化隐藏状态
        self.register_buffer('rssm_hidden', None)

        # 场景名称（用于可解释性）
        self.scene_names = ['平峰', '早高峰', '晚高峰', '拥堵']

    def forward(
        self,
        observation: Dict[str, Any],
        deterministic: bool = False
    ) -> Dict[str, Any]:
        """完整前向传播"""
        # 1. 感知层
        gnn_output = self.perception_layer(
            node_features=observation['node_features'],
            edge_index=observation['edge_index'],
            edge_features=observation['edge_features'],
            risk_features=observation['risk_features'],
            batch=observation.get('batch')
        )

        node_embeddings = gnn_output['node_embeddings']
        global_embedding = gnn_output['global_embedding']

        # 2. 预测层
        rssm_output = self.prediction_layer(
            node_embeddings=node_embeddings,
            hidden_state=self.rssm_hidden
        )
        self.rssm_hidden = rssm_output['hidden_state']

        z_flow = rssm_output['z_flow']
        z_risk = rssm_output['z_risk']

        # 3. 动态权重门控（增强版：场景识别）
        prediction_features = torch.cat([
            z_flow.mean(dim=0, keepdim=True),
            z_risk.mean(dim=0, keepdim=True)
        ], dim=-1)

        weight_output = self.weight_gating(
            global_state=global_embedding,
            prediction_features=prediction_features
        )
        dynamic_weights = weight_output['all_weights']
        scene_probs = weight_output['scene_probs']

        # 4. 决策层（增强版：可学习权重+自适应Top-K）
        decision_output = self.decision_layer(
            gnn_embedding=node_embeddings,
            z_flow=z_flow,
            z_risk=z_risk,
            global_metrics=observation['global_metrics'],
            vehicle_ids=observation['vehicle_ids'],
            is_icv=observation['is_icv'],
            deterministic=deterministic
        )

        alpha_final = decision_output['alpha_final']
        beta_final = decision_output['beta_final']
        adaptive_k = decision_output['adaptive_k']

        # 5. 安全屏障
        safe_actions = decision_output['raw_actions']
        barrier_info = {}

        if 'vehicle_states' in observation:
            safe_actions, barrier_info = self.safety_barrier(
                actions=decision_output['raw_actions'],
                vehicle_states=observation['vehicle_states'],
                ttc_values=observation.get('ttc_values')
            )

        # 汇总输出
        result = {
            'selected_vehicle_ids': decision_output['selected_vehicle_ids'],
            'safe_actions': safe_actions,
            'num_controlled': len(decision_output['selected_vehicle_ids']),
            'influence_scores': decision_output['influence_scores'],
            'dynamic_weights': dynamic_weights,
            'alpha_final': alpha_final,
            'beta_final': beta_final,
            'adaptive_k': adaptive_k,
            'scene_probs': scene_probs,
            'value_estimates': decision_output['value_estimates'],
            'cost_estimates': decision_output['cost_estimates'],
            'gnn_output': gnn_output,
            'rssm_output': rssm_output,
            'decision_output': decision_output,
            'barrier_info': barrier_info
        }

        return result

    def compute_constrained_loss(
        self,
        batch: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        num_controlled: int,
        total_vehicles: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算约束损失（使用拉格朗日优化器）"""
        output = self.forward(batch, deterministic=False)

        # 基础损失
        value_pred = output.get('value_estimates')
        cost_pred = output.get('cost_estimates')

        if value_pred is None:
            value_loss = torch.tensor(0.0, device=self.device)
        else:
            value_target = targets.get('value_target', torch.zeros_like(value_pred))
            value_loss = F.mse_loss(value_pred.mean(), value_target.mean())

        if cost_pred is None:
            cost_loss = torch.tensor(0.0, device=self.device)
        else:
            cost_target = targets.get('cost_target', torch.zeros_like(cost_pred))
            cost_loss = F.smooth_l1_loss(cost_pred.mean(), cost_target.mean())

        # 约束项（使用拉格朗日优化器）
        actual_cost = num_controlled / max(total_vehicles, 1)
        cost_violation = actual_cost - 0.1  # cost_limit = 0.1

        # 动态更新λ
        self.lagrangian_optimizer.update_lambda(cost_violation)
        lambda_param = self.lagrangian_optimizer.lambda_param
        constraint_term = lambda_param * max(0, cost_violation)

        # 总损失
        total_loss = value_loss + 0.5 * cost_loss + lambda_param * constraint_term

        info = {
            'total_loss': total_loss.item(),
            'value_loss': value_loss.item(),
            'cost_loss': cost_loss.item(),
            'constraint_term': constraint_term if isinstance(constraint_term, float) else constraint_term.item(),
            'lambda_param': lambda_param,
            'actual_cost': actual_cost,
            'cost_violation': cost_violation
        }

        return total_loss, info

    def get_explanation(self) -> Dict[str, Any]:
        """获取可解释性信息"""
        explanation = {}

        # 场景识别
        if hasattr(self.weight_gating, 'prev_weights') and self.weight_gating.prev_weights is not None:
            weights = self.weight_gating.prev_weights
            explanation['current_weights'] = {
                'efficiency': f"{weights[0, 0].item():.2%}",
                'stability': f"{weights[0, 1].item():.2%}",
                'cost': f"{weights[0, 2].item():.2%}"
            }

        # Top-K参数（可学习权重）
        alpha, beta = self.decision_layer.get_alpha_beta()
        explanation['top_k_params'] = {
            'alpha': f"{alpha:.3f}",
            'beta': f"{beta:.3f}"
        }

        # 拉格朗日参数
        lagrangian_stats = self.lagrangian_optimizer.get_stats()
        explanation['lagrangian'] = {
            'lambda': f"{lagrangian_stats['lambda_param']:.3f}",
            'violation_rate': f"{lagrangian_stats['violation_rate']:.2%}"
        }

        return explanation

    def freeze_perception(self):
        """冻结感知层"""
        for param in self.perception_layer.parameters():
            param.requires_grad = False
        print("[OK] Frozen perception layer (GNN)")

    def freeze_prediction(self):
        """冻结预测层"""
        for param in self.prediction_layer.parameters():
            param.requires_grad = False
        print("[OK] Frozen prediction layer (World Model)")

    def freeze_decision(self):
        """冻结决策层"""
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
