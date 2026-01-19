"""
轻量级OCR驱动架构 v5.0 - 初赛专用

核心设计理念:
1. 直接优化OCR（OD完成率）而非速度/吞吐量
2. 稀疏控制机制：只控制5-10%的关键车辆
3. 移除世界模型：简化架构，加速训练
4. 可解释的车辆选择：基于交通工程理论

适用场景：初赛（只控制车辆，不控制设施）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class LightweightGraphConvolution(nn.Module):
    """
    轻量级图卷积层

    使用简化的消息传递机制，避免复杂的GNN库依赖
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, in_features] 节点特征
            adj: [N, N] 邻接矩阵（可以是注意力权重）

        Returns:
            out: [N, out_features] 更新后的节点特征
        """
        # 消息传递：聚合邻居信息
        # adj[i, j] 表示节点j对节点i的影响权重
        messages = torch.bmm(adj.unsqueeze(0), x.unsqueeze(0)).squeeze(0)

        # 线性变换
        out = self.linear(messages)

        # 归一化和激活
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)

        return out


class VehicleInfluenceScorer(nn.Module):
    """
    车辆影响力评分模块

    基于可解释特征计算每辆车对OCR的潜在贡献：
    - 位置权重：瓶颈区域的车辆更重要
    - 速度权重：异常速度（慢车）影响后车
    - 车道权重：汇流区车辆需要协调
    - 邻居数：周围车辆多则影响力大
    """

    def __init__(self, node_dim: int, hidden_dim: int = 64):
        super().__init__()

        # 可学习的特征融合权重
        self.feature_fusion = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出[0, 1]的影响力评分
        )

        # 可学习的手工特征权重（4个维度）
        self.handcrafted_weights = nn.Parameter(torch.ones(4) / 4)  # 初始化为均匀权重

    def forward(
        self,
        vehicle_states: torch.Tensor,
        gnn_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算车辆影响力评分

        Args:
            vehicle_states: [N, 9] 原始车辆状态
                [s, d, vs, vd, speed, accel, lane, angle, is_icv]
            gnn_embeddings: [N, dim] GNN输出的嵌入

        Returns:
            influence_scores: [N, 1] 影响力评分
            debug_info: 调试信息字典
        """
        N = vehicle_states.size(0)
        device = vehicle_states.device

        # ========== 1. 提取可解释特征 ==========
        s = vehicle_states[:, 0]          # 纵向位置 [0, 1]
        lanes = vehicle_states[:, 6]      # 车道索引
        speeds = vehicle_states[:, 4]     # 速度 [0, 1]

        # 特征1：位置权重（瓶颈区域优先）
        # 主线末端(s > 0.7)且非匝道(lane < 3)的车辆权重更高
        in_bottleneck = ((s > 0.7) & (lanes < 3)).float()

        # 特征2：速度异常（慢车优先）
        # 使用softmax归一化，慢车获得高分
        speed_scores = F.softmax(-speeds / (speeds.std() + 1e-6), dim=0)

        # 特征3：汇流区域（车道协调）
        # 匝道车道(lane >= 3)的车辆在汇流区重要
        in_merge_zone = (s > 0.5) & (s < 0.8) & (lanes >= 3)
        merge_score = in_merge_zone.float()

        # 特征4：密度影响（周围车辆数）
        # 这里简化处理：假设所有车都有影响力，后续可以通过邻居数计算

        # 手工特征矩阵 [N, 4]
        handcrafted_features = torch.stack([
            in_bottleneck,
            speed_scores,
            merge_score,
            torch.ones(N, device=device)  # 邻居数占位
        ], dim=1)

        # ========== 2. 学习特征融合 ==========
        # 将GNN嵌入映射为影响力评分
        learned_influence = self.feature_fusion(gnn_embeddings)  # [N, 1]

        # ========== 3. 组合评分（可解释 + 可学习）==========
        # 归一化手工特征权重
        weights = F.softmax(self.handcrafted_weights, dim=0)

        # 计算手工特征得分
        handcrafted_influence = (handcrafted_features * weights).sum(dim=1, keepdim=True)  # [N, 1]

        # 组合：70%学习 + 30%手工（可调整）
        final_influence = 0.7 * learned_influence + 0.3 * handcrafted_influence

        # 归一化到[0, 1]
        final_influence = torch.sigmoid(final_influence)

        debug_info = {
            'learned_influence': learned_influence,
            'handcrafted_influence': handcrafted_influence,
            'in_bottleneck': in_bottleneck,
            'speed_scores': speed_scores,
            'merge_score': merge_score,
        }

        return final_influence, debug_info


class LightweightOCRGNN(nn.Module):
    """
    轻量级OCR驱动的GNN

    核心特性:
    1. 3层图卷积（足够表达交通流动力学）
    2. 内置影响力评分模块
    3. Top-K选择机制（硬约束，只控制K辆车）
    4. 策略头只对Top-K车辆输出动作
    """

    def __init__(
        self,
        node_dim: int = 9,
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        top_k_ratio: float = 0.05  # 只控制5%的车辆
    ):
        super().__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.top_k_ratio = top_k_ratio

        # ========== 输入嵌入层 ==========
        self.input_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ========== GNN层 ==========
        self.gnn_layers = nn.ModuleList([
            LightweightGraphConvolution(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # ========== 车辆影响力评分器 ==========
        self.influence_scorer = VehicleInfluenceScorer(
            node_dim=hidden_dim,
            hidden_dim=hidden_dim // 2
        )

        # ========== 策略头（只对Top-K车辆使用）==========
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # 2个动作：加速度 + 换道概率
        )

        # ========== 价值头（全局状态评估）==========
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # ========== 初始化参数 ==========
        self.reset_parameters()

    def reset_parameters(self):
        """重置所有参数"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _build_adjacency_matrix(
        self,
        vehicle_states: torch.Tensor,
        num_vehicles: int,
        interaction_radius: float = 0.15
    ) -> torch.Tensor:
        """
        构建邻接矩阵（基于空间距离）

        Args:
            vehicle_states: [N, 9] 车辆状态（可能包含padding）
            num_vehicles: 实际车辆数
            interaction_radius: 相互作用半径（归一化距离）

        Returns:
            adj: [N, N] 邻接矩阵（行归一化）
        """
        N = vehicle_states.size(0)
        device = vehicle_states.device

        # 提取位置信息
        s = vehicle_states[:, 0]  # 纵向位置 [0, 1]
        d = vehicle_states[:, 1]  # 横向位置 [0, 1]
        lanes = vehicle_states[:, 6].long()  # 车道索引

        # 计算距离矩阵
        s_diff = s.unsqueeze(1) - s.unsqueeze(0)  # [N, N]
        d_diff = d.unsqueeze(1) - d.unsqueeze(0)  # [N, N]
        lane_diff = (lanes.unsqueeze(1) - lanes.unsqueeze(0)).abs()  # [N, N]

        # 空间距离
        dist_sq = s_diff ** 2 + d_diff ** 2  # [N, N]

        # 掩码：只连接同车道或相邻车道
        lane_mask = (lane_diff <= 1).float()

        # 掩码：只连接半径内的车辆
        dist_mask = (dist_sq < interaction_radius ** 2).float()

        # 掩码：排除自连接
        not_self = 1.0 - torch.eye(N, device=device)

        # 组合掩码
        mask = lane_mask * dist_mask * not_self

        # 掩码padding的车辆（位置为0的车辆）
        valid_mask = (s > 0).float()
        mask = mask * valid_mask.unsqueeze(1) * valid_mask.unsqueeze(0)

        # 归一化：每一行和为1（消息传递的权重）
        row_sum = mask.sum(dim=1, keepdim=True)
        adj = mask / (row_sum + 1e-6)

        return adj

    def forward(
        self,
        vehicle_states: torch.Tensor,
        num_vehicles: int,
        return_debug_info: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            vehicle_states: [batch_size, max_vehicles, 9] 车辆状态
            num_vehicles: 实际车辆数（用于Top-K计算）
            return_debug_info: 是否返回调试信息

        Returns:
            outputs: 包含以下字段的字典
                - top_k_indices: Top-K车辆的索引
                - top_k_actions: Top-K车辆的动作
                - top_k_mask: Top-K掩码
                - value: 价值估计
                - influence_scores: 影响力评分
        """
        batch_size = vehicle_states.size(0)
        max_vehicles = vehicle_states.size(1)
        device = vehicle_states.device

        # ========== 1. 输入嵌入 ==========
        h = self.input_embedding(vehicle_states)  # [B, N, hidden_dim]

        # ========== 2. 构建邻接矩阵 ==========
        # 为了效率，只对第一个样本建图，然后batch复用
        adj = self._build_adjacency_matrix(
            vehicle_states[0],  # [N, 9]
            num_vehicles
        )  # [N, N]

        # 扩展到batch
        adj_batch = adj.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, N]

        # ========== 3. GNN层（消息传递）==========
        for gnn_layer in self.gnn_layers:
            # 逐batch应用GNN（避免内存爆炸）
            batch_outputs = []
            for b in range(batch_size):
                h_b = gnn_layer(h[b], adj_batch[b])  # [N, hidden_dim]
                batch_outputs.append(h_b)
            h = torch.stack(batch_outputs, dim=0)  # [B, N, hidden_dim]

        # ========== 4. 计算影响力评分 ==========
        # 只对第一个样本计算（batch中所有样本使用相同的Top-K选择）
        influence_scores, debug_info = self.influence_scorer(
            vehicle_states[0],  # [N, 9]
            h[0]  # [N, hidden_dim]
        )  # [N, 1]

        # ========== 5. Top-K选择 ==========
        k = max(1, int(num_vehicles * self.top_k_ratio))

        # 选择影响力最高的K辆车
        top_k_scores, top_k_indices = torch.topk(
            influence_scores.squeeze(), k=k, dim=0
        )  # [k]

        # 创建Top-K掩码（用于后续动作选择）
        top_k_mask = torch.zeros(max_vehicles, device=device)
        top_k_mask[top_k_indices] = 1.0

        # 扩展到batch
        top_k_mask_batch = top_k_mask.unsqueeze(0).expand(batch_size, -1)  # [B, N]

        # ========== 6. 策略输出（只对Top-K车辆）==========
        # 所有车的策略输出
        policy_raw = self.policy_head(h)  # [B, N, 2]

        # 加速：[-1, 1] → [-3, 2] m/s²
        accel_raw = policy_raw[:, :, 0:1]  # [B, N, 1]
        accel = (accel_raw + 1.0) / 2.0 * (2.0 - (-3.0)) + (-3.0)

        # 换道：[-1, 1] → [0, 1] 概率
        lane_change_raw = policy_raw[:, :, 1:2]  # [B, N, 1]
        lane_change_prob = (lane_change_raw + 1.0) / 2.0
        lane_change_prob = lane_change_prob.clamp(0.0, 1.0)

        # 应用Top-K掩码：非Top-K车辆不控制（动作=0）
        top_k_actions = torch.cat([
            accel * top_k_mask_batch.unsqueeze(-1),
            lane_change_prob * top_k_mask_batch.unsqueeze(-1)
        ], dim=-1)  # [B, N, 2]

        # ========== 7. 价值估计（全局池化）==========
        # 使用attention池化
        attention_weights = F.softmax(h.mean(dim=-1), dim=1)  # [B, N]
        global_features = (h * attention_weights.unsqueeze(-1)).sum(dim=1)  # [B, hidden_dim]
        value = self.value_head(global_features)  # [B, 1]

        # ========== 8. 准备输出 ==========
        outputs = {
            'top_k_indices': top_k_indices,  # [k]
            'top_k_actions': top_k_actions,  # [B, N, 2]
            'top_k_mask': top_k_mask_batch,  # [B, N]
            'value': value,  # [B, 1]
            'influence_scores': influence_scores,  # [N, 1]
            'node_embeddings': h,  # [B, N, hidden_dim]
        }

        if return_debug_info:
            outputs['debug_info'] = debug_info
            outputs['adjacency_matrix'] = adj

        return outputs


class LightweightPolicyV5(nn.Module):
    """
    轻量级策略网络 v5.0 - 初赛专用

    架构层次：
    1. 轻量级OCR-GNN（特征提取 + Top-K选择）
    2. 动作投影层（生成加速度和换道指令）
    3. 价值网络（用于PPO训练）
    """

    def __init__(
        self,
        obs_dim: int = 321,  # max_vehicles(32) * 9 + 32(global) + 1(num)
        action_dim: int = 2,  # 加速度 + 换道
        config: Optional[Dict] = None
    ):
        super().__init__()

        # 从obs_dim推断max_vehicles
        # obs_dim = max_vehicles * 9 + 32 + 1
        self.max_vehicles = (obs_dim - 32 - 1) // 9
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 解析配置
        if config is None:
            config = {}
        model_config = config.get('model', {})

        gnn_config = model_config.get('gnn', {})
        self.top_k_ratio = gnn_config.get('top_k_ratio', 0.05)

        # ========== 核心GNN ==========
        self.gnn = LightweightOCRGNN(
            node_dim=9,
            hidden_dim=gnn_config.get('hidden_dim', 64),
            output_dim=gnn_config.get('output_dim', 64),
            num_layers=gnn_config.get('num_layers', 3),
            dropout=gnn_config.get('dropout', 0.1),
            top_k_ratio=self.top_k_ratio
        )

        # ========== 动作分布参数 ==========
        # 可学习的log_std（用于探索）
        self.log_std = nn.Parameter(torch.full((self.max_vehicles * 2,), np.log(0.1)))

        print(f"[LightweightPolicyV5] Initialized with max_vehicles={self.max_vehicles}, top_k_ratio={self.top_k_ratio}")

    def _parse_observation(
        self,
        obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        解析观测张量

        Args:
            obs: [batch_size, obs_dim] 扁平化观测

        Returns:
            vehicle_states: [batch_size, max_vehicles, 9]
            global_stats: [batch_size, 32]
            num_vehicles: int
        """
        batch_size = obs.size(0)
        device = obs.device

        # 动态推断max_vehicles（适配不同课程级别）
        actual_max_vehicles = (obs.size(1) - 32 - 1) // 9

        # 车辆特征
        vehicle_dim = actual_max_vehicles * 9
        vehicle_features = obs[:, :vehicle_dim]  # [B, N*9]
        vehicle_states = vehicle_features.view(batch_size, actual_max_vehicles, 9)

        # 全局统计
        global_stats = obs[:, vehicle_dim:vehicle_dim + 32]  # [B, 32]

        # 实际车辆数（取最后一个特征）
        num_vehicles_tensor = obs[:, vehicle_dim + 32:vehicle_dim + 33]  # [B, 1]
        num_vehicles = int(num_vehicles_tensor[0, 0].item())

        return vehicle_states, global_stats, num_vehicles

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播（训练模式）

        Args:
            obs: [batch_size, obs_dim]
            deterministic: 是否确定性动作

        Returns:
            actions: [batch_size, max_vehicles * 2]
            values: [batch_size, 1]
            log_prob: [batch_size]
        """
        batch_size = obs.size(0)
        device = obs.device

        # 解析观测
        vehicle_states, global_stats, num_vehicles = self._parse_observation(obs)

        # GNN前向传播
        gnn_outputs = self.gnn(
            vehicle_states,
            num_vehicles,
            return_debug_info=False
        )

        # 提取Top-K动作
        top_k_actions = gnn_outputs['top_k_actions']  # [B, N, 2]
        value = gnn_outputs['value']  # [B, 1]
        top_k_mask = gnn_outputs['top_k_mask']  # [B, N]

        # 展平动作：[B, N, 2] → [B, N*2]
        actions_flat = top_k_actions.reshape(batch_size, -1)

        # 动作分布（使用可学习log_std）
        log_std = self.log_std.unsqueeze(0).expand_as(actions_flat)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)  # 数值稳定性

        mean = actions_flat
        std = torch.exp(log_std)

        # 采样动作
        if deterministic:
            actions_out = mean
        else:
            actions_out = mean + std * torch.randn_like(mean)

        # 计算log_prob
        log_prob = -0.5 * (((actions_out - mean) / (std + 1e-6)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1)  # [B]

        return actions_out, value, log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作（PPO训练时使用）

        重新计算给定obs下actions的log_prob和entropy
        """
        batch_size = obs.size(0)

        # 解析观测
        vehicle_states, global_stats, num_vehicles = self._parse_observation(obs)

        # GNN前向传播
        gnn_outputs = self.gnn(vehicle_states, num_vehicles)

        top_k_actions = gnn_outputs['top_k_actions']
        value = gnn_outputs['value']

        # 展平
        actions_flat = top_k_actions.reshape(batch_size, -1)

        # 动作分布
        log_std = self.log_std.unsqueeze(0).expand_as(actions_flat)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)

        mean = actions_flat
        std = torch.exp(log_std)

        # 计算log_prob和entropy
        log_prob = -0.5 * (((actions - mean) / (std + 1e-6)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1)

        entropy = 0.5 * (np.log(2 * np.pi) + 2 * log_std + 1).sum(dim=-1)
        entropy = entropy.mean()  # 标量

        return value, log_prob, entropy

    def select_vehicles(
        self,
        obs: np.ndarray
    ) -> Tuple[List[int], Dict]:
        """
        选择要控制的车辆（推理模式）

        Args:
            obs: [obs_dim] 单个观测

        Returns:
            selected_indices: 选中的车辆索引列表
            info: 额外信息（用于调试）
        """
        device = next(self.parameters()).device
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            # 解析观测
            vehicle_states, _, num_vehicles = self._parse_observation(obs_tensor)

            # GNN前向传播
            gnn_outputs = self.gnn(
                vehicle_states,
                num_vehicles,
                return_debug_info=True
            )

            # 提取Top-K索引
            top_k_indices = gnn_outputs['top_k_indices'].cpu().numpy()
            influence_scores = gnn_outputs['influence_scores'].cpu().numpy()
            debug_info = gnn_outputs.get('debug_info', {})

            info = {
                'selected_indices': top_k_indices.tolist(),
                'influence_scores': influence_scores.tolist(),
                'num_vehicles': num_vehicles,
                'k': len(top_k_indices),
            }

            # 添加调试信息
            if 'learned_influence' in debug_info:
                info['learned_influence'] = debug_info['learned_influence'].cpu().numpy().tolist()
            if 'handcrafted_influence' in debug_info:
                info['handcrafted_influence'] = debug_info['handcrafted_influence'].cpu().numpy().tolist()

        return top_k_indices.tolist(), info


def create_lightweight_policy_v5(obs_dim: int, action_dim: int, config: Optional[Dict] = None):
    """
    创建轻量级策略网络 v5.0

    Args:
        obs_dim: 观测维度
        action_dim: 动作维度
        config: 配置字典

    Returns:
        policy: LightweightPolicyV5实例
    """
    policy = LightweightPolicyV5(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config if config is not None else {}
    )
    return policy


# 兼容性别名
create_policy = create_lightweight_policy_v5
