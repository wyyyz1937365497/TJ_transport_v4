"""神经网络模块 - 简化且高效"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Any

from .graph import FastGraphBuilder


class RiskSensitiveGNN(nn.Module):
    """风险感知图神经网络"""

    def __init__(
        self,
        node_dim: int = 9,
        edge_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 256,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads

        # 输入投影
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # GAT层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim * heads if i > 0 else hidden_dim
            self.convs.append(
                GATConv(
                    in_dim,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            )
            if use_batch_norm:
                self.norms.append(nn.BatchNorm1d(hidden_dim * heads))
            else:
                self.norms.append(nn.Identity())

        self.dropout = nn.Dropout(dropout)

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim * heads, output_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            node_features: [N, node_dim]
            edge_index: [2, E]
            edge_features: [E, edge_dim]
            batch: [N] (可选)

        Returns:
            Dict包含node_embedding和graph_embedding
        """
        x = self.input_proj(node_features)

        # GAT层
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr=edge_features)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)

        # 节点嵌入
        node_embedding = self.output_proj(x)  # [N, output_dim]

        # 图嵌入（全局池化）
        if batch is not None:
            graph_embedding = global_mean_pool(node_embedding, batch)
        else:
            graph_embedding = node_embedding.mean(dim=0, keepdim=True)

        return {
            "node_embedding": node_embedding,
            "graph_embedding": graph_embedding,
        }


class ProgressiveWorldModel(nn.Module):
    """渐进式世界模型 - 预测未来状态"""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        future_steps: int = 5,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.future_steps = future_steps

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 输出投影
        self.output_proj = nn.Linear(lstm_output_dim, input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: [B, input_dim] 或 [B, T, input_dim]

        Returns:
            Dict包含预测的未来状态
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, input_dim]

        B, T, _ = x.shape

        # 投影到隐藏维度
        x = self.input_proj(x)
        x = self.dropout(x)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # 输出投影
        predictions = self.output_proj(lstm_out)  # [B, T, input_dim]

        return {
            "future_states": predictions,
            "next_state": predictions[:, -1],  # [B, input_dim]
        }


class InfluenceDrivenController(nn.Module):
    """影响力驱动的控制器"""

    def __init__(
        self,
        gnn_dim: int = 256,
        world_dim: int = 256,
        global_dim: int = 16,
        hidden_dim: int = 128,
        action_dim: int = 2,
        top_k: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gnn_dim = gnn_dim
        self.world_dim = world_dim
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.top_k = top_k

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(gnn_dim + world_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 影响力评分
        self.influence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 动作生成
        self.action_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),  # 输出范围 [-1, 1]
        )

        # 价值估计
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        gnn_embedding: torch.Tensor,
        world_predictions: torch.Tensor,
        global_metrics: torch.Tensor,
        vehicle_ids: List[str],
        is_icv: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            gnn_embedding: [N, gnn_dim]
            world_predictions: [N, world_dim]
            global_metrics: [B, global_dim]
            vehicle_ids: 车辆ID列表
            is_icv: [N] 是否为ICV

        Returns:
            Dict包含动作、影响力评分等
        """
        # 扩展全局特征
        B = global_metrics.size(0)
        N = gnn_embedding.size(0)
        global_expanded = global_metrics.expand(N, -1)

        # 特征融合
        fused = self.fusion(
            torch.cat([gnn_embedding, world_predictions, global_expanded], dim=-1)
        )  # [N, hidden_dim]

        # 影响力评分
        influence_scores = self.influence_net(fused).squeeze(-1)  # [N]

        # 只考虑ICV
        icv_mask = is_icv.bool()
        if not icv_mask.any():
            return {
                "selected_vehicle_ids": [],
                "selected_indices": [],
                "raw_actions": torch.zeros(0, self.action_dim, device=fused.device),
                "influence_scores": influence_scores,
                "value_estimates": None,
                "cost_estimates": None,
            }

        # 选择Top-K
        icv_influence = influence_scores.clone()
        icv_influence[~icv_mask] = float("-inf")

        top_k = min(self.top_k, icv_mask.sum().item())
        if top_k == 0:
            return {
                "selected_vehicle_ids": [],
                "selected_indices": [],
                "raw_actions": torch.zeros(0, self.action_dim, device=fused.device),
                "influence_scores": influence_scores,
                "value_estimates": None,
                "cost_estimates": None,
            }

        selected_indices = torch.topk(icv_influence, top_k).indices
        selected_vehicle_ids = [vehicle_ids[i] for i in selected_indices]

        # 生成动作
        selected_fused = fused[selected_indices]
        raw_actions = self.action_net(selected_fused)  # [top_k, action_dim]

        # 缩放动作到合理范围
        raw_actions = raw_actions * torch.tensor(
            [2.0, 1.0], device=raw_actions.device
        )  # [accel, lane_change]

        # 价值估计
        value_estimates = self.value_net(selected_fused).squeeze(-1)

        return {
            "selected_vehicle_ids": selected_vehicle_ids,
            "selected_indices": selected_indices,
            "raw_actions": raw_actions,
            "influence_scores": influence_scores,
            "value_estimates": value_estimates,
            "cost_estimates": torch.abs(raw_actions[:, 0]).mean(),
        }


class DualModeSafetyShield(nn.Module):
    """双模式安全屏障"""

    def __init__(
        self,
        ttc_threshold: float = 2.0,
        thw_threshold: float = 1.5,
        max_accel: float = 2.0,
        max_decel: float = -3.0,
        emergency_decel: float = -5.0,
    ):
        super().__init__()
        self.ttc_threshold = ttc_threshold
        self.thw_threshold = thw_threshold
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.emergency_decel = emergency_decel

    def forward(
        self,
        raw_actions: torch.Tensor,
        vehicle_states: Dict[str, Any],
        selected_vehicle_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        应用安全约束

        Args:
            raw_actions: [N, 2] 原始动作
            vehicle_states: 车辆状态字典
            selected_vehicle_indices: 选中的车辆索引

        Returns:
            安全动作
        """
        if raw_actions.size(0) == 0:
            return {
                "safe_actions": raw_actions,
                "level1_interventions": torch.zeros(0, device=raw_actions.device),
                "level2_interventions": torch.zeros(0, device=raw_actions.device),
            }

        safe_actions = raw_actions.clone()

        # 加速度裁剪
        safe_actions[:, 0] = torch.clamp(safe_actions[:, 0], self.max_decel, self.max_accel)

        # 记录干预
        level1 = (safe_actions[:, 0] != raw_actions[:, 0]).float()
        level2 = torch.zeros_like(level1)

        return {
            "safe_actions": safe_actions,
            "level1_interventions": level1,
            "level2_interventions": level2,
        }


class TrafficController(nn.Module):
    """完整的交通控制器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # 提取子配置
        gnn_cfg = config.get("gnn", {})
        world_cfg = config.get("world_model", {})
        ctrl_cfg = config.get("controller", {})
        safety_cfg = config.get("safety", {})
        graph_cfg = config.get("graph", {})

        # 创建组件
        self.risk_gnn = RiskSensitiveGNN(**gnn_cfg)
        self.world_model = ProgressiveWorldModel(**world_cfg)
        self.controller = InfluenceDrivenController(**ctrl_cfg)
        self.safety_shield = DualModeSafetyShield(**safety_cfg)
        self.graph_builder = FastGraphBuilder(**graph_cfg)

        # 拉格朗日乘子
        self.register_buffer(
            "lagrange_multiplier",
            torch.tensor(config.get("initial_lambda", 1.0)),
        )
        self.cost_limit = config.get("cost_limit", 0.1)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """端到端前向传播"""
        # 构建图（如果还没有）
        if "graph_data" not in batch:
            # TODO: 从batch构建图
            pass

        graph_data = batch["graph_data"]

        # GNN特征提取
        gnn_output = self.risk_gnn(
            node_features=graph_data.x,
            edge_index=graph_data.edge_index,
            edge_features=graph_data.edge_attr,
            batch=batch.get("batch", None),
        )

        gnn_embedding = gnn_output["node_embedding"]

        # 世界模型预测
        world_predictions = self.world_model(gnn_embedding)
        next_state = world_predictions["next_state"]

        # 控制器
        controller_output = self.controller(
            gnn_embedding=gnn_embedding,
            world_predictions=next_state,
            global_metrics=batch.get("global_metrics", torch.zeros(1, 16)),
            vehicle_ids=batch.get("vehicle_ids", []),
            is_icv=batch.get("is_icv", torch.zeros(gnn_embedding.size(0))),
        )

        # 安全屏障
        safe_actions = self.safety_shield(
            raw_actions=controller_output["raw_actions"],
            vehicle_states=batch.get("vehicle_states", {}),
            selected_vehicle_indices=controller_output["selected_indices"],
        )

        return {
            **controller_output,
            "safe_actions": safe_actions["safe_actions"],
            "gnn_embedding": gnn_embedding,
            "world_predictions": world_predictions,
        }

    def update_lagrange_multiplier(self, cost: float):
        """更新拉格朗日乘子"""
        lambda_lr = self.config.get("lambda_lr", 0.01)
        if cost > self.cost_limit:
            self.lagrange_multiplier.data = torch.clamp(
                self.lagrange_multiplier.data * (1 + lambda_lr),
                min=0.1,
                max=10.0,
            )
        else:
            self.lagrange_multiplier.data = torch.clamp(
                self.lagrange_multiplier.data * (1 - lambda_lr * 0.5),
                min=0.1,
                max=10.0,
            )
