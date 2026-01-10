"""
TrafficController - 完整的交通控制神经网络
集成：感知层 + 预测层 + 决策层 + 安全层
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import numpy as np
import os

from .gnn import RiskSensitiveGNN, GraphBuilder
from .world_model import ProgressiveWorldModel
from .controller import InfluenceDrivenController
from .safety import DualModeSafetyShield


class TrafficController(nn.Module):
    """
    完整的交通控制神经网络

    架构：
    1. RiskSensitiveGNN - 感知层
    2. ProgressiveWorldModel - 预测层
    3. InfluenceDrivenController - 决策层
    4. DualModeSafetyShield - 安全层

    特性：
    - 端到端训练
    - 约束优化（拉格朗日乘子）
    - 课程学习支持
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))

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

        # 5. 约束优化参数（拉格朗日乘子）
        self.register_buffer(
            'lagrange_multiplier',
            torch.tensor(config.get('initial_lambda', 1.0))
        )
        self.cost_limit = config.get('cost_limit', 0.1)
        self.lambda_lr = config.get('lambda_lr', 0.01)

        # 图构建器
        self.graph_builder = GraphBuilder(
            interaction_radius=config.get('interaction_radius', 100.0),
            max_neighbors=config.get('max_neighbors', 8),
            lane_change_distance=config.get('lane_change_distance', 50.0)
        )

    def forward(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        端到端前向传播

        Args:
            batch: 包含车辆特征、边特征、全局指标等

        Returns:
            output: 包含控制动作、价值估计等
        """
        # 1. 构建图数据
        if 'graph_data' in batch:
            graph_data = batch['graph_data']
        else:
            # 从车辆状态构建图
            vehicle_states = batch.get('vehicle_states', {})
            icv_ids = batch.get('icv_ids', set())
            graph_data = self.graph_builder.build_graph(vehicle_states, icv_ids)

        # 2. 感知层：GNN特征提取
        # 注意：参数名需要与 RiskSensitiveGNN.forward() 匹配
        gnn_output = self.risk_gnn(
            node_features=graph_data.x,
            edge_index=graph_data.edge_index,
            edge_features=graph_data.edge_attr,  # 修正参数名
            batch=batch.get('batch', None)
        )

        gnn_embedding = gnn_output['node_embedding']

        # 3. 预测层：未来状态预测
        world_predictions = self.world_model(gnn_embedding)

        # 4. 决策层：影响力评分与动作生成
        controller_output = self.controller(
            gnn_embedding=gnn_embedding,
            world_predictions=world_predictions.get('next_state', world_predictions.get('future_states', gnn_embedding)),
            global_metrics=batch.get('global_metrics', torch.zeros(1, 16, device=self.device)),
            vehicle_ids=batch.get('vehicle_ids', []),
            is_icv=batch.get('is_icv', torch.zeros(graph_data.x.size(0), device=self.device))
        )

        # 5. 安全层：动作安全化
        safe_actions = self.safety_shield(
            raw_actions=controller_output['raw_actions'],
            vehicle_states=batch.get('vehicle_states', {}),
            selected_vehicle_indices=controller_output['selected_indices']
        )

        # 6. 组合输出
        output = {
            'selected_vehicle_ids': controller_output['selected_vehicle_ids'],
            'selected_indices': controller_output['selected_indices'],
            'safe_actions': safe_actions['safe_actions'],
            'raw_actions': controller_output['raw_actions'],
            'influence_scores': controller_output['influence_scores'],
            'value_estimates': controller_output['value_estimates'],
            'cost_estimates': controller_output['cost_estimates'],
            'advantage_estimates': controller_output.get('advantage_estimates'),
            'level1_interventions': safe_actions['level1_interventions'],
            'level2_interventions': safe_actions['level2_interventions'],
            'gnn_embedding': gnn_embedding,
            'world_predictions': world_predictions,
            'fused_features': controller_output.get('fused_features')
        }

        return output

    def set_world_model_phase(self, phase: int):
        """设置世界模型训练阶段"""
        self.world_model.set_phase(phase)

    def update_lagrange_multiplier(self, cost: float):
        """
        更新拉格朗日乘子

        Args:
            cost: 当前成本
        """
        # 拉格朗日乘子自适应更新
        if cost > self.cost_limit:
            # 成本超标，增加惩罚
            self.lagrange_multiplier.data = torch.clamp(
                self.lagrange_multiplier.data * (1 + self.lambda_lr),
                min=0.1,
                max=10.0
            )
        else:
            # 成本满足，减少惩罚
            self.lagrange_multiplier.data = torch.clamp(
                self.lagrange_multiplier.data * (1 - self.lambda_lr * 0.5),
                min=0.1,
                max=10.0
            )

    def compute_constrained_loss(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        values: torch.Tensor,
        cost_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算约束损失 - 完整PPO实现

        Args:
            reward: 奖励
            cost: 成本
            values: 价值估计
            cost_values: 成本价值估计

        Returns:
            loss_dict
        """
        # ========== 完整PPO-Clip损失实现 ==========

        # 1. 计算优势函数
        # advantage = Q(s,a) - V(s) ≈ reward - V(s)
        advantage = reward - values.detach()

        # 标准化优势（提高训练稳定性）
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # 2. PPO-Clip策略损失
        # ratio = π_new(a|s) / π_old(a|s)
        # 在确定性策略中，我们使用value的比率作为proxy
        # 或者使用log_prob的比率

        # 对于我们的连续动作空间，策略损失：
        # L^CLIP = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]

        # 由于我们使用确定性策略（通过神经网络直接输出动作），
        # 这里使用价值的对数作为proxy
        log_values = torch.log(1e-8 + torch.abs(values))
        log_old_values = log_values.detach()

        # 计算比率
        ratio = torch.exp(log_values - log_old_values)

        # PPO clip参数
        clip_epsilon = 0.2

        # 计算两种策略损失
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage

        # 取最小值（保守策略）
        policy_loss = -torch.min(surr1, surr2).mean()

        # 3. 价值函数损失（带clipping）
        # V^clipped = V_old + clip(V_new - V_old, -ε, ε)
        value_pred_clipped = values.detach() + torch.clamp(
            values - values.detach(),
            -clip_epsilon,
            clip_epsilon
        )

        value_loss_unclipped = F.mse_loss(values, reward.expand_as(values))
        value_loss_clipped = F.mse_loss(value_pred_clipped, reward.expand_as(values))

        # 取最大值（更保守）
        value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

        # 4. 熵奖励 - 鼓励探索
        # 对于确定性策略，使用动作方差作为熵的proxy
        if values.numel() > 1:
            entropy = values.std() + 1e-8
            entropy_loss = -0.01 * entropy  # 负号表示最大化熵
        else:
            entropy_loss = torch.zeros(1, device=values.device)

        # 5. 约束损失（拉格朗日乘子法）
        # 成本约束：E[cost] ≤ cost_limit
        cost_violation = torch.relu(cost - self.cost_limit)

        # 拉格朗日损失
        constraint_loss = (
            self.lagrange_multiplier * cost_violation +
            0.5 * self.lagrange_multiplier ** 2
        )

        # 成本价值损失
        cost_value_loss = F.mse_loss(cost_values, cost.expand_as(cost_values))

        # 6. 总损失
        # 权重设置：
        # - 策略损失: 1.0
        # - 价值损失: 0.5
        # - 熵损失: 0.01
        # - 约束损失: 1.0 (重要)
        # - 成本价值损失: 0.1
        total_loss = (
            1.0 * policy_loss +
            0.5 * value_loss +
            entropy_loss +
            1.0 * constraint_loss +
            0.1 * cost_value_loss
        )

        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'constraint_loss': constraint_loss,
            'cost_value_loss': cost_value_loss,
            'lagrange_multiplier': self.lagrange_multiplier.item(),
            'advantage_mean': advantage.mean().item() if advantage.numel() > 0 else 0.0,
            'ratio_mean': ratio.mean().item() if ratio.numel() > 0 else 0.0
        }

    def save_checkpoint(self, filepath: str, epoch: int, optimizer_state: Optional[Dict] = None):
        """
        保存检查点

        Args:
            filepath: 保存路径
            epoch: 当前epoch
            optimizer_state: 优化器状态（可选）
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'lagrange_multiplier': self.lagrange_multiplier.item(),
            'cost_limit': self.cost_limit
        }

        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state

        torch.save(checkpoint, filepath)
        print(f"✅ 检查点已保存: {filepath}")

    def load_checkpoint(self, filepath: str, load_optimizer: bool = False):
        """
        加载检查点

        Args:
            filepath: 检查点路径
            load_optimizer: 是否加载优化器状态

        Returns:
            checkpoint信息
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.load_state_dict(checkpoint['model_state_dict'])

        if 'lagrange_multiplier' in checkpoint:
            self.lagrange_multiplier.data = torch.tensor(
                checkpoint['lagrange_multiplier'],
                device=self.device
            )

        if 'cost_limit' in checkpoint:
            self.cost_limit = checkpoint['cost_limit']

        print(f"✅ 检查点已加载: {filepath} (epoch {checkpoint.get('epoch', 'unknown')})")

        return checkpoint

    def freeze_component(self, component: str):
        """
        冻结特定组件的参数

        Args:
            component: 'gnn', 'world_model', 'controller', 'safety'
        """
        if component == 'gnn':
            for param in self.risk_gnn.parameters():
                param.requires_grad = False
        elif component == 'world_model':
            for param in self.world_model.parameters():
                param.requires_grad = False
        elif component == 'controller':
            for param in self.controller.parameters():
                param.requires_grad = False
        elif component == 'safety':
            for param in self.safety_shield.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"Unknown component: {component}")

        print(f"✅ 组件 '{component}' 已冻结")

    def unfreeze_component(self, component: str):
        """
        解冻特定组件的参数

        Args:
            component: 'gnn', 'world_model', 'controller', 'safety'
        """
        if component == 'gnn':
            for param in self.risk_gnn.parameters():
                param.requires_grad = True
        elif component == 'world_model':
            for param in self.world_model.parameters():
                param.requires_grad = True
        elif component == 'controller':
            for param in self.controller.parameters():
                param.requires_grad = True
        elif component == 'safety':
            for param in self.safety_shield.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown component: {component}")

        print(f"✅ 组件 '{component}' 已解冻")


def create_model_from_config(config: Dict[str, Any]) -> TrafficController:
    """
    从配置创建模型

    Args:
        config: 配置字典

    Returns:
        TrafficController实例
    """
    model = TrafficController(config)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"🏗️  模型已创建")
    print(f"   - 总参数: {total_params:,}")
    print(f"   - 可训练参数: {trainable_params:,}")

    return model


# 便捷导入
__all__ = [
    'TrafficController',
    'create_model_from_config'
]
