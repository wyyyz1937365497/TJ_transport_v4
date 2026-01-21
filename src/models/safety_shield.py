"""
安全屏障（Safety Shield）- 双层动作过滤

功能：
1. Level 1: 物理限制裁剪（加速度边界）
2. Level 2: TTC检查（碰撞风险检测）
3. 轻量级实现（CPU执行，避免GPU传输开销）

核心思想：
- 在动作执行前进行安全检查
- 违反物理限制 → 裁剪到合法范围
- 碰撞风险 → 强制制动
- 返回安全奖励以引导策略学习安全行为

参考文献：
- Alshiekh et al. (2018) "Safe Reinforcement Learning via Shielding"
- guarantees temporal logic constraints
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class SafetyShield:
    """
    双层安全屏障

    不继承nn.Module，因为需要轻量级CPU实现
    """

    def __init__(
        self,
        # Level 1参数
        min_accel: float = -4.5,  # 最小减速度（m/s²）
        max_accel: float = 2.0,   # 最大加速度（m/s²）
        min_decel: float = -4.5,  # 紧急制动减速度

        # Level 2参数
        ttc_threshold: float = 2.0,  # TTC阈值（秒）
        min_distance: float = 2.0,    # 最小车距（米）
        reaction_time: float = 0.5,   # 反应时间（秒）

        # 奖励参数
        safety_reward: float = 0.1,    # 安全过滤奖励
        violation_penalty: float = -1.0,  # 违规惩罚

        # 执行模式
        device: str = 'cpu'  # 强制使用CPU避免传输开销
    ):
        """
        Args:
            min_accel: 最小加速度（舒适制动）
            max_accel: 最大加速度
            min_decel: 紧急制动减速度
            ttc_threshold: TTC阈值（低于此值触发制动）
            min_distance: 最小安全距离
            reaction_time: 驾驶员反应时间
            safety_reward: 成功过滤动作的奖励
            violation_penalty: 违规惩罚
            device: 执行设备（强制CPU）
        """
        self.min_accel = min_accel
        self.max_accel = max_accel
        self.min_decel = min_decel

        self.ttc_threshold = ttc_threshold
        self.min_distance = min_distance
        self.reaction_time = reaction_time

        self.safety_reward = safety_reward
        self.violation_penalty = violation_penalty

        self.device = device

        # 统计信息
        self.stats = {
            'level1_clips': 0,
            'level2_brakes': 0,
            'total_checks': 0
        }

    def level1_physical_clipping(
        self,
        actions: torch.Tensor,
        vehicle_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Level 1: 物理限制裁剪

        检查并裁剪超出物理限制的加速度

        Args:
            actions: [B, N, 2] 动作 (accel, lane_change)
            vehicle_states: [B, N, 9] 车辆状态

        Returns:
            clipped_actions: [B, N, 2] 裁剪后的动作
            clip_mask: [B, N] 是否被裁剪
        """
        # 分离加速度和换道
        accel = actions[..., 0]  # [B, N]
        lane_change = actions[..., 1]  # [B, N]

        # 裁剪加速度到物理限制范围
        accel_clipped = torch.clamp(accel, self.min_accel, self.max_accel)

        # 记录被裁剪的动作
        clip_mask = (accel != accel_clipped)  # [B, N]

        # 重新组合动作
        clipped_actions = torch.stack([accel_clipped, lane_change], dim=-1)  # [B, N, 2]

        return clipped_actions, clip_mask

    def level2_ttc_check(
        self,
        actions: torch.Tensor,
        vehicle_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Level 2: TTC检查

        检测碰撞风险，必要时触发紧急制动

        Args:
            actions: [B, N, 2] 动作
            vehicle_states: [B, N, 9] 车辆状态
                - [0]: x (纵向位置)
                - [2]: vx (纵向速度)
                - [6]: lane (车道索引)

        Returns:
            safe_actions: [B, N, 2] 安全动作（可能已制动）
            brake_mask: [B, N] 是否触发制动
            ttc_info: TTC信息字典
        """
        batch_size, num_vehicles, _ = vehicle_states.shape
        device = vehicle_states.device

        # 提取关键信息
        x = vehicle_states[:, :, 0]  # [B, N] 纵向位置
        vx = vehicle_states[:, :, 2]  # [B, N] 纵向速度
        lanes = vehicle_states[:, :, 6].long()  # [B, N] 车道索引

        # 计算TTC矩阵
        ttc_matrix = self._compute_ttc_matrix(x, vx, lanes)  # [B, N, N]

        # 找到最小TTC（排除自己）
        # 对每个batch填充对角线为inf
        for b in range(ttc_matrix.size(0)):
            ttc_matrix[b].fill_diagonal_(float('inf'))
        min_ttc, _ = ttc_matrix.min(dim=-1)  # [B, N]

        # 检测碰撞风险
        brake_mask = min_ttc < self.ttc_threshold  # [B, N]

        # 应用制动
        accel = actions[..., 0].clone()  # [B, N]
        lane_change = actions[..., 1].clone()  # [B, N]

        # 触发制动的车辆：设置加速度为紧急制动
        accel[brake_mask] = self.min_decel
        # 制动时禁止换道
        lane_change[brake_mask] = 0.0

        # 组合安全动作
        safe_actions = torch.stack([accel, lane_change], dim=-1)  # [B, N, 2]

        # TTC信息
        ttc_info = {
            'min_ttc': min_ttc,  # [B, N]
            'ttc_matrix': ttc_matrix,  # [B, N, N]
            'brake_mask': brake_mask  # [B, N]
        }

        return safe_actions, brake_mask, ttc_info

    def _compute_ttc_matrix(
        self,
        x: torch.Tensor,
        vx: torch.Tensor,
        lanes: torch.Tensor
    ) -> torch.Tensor:
        """
        计算TTC矩阵

        TTC = distance / relative_velocity
        只考虑同车道前车

        Args:
            x: [B, N] 纵向位置
            vx: [B, N] 纵向速度
            lanes: [B, N] 车道索引

        Returns:
            ttc_matrix: [B, N, N] TTC矩阵
        """
        batch_size, num_vehicles = x.shape
        device = x.device

        # 初始化TTC矩阵
        ttc_matrix = torch.full(
            (batch_size, num_vehicles, num_vehicles),
            float('inf'),
            device=device
        )

        # 对每辆车计算TTC
        for i in range(num_vehicles):
            # 车辆i的信息
            x_i = x[:, i]  # [B]
            vx_i = vx[:, i]  # [B]
            lane_i = lanes[:, i]  # [B]

            # 找到同车道的其他车辆
            for j in range(num_vehicles):
                if i == j:
                    continue

                # 车辆j的信息
                x_j = x[:, j]  # [B]
                vx_j = vx[:, j]  # [B]
                lane_j = lanes[:, j]  # [B]

                # 只考虑同车道
                same_lane = (lane_i == lane_j)  # [B]

                # 计算距离和相对速度
                # j在i前面：x_j > x_i
                distance = x_j - x_i  # [B]
                relative_vx = vx_i - vx_j  # [B] (正数表示接近)

                # 只考虑前车（distance > 0）且接近（relative_vx > 0）
                valid = same_lane & (distance > self.min_distance) & (relative_vx > 0)  # [B]

                # 计算TTC
                ttc = distance / (relative_vx + 1e-6)  # [B]

                # 只在有效时更新
                ttc_matrix[:, i, j] = torch.where(
                    valid,
                    ttc,
                    torch.full_like(ttc, float('inf'))
                )

        return ttc_matrix

    def filter_actions(
        self,
        actions: torch.Tensor,
        vehicle_states: torch.Tensor,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        完整的安全过滤流程

        Args:
            actions: [B, N, 2] 原始动作
            vehicle_states: [B, N, 9] 车辆状态
            return_details: 是否返回详细信息

        Returns:
            Dict:
                'safe_actions': [B, N, 2] 安全动作
                'safety_reward': [B] 安全奖励
                'level1_mask': [B, N] Level 1裁剪掩码（可选）
                'level2_mask': [B, N] Level 2制动掩码（可选）
                'ttc_info': TTC信息（可选）
        """
        batch_size = actions.size(0)
        device = actions.device

        # 更新统计
        self.stats['total_checks'] += batch_size

        # 保存原始动作（用于计算奖励）
        original_actions = actions.clone()

        # Level 1: 物理限制裁剪
        clipped_actions, level1_mask = self.level1_physical_clipping(
            actions, vehicle_states
        )

        self.stats['level1_clips'] += level1_mask.sum().item()

        # Level 2: TTC检查
        safe_actions, level2_mask, ttc_info = self.level2_ttc_check(
            clipped_actions, vehicle_states
        )

        self.stats['level2_brakes'] += level2_mask.sum().item()

        # 计算安全奖励
        # Level 1违规：轻微惩罚
        level1_penalty = level1_mask.any(dim=-1).float() * self.violation_penalty * 0.5

        # Level 2违规：严重惩罚
        level2_penalty = level2_mask.any(dim=-1).float() * self.violation_penalty

        # 总奖励
        safety_reward = self.safety_reward + level1_penalty + level2_penalty  # [B]

        # 构建返回字典
        result = {
            'safe_actions': safe_actions,
            'safety_reward': safety_reward
        }

        if return_details:
            result['level1_mask'] = level1_mask
            result['level2_mask'] = level2_mask
            result['ttc_info'] = ttc_info

        return result

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'level1_clips': 0,
            'level2_brakes': 0,
            'total_checks': 0
        }

    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        stats = self.stats.copy()

        if stats['total_checks'] > 0:
            stats['level1_rate'] = stats['level1_clips'] / stats['total_checks']
            stats['level2_rate'] = stats['level2_brakes'] / stats['total_checks']
        else:
            stats['level1_rate'] = 0.0
            stats['level2_rate'] = 0.0

        return stats


def create_safety_shield(
    min_accel: float = -4.5,
    max_accel: float = 2.0,
    ttc_threshold: float = 2.0,
    device: str = 'cpu'
) -> SafetyShield:
    """
    创建安全屏障的工厂函数

    Args:
        min_accel: 最小加速度
        max_accel: 最大加速度
        ttc_threshold: TTC阈值
        device: 执行设备

    Returns:
        safety_shield: SafetyShield实例
    """
    return SafetyShield(
        min_accel=min_accel,
        max_accel=max_accel,
        ttc_threshold=ttc_threshold,
        device=device
    )
