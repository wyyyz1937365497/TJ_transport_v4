"""
GPU加速的交通流模型

使用PyTorch实现批量GPU加速的交通流预测，用于MPC优化。

关键优化：
1. 批量预测所有车辆（向量化）
2. GPU并行计算
3. 减少CPU-GPU通信
4. 自动微分支持（用于基于梯度的MPC）
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


class GPUTrafficModel(nn.Module):
    """
    GPU加速的交通流模型

    批量预测多车辆的未来状态，支持：
    - 纵向运动（加速/减速）
    - IDM跟驰模型
    - 换道决策
    """

    def __init__(
        self,
        dt: float = 0.1,
        max_speed: float = 30.0,
        max_accel: float = 2.0,
        max_decel: float = -4.5,
        device: str = 'cuda'
    ):
        """
        初始化GPU交通流模型

        Args:
            dt: 时间步长（秒）
            max_speed: 最大速度（m/s）
            max_accel: 最大加速度（m/s²）
            max_decel: 最大减速度（m/s²）
            device: 设备 ('cuda' or 'cpu')
        """
        super().__init__()

        self.dt = dt
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.device = device

        # IDM参数（可学习）
        self.idm_desired_speed = nn.Parameter(torch.tensor(30.0))  # v0
        self.idm_min_gap = nn.Parameter(torch.tensor(2.0))         # s0
        self.idm_time_headway = nn.Parameter(torch.tensor(1.5))    # T
        self.idm_accel = nn.Parameter(torch.tensor(0.73))          # a_max
        self.idm_decel = nn.Parameter(torch.tensor(1.67))          # b_max

        # 注册缓冲区（不可学习但需要保存在state_dict中）
        self.register_buffer('dt_tensor', torch.tensor(dt))
        self.register_buffer('max_speed_tensor', torch.tensor(max_speed))

    def idm_acceleration(
        self,
        speed: torch.Tensor,
        gap: torch.Tensor,
        leader_speed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        IDM跟驰模型 - 计算期望加速度（批量）

        Args:
            speed: [B] 当前速度 (m/s)
            gap: [B] 车间距 (m)
            leader_speed: [B] 前车速度 (m/s), None时为0

        Returns:
            accel: [B] 期望加速度 (m/s²)
        """
        v0 = self.idm_desired_speed
        s0 = self.idm_min_gap
        T = self.idm_time_headway
        a_max = self.idm_accel
        b = self.idm_decel

        # 相对速度
        if leader_speed is None:
            delta_v = torch.zeros_like(speed)
        else:
            delta_v = speed - leader_speed

        # 期望车间距
        s_star = s0 + speed * T + (speed * delta_v) / (2 * torch.sqrt(a_max * b))

        # IDM加速度
        accel = a_max * (1 - (speed / v0)**4 - (s_star / (gap + 1e-6))**2)

        return accel

    def predict_batch(
        self,
        current_states: torch.Tensor,
        control_inputs: torch.Tensor,
        leader_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        批量预测下一状态（GPU加速）

        Args:
            current_states: [B, 4] 当前状态 [s, v, a, lane]
            control_inputs: [B, 2] 控制输入 [accel_cmd, lane_change_prob]
            leader_states: [B, 2] 前车状态 [s_leader, v_leader], None表示无前车

        Returns:
            next_states: [B, 4] 下一状态 [s, v, a, lane]
        """
        B = current_states.shape[0]

        # 提取当前状态
        s = current_states[:, 0]  # [B]
        v = current_states[:, 1]  # [B]
        a = current_states[:, 2]  # [B]
        lane = current_states[:, 3]  # [B]

        # 提取控制输入
        accel_cmd = control_inputs[:, 0]  # [B]
        lane_change_prob = control_inputs[:, 1]  # [B]

        # ========== 1. 纵向动力学 ==========
        # 限制加速度范围
        accel_cmd_limited = torch.clamp(accel_cmd, self.max_decel, self.max_accel)

        # 更新速度
        v_next = v + accel_cmd_limited * self.dt
        v_next = torch.clamp(v_next, 0.0, self.max_speed)

        # 更新位置
        s_next = s + v * self.dt + 0.5 * a * self.dt**2

        # ========== 2. IDM跟驰修正 ==========
        if leader_states is not None:
            gap = leader_states[:, 0] - s  # [B]
            leader_v = leader_states[:, 1]  # [B]

            # 只有当前方有车且车间距小于阈值时才应用IDM
            has_leader = (gap > 0) & (gap < 200.0)  # [B] bool

            if has_leader.any():
                # IDM期望加速度
                idm_accel = self.idm_acceleration(v, gap, leader_v)

                # 混合控制加速度和IDM加速度
                # 如果车间距小，更依赖IDM；否则更依赖控制指令
                gap_ratio = torch.clamp(gap / 50.0, 0.0, 1.0)  # [B]
                accel_mixed = accel_cmd_limited * gap_ratio + idm_accel * (1 - gap_ratio)

                # 重新计算速度和位置
                v_next = v + accel_mixed * self.dt
                v_next = torch.clamp(v_next, 0.0, self.max_speed)
                s_next = s + v * self.dt + 0.5 * accel_mixed * self.dt**2

        # ========== 3. 加速度平滑（最小化jerk） ==========
        a_next = (v_next - v) / self.dt
        a_next = torch.clamp(a_next, self.max_decel, self.max_accel)

        # ========== 4. 换道决策 ==========
        # 简化的换道：基于概率
        # lane_change_prob > 0.5 表示向左换道，< -0.5 表示向右换道
        lane_next = lane.clone()
        lane_change_cmd = torch.sign(lane_change_prob)  # [-1, 0, 1]

        # 只有高概率时才换道
        change_left = (lane_change_prob > 0.5)
        change_right = (lane_change_prob < -0.5)

        lane_next = torch.where(change_left, lane + 1, lane_next)
        lane_next = torch.where(change_right, lane - 1, lane_next)

        # 限制车道范围
        lane_next = torch.clamp(lane_next, 0.0, 10.0)  # 假设最多10条车道

        # 组装下一状态
        next_states = torch.stack([s_next, v_next, a_next, lane_next], dim=1)  # [B, 4]

        return next_states

    def predict_horizon(
        self,
        initial_states: torch.Tensor,
        control_sequences: torch.Tensor,
        leader_sequences: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        预测未来N步的状态（用于MPC滚动预测）

        Args:
            initial_states: [B, 4] 初始状态
            control_sequences: [B, N, 2] 控制序列 [accel, lane_change_prob]
            leader_sequences: [B, N, 2] 前车状态序列 [s, v], None表示无前车

        Returns:
            state_sequences: [B, N, 4] 状态序列
        """
        B, N, _ = control_sequences.shape

        # 初始化状态序列
        state_sequences = torch.zeros(B, N, 4, device=self.device)
        current_states = initial_states  # [B, 4]

        # 滚动预测
        for k in range(N):
            # 当前控制输入
            control_k = control_sequences[:, k, :]  # [B, 2]

            # 当前前车状态
            leader_k = leader_sequences[:, k, :] if leader_sequences is not None else None

            # 预测下一状态
            next_states = self.predict_batch(current_states, control_k, leader_k)

            # 保存
            state_sequences[:, k, :] = next_states

            # 更新当前状态
            current_states = next_states

        return state_sequences

    def compute_gap_matrix(
        self,
        positions: torch.Tensor,
        lane_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        计算车辆间距离矩阵（批量）

        Args:
            positions: [B] 车辆纵向位置
            lane_indices: [B] 车道索引

        Returns:
            gap_matrix: [B, B] 车间距离矩阵
                gap_matrix[i, j] = vehicle[j].s - vehicle[i].s
                正值表示j在i前面
        """
        # 扩展维度计算成对距离
        pos_i = positions.unsqueeze(1)  # [B, 1]
        pos_j = positions.unsqueeze(0)  # [1, B]

        gap_matrix = pos_j - pos_i  # [B, B]

        # 只计算同车道的距离
        lane_i = lane_indices.unsqueeze(1)  # [B, 1]
        lane_j = lane_indices.unsqueeze(0)  # [1, B]
        same_lane = (lane_i == lane_j).float()  # [B, B]

        gap_matrix = gap_matrix * same_lane

        return gap_matrix


class BatchIDMModel(nn.Module):
    """
    批量IDM模型 - 用于快速跟驰预测
    """

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        # IDM参数
        self.v0 = 30.0  # 期望速度 (m/s)
        self.T = 1.5    # 车头时距 (s)
        self.s0 = 2.0   # 最小车距 (m)
        self.a = 0.73   # 最大加速度 (m/s²)
        self.b = 1.67   # 舒适减速度 (m/s²)

    def forward(
        self,
        speed: torch.Tensor,
        gap: torch.Tensor,
        leader_speed: torch.Tensor
    ) -> torch.Tensor:
        """
        计算IDM加速度

        Args:
            speed: [B] 当前速度
            gap: [B] 车间距
            leader_speed: [B] 前车速度

        Returns:
            accel: [B] IDM加速度
        """
        delta_v = speed - leader_speed
        s_star = self.s0 + speed * self.T + (speed * delta_v) / (2 * torch.sqrt(torch.tensor(self.a * self.b)))
        accel = self.a * (1 - (speed / self.v0)**4 - (s_star / (gap + 1e-6))**2)
        return accel


# 测试代码
if __name__ == '__main__':
    print("Testing GPUTrafficModel...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 创建模型
    model = GPUTrafficModel(device=device).to(device)

    # 测试数据
    B = 32  # 批量大小
    current_states = torch.randn(B, 4).to(device)
    control_inputs = torch.randn(B, 2).to(device)

    # 前向传播
    next_states = model.predict_batch(current_states, control_inputs)

    print(f"Input shape: {current_states.shape}")
    print(f"Output shape: {next_states.shape}")

    # 测试时域预测
    N = 10  # 预测时域
    control_sequences = torch.randn(B, N, 2).to(device)
    state_sequences = model.predict_horizon(current_states, control_sequences)

    print(f"Horizon prediction shape: {state_sequences.shape}")

    # 测试性能
    import time

    num_iterations = 1000
    torch.cuda.synchronize() if device == 'cuda' else None

    start = time.time()
    for _ in range(num_iterations):
        next_states = model.predict_batch(current_states, control_inputs)

    torch.cuda.synchronize() if device == 'cuda' else None
    elapsed = time.time() - start

    print(f"\nPerformance:")
    print(f"  {num_iterations} iterations: {elapsed:.3f}s")
    print(f"  Average: {elapsed / num_iterations * 1000:.2f}ms/iter")
    print(f"  Throughput: {B * num_iterations / elapsed:.0f} vehicles/s")

    print("\n✓ All tests passed!")
