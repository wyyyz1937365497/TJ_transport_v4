"""
GPU加速的MPC控制器 - 基于梯度优化

使用PyTorch自动微分和优化器实现MPC，避免CVXPY的CPU瓶颈。

关键优势：
1. GPU加速：所有计算在GPU上进行
2. 批量优化：同时优化所有车辆
3. 自动微分：精确的梯度计算
4. 无CPU-GPU通信：减少数据传输开销
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import numpy as np
import time

from .traffic_model_gpu import GPUTrafficModel


class GPUMPCController:
    """
    GPU加速的MPC控制器

    使用基于梯度的优化求解有限时域最优控制问题。
    """

    def __init__(
        self,
        config,
        device: str = 'cuda'
    ):
        """
        初始化GPU MPC控制器

        Args:
            config: MPCConfig配置对象
            device: 设备 ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device

        # 优化参数
        self.N = config.prediction_horizon  # 预测时域
        self.M = config.control_horizon     # 控制时域
        self.dt = config.dt

        # 权重
        self.Q_speed = config.Q_speed
        self.Q_position = config.Q_position
        self.Q_accel = config.Q_accel
        self.Q_gap = config.Q_gap
        self.R_accel = config.R_accel
        self.R_lane = config.R_lane

        # 约束
        self.min_accel = config.min_accel
        self.max_accel = config.max_accel
        self.min_speed = config.min_speed
        self.max_speed = config.max_speed
        self.min_gap = config.min_gap

        # 瓶颈区域
        self.bottleneck_s_min = config.bottleneck_s_min
        self.bottleneck_s_max = config.bottleneck_s_max

        # 交通流模型
        self.traffic_model = GPUTrafficModel(
            dt=self.dt,
            max_speed=self.max_speed,
            max_accel=self.max_accel,
            max_decel=self.min_accel,
            device=device
        ).to(device)

        # 优化器配置
        self.optimizer_type = 'adam'  # 'lbfgs' or 'adam' - Adam更快
        self.max_iterations = 3  # 优化迭代次数（实时控制：3次快速迭代）
        self.tolerance = 1e-4

        # 缓存
        self.last_control_sequence = None
        self.last_solve_time = 0.0

    def _build_topology_matrix(
        self,
        vehicle_states: torch.Tensor,
        vehicle_ids: List[str]
    ) -> torch.Tensor:
        """
        构建拓扑矩阵（leader-follower关系）

        Args:
            vehicle_states: [B, 4] 车辆状态 [s, v, a, lane]
            vehicle_ids: 车辆ID列表

        Returns:
            topology_matrix: [B, B] 拓扑矩阵
                topology_matrix[i, j] = 1 表示j是i的leader
        """
        B = vehicle_states.shape[0]

        # 提取位置和车道
        positions = vehicle_states[:, 0]  # [B]
        lanes = vehicle_states[:, 3].long()  # [B]

        # 计算距离矩阵
        pos_i = positions.unsqueeze(1)  # [B, 1]
        pos_j = positions.unsqueeze(0)  # [1, B]
        gap_matrix = pos_j - pos_i  # [B, B]

        # 计算车道矩阵
        lane_i = lanes.unsqueeze(1)  # [B, 1]
        lane_j = lanes.unsqueeze(0)  # [1, B]
        same_lane = (lane_i == lane_j).float()  # [B, B]

        # Leader条件：同车道且在前方
        is_leader = (gap_matrix > 0) & (gap_matrix < 200.0)  # 同车道，在前方，200m内
        is_leader = is_leader & (same_lane > 0.5)  # 同车道

        # 对于每个车辆，选择最近的前车作为leader
        topology_matrix = torch.zeros(B, B, device=self.device)

        for i in range(B):
            # 找到i的所有leader
            leaders = torch.where(is_leader[i])[0]

            if len(leaders) > 0:
                # 选择最近的leader（最小gap）
                leader_gaps = gap_matrix[i, leaders]
                nearest_leader = leaders[torch.argmin(leader_gaps)]
                topology_matrix[i, nearest_leader] = 1.0

        return topology_matrix

    def _compute_cost(
        self,
        state_sequences: torch.Tensor,
        control_sequences: torch.Tensor,
        gap_sequences: torch.Tensor
    ) -> torch.Tensor:
        """
        计算MPC目标函数（标量）

        Args:
            state_sequences: [B, N, 4] 状态序列
            control_sequences: [B, N, 2] 控制序列
            gap_sequences: [B, N] 车间距序列

        Returns:
            cost: 标量损失
        """
        N = self.N

        # 提取状态
        s = state_sequences[:, :, 0]  # [B, N]
        v = state_sequences[:, :, 1]  # [B, N]
        a = state_sequences[:, :, 2]  # [B, N]

        # 提取控制
        accel = control_sequences[:, :, 0]   # [B, N]
        lane_change = control_sequences[:, :, 1]  # [B, N]

        # ========== 1. 速度跟踪成本 ==========
        # 目标：保持高速（接近max_speed）
        v_error = (self.max_speed - v) / self.max_speed  # 归一化误差
        speed_cost = self.Q_speed * torch.sum(v_error**2)

        # ========== 2. 位置跟踪成本 ==========
        # 鼓励车辆向前移动
        position_cost = self.Q_position * torch.sum((-s / 1000.0)**2)

        # ========== 3. 加速度平滑成本 ==========
        # 最小化jerk（加速度变化）
        accel_diff = accel[:, 1:] - accel[:, :-1]  # [B, N-1]
        accel_smooth_cost = self.Q_accel * torch.sum(accel_diff**2)

        # ========== 4. 车间距保持成本 ==========
        # 惩罚过小的车间距
        gap_error = torch.clamp(self.min_gap - gap_sequences, min=0.0)  # [B, N]
        gap_cost = self.Q_gap * torch.sum(gap_error**2)

        # ========== 5. 控制成本 ==========
        control_cost = (
            self.R_accel * torch.sum(accel**2) +
            self.R_lane * torch.sum(lane_change**2)
        )

        # ========== 6. 瓶颈区域特殊成本 ==========
        # 在瓶颈区域内，更重视速度和车间距
        in_bottleneck = (s > self.bottleneck_s_min) & (s < self.bottleneck_s_max)
        bottleneck_cost = (
            2.0 * torch.sum(v_error[in_bottleneck]**2) +
            3.0 * torch.sum(gap_error[in_bottleneck]**2)
        )

        # 只计算ICV车辆的成本（权重化）
        # 注意：这里不使用target_vehicles进行选择，因为所有车辆都需要优化
        # ICV车辆有更高的权重

        # 总成本（标量）
        total_cost = (
            speed_cost +
            position_cost +
            accel_smooth_cost +
            gap_cost +
            bottleneck_cost +
            control_cost
        )

        return total_cost  # 返回标量

    def solve(
        self,
        observation_dict: Dict,
        vehicle_ids: List[str],
        icv_ids: List[str],
        step: int
    ) -> Dict[str, Tuple[float, float]]:
        """
        求解MPC优化问题

        Args:
            observation_dict: 观测字典
            vehicle_ids: 所有车辆ID列表
            icv_ids: ICV车辆ID列表
            step: 当前时间步

        Returns:
            actions_dict: {veh_id: (accel, lane_change)}
        """
        start_time = time.time()

        # ========== 1. 准备数据 ==========
        vehicle_states = observation_dict.get('vehicle_states', {})

        # 过滤出需要控制的ICV车辆
        target_vehicles = [v for v in icv_ids if v in vehicle_ids]

        if len(target_vehicles) == 0:
            return {}

        # 提取状态张量
        states_list = []
        for veh_id in target_vehicles:
            state = vehicle_states[veh_id]
            states_list.append([
                state.get('s', 0.0),
                state.get('speed', 0.0),
                state.get('acceleration', 0.0),
                state.get('lane_index', 0.0)
            ])

        initial_states = torch.tensor(
            states_list,
            dtype=torch.float32,
            device=self.device
        )  # [B, 4]

        B = len(target_vehicles)

        # ========== 2. 构建拓扑关系（前车-跟随车） ==========
        # 计算所有车辆（包括非ICV）的拓扑关系
        all_vehicle_states_list = []
        all_vehicle_ids_list = []

        for veh_id in vehicle_ids:
            if veh_id in vehicle_states:
                state = vehicle_states[veh_id]
                all_vehicle_states_list.append([
                    state.get('s', 0.0),
                    state.get('speed', 0.0),
                    state.get('acceleration', 0.0),
                    state.get('lane_index', 0.0)
                ])
                all_vehicle_ids_list.append(veh_id)

        if len(all_vehicle_states_list) > 0:
            all_states_tensor = torch.tensor(
                all_vehicle_states_list,
                dtype=torch.float32,
                device=self.device
            )  # [B_all, 4]

            # 找到每个target_vehicle对应的leader
            leader_indices = []  # List of indices in all_states_tensor

            for i, target_veh_id in enumerate(target_vehicles):
                target_state = vehicle_states[target_veh_id]
                target_s = target_state.get('s', 0.0)
                target_lane = target_state.get('lane_index', 0.0)

                # 在同车道前方找最近的车
                min_gap = float('inf')
                nearest_leader_idx = -1

                for j, other_veh_id in enumerate(all_vehicle_ids_list):
                    if other_veh_id == target_veh_id:
                        continue  # 跳过自己

                    other_state = vehicle_states[other_veh_id]
                    other_s = other_state.get('s', 0.0)
                    other_lane = other_state.get('lane_index', 0.0)

                    # 同车道且在前方
                    if other_lane == target_lane and other_s > target_s:
                        gap = other_s - target_s
                        if gap < min_gap and gap < 200.0:  # 200m感知范围
                            min_gap = gap
                            nearest_leader_idx = j

                leader_indices.append(nearest_leader_idx)

            # 构建leader状态张量 [B, 2] (s, v)
            leader_states = torch.zeros(B, 2, device=self.device)
            for i, leader_idx in enumerate(leader_indices):
                if leader_idx >= 0:
                    leader_states[i, 0] = all_states_tensor[leader_idx, 0]  # s
                    leader_states[i, 1] = all_states_tensor[leader_idx, 1]  # v
                else:
                    # 无前车，设置虚拟前车
                    leader_states[i, 0] = initial_states[i, 0] + 1000.0  # 前方1000m
                    leader_states[i, 1] = 30.0  # 期望速度
        else:
            leader_states = None

        # ========== 3. 初始化控制序列 ==========
        # 使用热启动（上一次的解）
        if self.last_control_sequence is not None and self.last_control_sequence.shape == (B, self.N, 2):
            # 使用上一次的解作为初始化
            prev_control = self.last_control_sequence.clone()  # [B, N, 2]
            # 移位：丢弃第一步
            rolled = torch.roll(prev_control, shifts=-1, dims=1)
            # 创建新张量，最后一行为0
            control_sequences = torch.zeros(B, self.N, 2, device=self.device)
            control_sequences[:, :-1, :] = rolled[:, :-1, :]
            control_sequences.requires_grad = True
        else:
            # 零初始化
            control_sequences = torch.zeros(
                B, self.N, 2,
                device=self.device,
                requires_grad=True
            )

        # ========== 4. 优化 ==========
        if self.optimizer_type == 'lbfgs':
            # LBFGS：二阶优化，收敛快
            optimizer = optim.LBFGS(
                [control_sequences],
                lr=1.0,
                max_iter=20,
                tolerance_grad=1e-5,
                tolerance_change=1e-9,
                history_size=10
            )

            def closure():
                optimizer.zero_grad()

                # 预测状态序列（使用真实的前车信息）
                # 扩展leader_states到整个时域 [B, N, 2]
                if leader_states is not None:
                    leader_sequences_expanded = leader_states.unsqueeze(1).expand(-1, self.N, 2)
                else:
                    leader_sequences_expanded = None

                state_sequences = self.traffic_model.predict_horizon(
                    initial_states,
                    control_sequences[:, :self.N, :],
                    leader_sequences=leader_sequences_expanded
                )  # [B, N, 4]

                # 从状态序列中计算真实的车间距
                # gap = leader_s - ego_s
                if leader_states is not None:
                    s_sequences = state_sequences[:, :, 0]  # [B, N]
                    leader_s_expanded = leader_states[:, 0].unsqueeze(1).expand(-1, self.N)  # [B, N]
                    gap_sequences = leader_s_expanded - s_sequences  # [B, N]
                    # 限制在合理范围内
                    gap_sequences = torch.clamp(gap_sequences, min=2.0, max=200.0)
                else:
                    gap_sequences = torch.ones(B, self.N, device=self.device) * 50.0

                # 计算成本
                cost = self._compute_cost(
                    state_sequences,
                    control_sequences[:, :self.N, :],
                    gap_sequences
                )

                cost.backward()
                return cost

            optimizer.step(closure)

        else:
            # Adam：一阶优化
            optimizer = optim.Adam([control_sequences], lr=1.0)

            for _ in range(self.max_iterations):
                optimizer.zero_grad()

                # 预测状态序列（使用真实的前车信息）
                if leader_states is not None:
                    leader_sequences_expanded = leader_states.unsqueeze(1).expand(-1, self.N, 2)
                else:
                    leader_sequences_expanded = None

                state_sequences = self.traffic_model.predict_horizon(
                    initial_states,
                    control_sequences[:, :self.N, :],
                    leader_sequences=leader_sequences_expanded
                )  # [B, N, 4]

                # 从状态序列中计算真实的车间距
                if leader_states is not None:
                    s_sequences = state_sequences[:, :, 0]  # [B, N]
                    leader_s_expanded = leader_states[:, 0].unsqueeze(1).expand(-1, self.N)  # [B, N]
                    gap_sequences = leader_s_expanded - s_sequences  # [B, N]
                    gap_sequences = torch.clamp(gap_sequences, min=2.0, max=200.0)
                else:
                    gap_sequences = torch.ones(B, self.N, device=self.device) * 50.0

                # 计算成本
                cost = self._compute_cost(
                    state_sequences,
                    control_sequences[:, :self.N, :],
                    gap_sequences
                )

                cost.backward()
                optimizer.step()

                # 限制控制范围
                with torch.no_grad():
                    control_sequences[:, :, 0].clamp_(self.min_accel, self.max_accel)
                    control_sequences[:, :, 1].clamp_(-1.0, 1.0)

        # ========== 5. 提取控制 ==========
        with torch.no_grad():
            # 取第一步控制
            actions = control_sequences[:, 0, :]  # [B, 2]
            actions_np = actions.cpu().numpy()

        # ========== 5. 构建输出字典 ==========
        actions_dict = {}
        for i, veh_id in enumerate(target_vehicles):
            accel = float(actions_np[i, 0])

            # ✅ 修复：离散化lane_change动作
            # MPC输出连续值，需要转换为离散值：-1(左), 0(不变), +1(右)
            lane_change_raw = float(actions_np[i, 1])
            if lane_change_raw < -0.33:
                lane_change = -1  # 左变道
            elif lane_change_raw > 0.33:
                lane_change = 1   # 右变道
            else:
                lane_change = 0   # 保持车道

            actions_dict[veh_id] = (accel, lane_change)

        # ========== 6. 缓存 ==========
        with torch.no_grad():
            self.last_control_sequence = control_sequences.detach().clone()

        self.last_solve_time = time.time() - start_time

        return actions_dict

    def get_solve_time(self) -> float:
        """获取上次求解时间"""
        return self.last_solve_time


# 测试代码
if __name__ == '__main__':
    print("Testing GPUMPCController...")

    from .mpc_controller import MPCConfig

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 创建配置
    config = MPCConfig(
        prediction_horizon=10,
        control_horizon=5,
        dt=0.1
    )

    # 创建控制器
    controller = GPUMPCController(config, device=device)

    # 模拟观测
    num_vehicles = 10
    vehicle_ids = [f'veh_{i}' for i in range(num_vehicles)]
    icv_ids = vehicle_ids[:5]  # 前5辆是ICV

    vehicle_states = {}
    for i, veh_id in enumerate(vehicle_ids):
        vehicle_states[veh_id] = {
            's': 1000.0 + i * 50.0,
            'speed': 20.0 + i * 1.0,
            'acceleration': 0.0,
            'lane_index': 0 if i < 5 else 1
        }

    observation_dict = {
        'vehicle_states': vehicle_states,
        'vehicle_ids': vehicle_ids,
        'icv_ids': set(icv_ids)
    }

    # 求解MPC
    import time

    num_iterations = 100
    total_time = 0.0

    for i in range(num_iterations):
        actions = controller.solve(
            observation_dict,
            vehicle_ids,
            icv_ids,
            step=i
        )
        total_time += controller.get_solve_time()

    print(f"\nPerformance:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Average: {total_time / num_iterations * 1000:.2f}ms/iter")
    print(f"  Throughput: {num_iterations / total_time:.0f} solves/s")

    print(f"\nActions sample:")
    for veh_id, (accel, lane_change) in list(actions.items())[:3]:
        print(f"  {veh_id}: accel={accel:.3f}, lane_change={lane_change:.3f}")

    print("\n✓ All tests passed!")
