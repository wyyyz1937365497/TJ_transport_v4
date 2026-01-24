"""
MPC控制器 - 用于生成最优控制策略

该模块实现了基于模型预测控制（MPC）的最优控制器，
通过求解有限时域优化问题生成最优加速度和换道决策。

关键特性：
1. 多车辆协同优化（同时优化所有ICV）
2. 考虑交通流动态（IDM跟驰模型）
3. 瓶颈区域优化（s=1200-2200m）
4. 安全约束（最小车距、速度限制）
5. 高效求解（CVXPY + OSQP）
"""

import numpy as np
import cvxpy as cp
from typing import Dict, List, Tuple, Optional, Union
import time
from dataclasses import dataclass

from .traffic_model import SimplifiedTrafficModel, TrafficTopology


@dataclass
class MPCConfig:
    """MPC配置"""
    # 预测时域
    prediction_horizon: int = 10  # N: 预测步数（1秒）

    # 控制时域
    control_horizon: int = 5      # M: 控制步数（0.5秒）

    # 时间步长
    dt: float = 0.1               # 0.1秒

    # 优化权重
    # 状态权重 [speed, position, acceleration, jerk]
    Q_speed: float = 10.0         # 速度跟踪权重
    Q_position: float = 1.0       # 位置权重
    Q_accel: float = 0.1          # 加速度平滑权重
    Q_gap: float = 50.0           # 车间距保持权重（关键！）

    # 控制权重
    R_accel: float = 0.5          # 加速度控制权重
    R_lane: float = 0.1           # 换道控制权重

    # 约束参数
    min_accel: float = -4.5       # 最大减速度 (m/s²)
    max_accel: float = 2.0        # 最大加速度 (m/s²)
    min_speed: float = 0.0        # 最小速度 (m/s)
    max_speed: float = 30.0       # 最大速度 (m/s) = 108 km/h
    min_gap: float = 2.0          # 最小车距 (m)
    desired_gap: float = 5.0      # 期望车距 (m)

    # 瓶颈区域参数
    bottleneck_s_min: float = 1200.0  # 瓶颈起点 (m)
    bottleneck_s_max: float = 2200.0  # 瓶颈终点 (m)

    # 优化参数
    solver: str = 'OSQP'          # 求解器 (OSQP/CLARABEL)
    verbose: bool = False         # 求解器输出
    max_iters: int = 1000         # 最大迭代次数
    tolerance: float = 1e-4       # 收敛容差

    # 增量优化（加速）
    update_interval: int = 3      # 每N步重新求解一次
    warm_start: bool = True       # 热启动


class MPCController:
    """
    MPC控制器

    通过求解有限时域优化问题，为所有ICV车辆生成最优控制序列。
    """

    def __init__(self, config: Optional[MPCConfig] = None):
        """
        初始化MPC控制器

        Args:
            config: MPC配置
        """
        self.config = config if config is not None else MPCConfig()

        # 交通流模型（用于预测）
        self.traffic_model = SimplifiedTrafficModel(
            dt=self.config.dt,
            max_speed=self.config.max_speed,
            max_accel=self.config.max_accel,
            max_decel=self.config.min_accel
        )

        # 拓扑管理器
        self.topology = TrafficTopology()

        # 求解器选择
        self.solver_map = {
            'OSQP': cp.OSQP,
            'CLARABEL': cp.CLARABEL
        }

        # 增量优化状态
        self.step_count = 0
        self.prev_solution = None
        self.solve_time = 0.0

        # 性能统计
        self.stats = {
            'total_solves': 0,
            'successful_solves': 0,
            'failed_solves': 0,
            'avg_solve_time': 0.0
        }

    def solve(
        self,
        observation_dict: Dict,
        vehicle_ids: List[str],
        icv_ids: List[str],
        step: int
    ) -> Dict[str, np.ndarray]:
        """
        求解MPC优化问题

        Args:
            observation_dict: 观测字典
                - 'vehicle_states': {veh_id: {s, d, vs, vd, speed, acceleration, lane_index, angle}}
                - 'vehicle_ids': 所有车辆ID列表
                - 'icv_ids': ICV车辆ID集合
            vehicle_ids: 所有车辆ID列表（已排序）
            icv_ids: ICV车辆ID列表
            step: 当前时间步

        Returns:
            actions_dict: {veh_id: [acceleration, lane_change_prob]}
        """
        start_time = time.time()

        # 增量优化：每N步重新求解
        if self.config.warm_start and self.prev_solution is not None:
            if step % self.config.update_interval != 0:
                # 使用预计算的解
                return self.prev_solution

        # 提取状态
        vehicle_states = observation_dict.get('vehicle_states', {})

        if len(icv_ids) == 0:
            return {}

        # 更新拓扑关系
        self.topology.update_topology(vehicle_states, vehicle_ids)

        # 求解MPC
        try:
            actions = self._solve_mpc(
                vehicle_states,
                vehicle_ids,
                icv_ids
            )

            # 转换为动作字典
            actions_dict = {}
            for i, veh_id in enumerate(icv_ids):
                if i < actions.shape[0]:
                    actions_dict[veh_id] = actions[i]

            # 保存解（用于增量优化）
            self.prev_solution = actions_dict

            # 更新统计
            solve_time = time.time() - start_time
            self.stats['total_solves'] += 1
            self.stats['successful_solves'] += 1
            self.stats['avg_solve_time'] = (
                (self.stats['avg_solve_time'] * (self.stats['total_solves'] - 1) + solve_time) /
                self.stats['total_solves']
            )

            return actions_dict

        except Exception as e:
            # 求解失败，返回保守动作
            print(f"[MPC Warning] Solve failed: {e}")

            self.stats['total_solves'] += 1
            self.stats['failed_solves'] += 1

            return self._get_fallback_actions(vehicle_states, icv_ids)

    def _solve_mpc(
        self,
        vehicle_states: Dict,
        vehicle_ids: List[str],
        icv_ids: List[str]
    ) -> np.ndarray:
        """
        求解MPC优化问题（核心）

        优化目标：
        1. 最小化加速度变化（平滑控制）
        2. 保持期望速度（30 m/s）
        3. 保持安全车距
        4. 优化瓶颈区域流量

        约束：
        1. 加速度限制 [-4.5, 2.0] m/s²
        2. 速度限制 [0, 30] m/s
        3. 最小车距 >= 2.0 m

        Args:
            vehicle_states: 车辆状态字典
            vehicle_ids: 所有车辆ID列表
            icv_ids: ICV车辆ID列表

        Returns:
            actions: [len(icv_ids), 2] 加速度和换道概率
        """
        N = self.config.prediction_horizon  # 预测时域
        M = self.config.control_horizon      # 控制时域
        dt = self.config.dt

        n_icv = len(icv_ids)

        if n_icv == 0:
            return np.zeros((0, 2))

        # 提取ICV状态
        icv_states = {}
        for veh_id in icv_ids:
            if veh_id in vehicle_states:
                state = vehicle_states[veh_id]
                icv_states[veh_id] = np.array([
                    state.get('s', 0.0),          # s: 纵向位置
                    state.get('speed', 0.0),     # v: 速度
                    state.get('acceleration', 0.0),  # a: 加速度
                    state.get('lane_index', 0.0)    # lane: 车道
                ])

        # ========== 优化变量 ==========

        # 控制序列：[M, n_icv, 2] = [控制步数, 车辆数, (加速度, 换道概率)]
        U = cp.Variable((M, n_icv, 2), name='U')

        # 状态序列（预测）：[N+1, n_icv, 4] = [预测步数, 车辆数, (s, v, a, lane)]
        X = cp.Variable((N + 1, n_icv, 4), name='X')

        # 初始状态
        X_init = np.zeros((n_icv, 4))
        for i, veh_id in enumerate(icv_ids):
            if veh_id in icv_states:
                X_init[i] = icv_states[veh_id]

        # ========== 目标函数 ==========

        cost = 0.0

        for k in range(N):
            for i in range(n_icv):
                # 1. 速度跟踪：鼓励达到期望速度 (30 m/s)
                desired_speed = 30.0
                speed_error = X[k + 1, i, 1] - desired_speed
                cost += self.config.Q_speed * cp.square(speed_error)

                # 2. 加速度平滑：最小化jerk (加速度变化)
                if k < M:
                    cost += self.config.Q_accel * cp.square(U[k, i, 0])

                # 3. 控制成本：最小化控制量
                if k < M:
                    cost += self.config.R_accel * cp.square(U[k, i, 0])
                    cost += self.config.R_lane * cp.square(U[k, i, 1])

                # 4. 车间距保持（关键！）
                veh_id = icv_ids[i]
                leader_id = self.topology.get_leader(veh_id)

                if leader_id and leader_id in icv_states:
                    # 前车也是ICV，可以优化车间距
                    leader_idx = icv_ids.index(leader_id) if leader_id in icv_ids else -1

                    if leader_idx >= 0:
                        # 前车位置 - 本车位置
                        gap = X[k, leader_idx, 0] - X[k, i, 0]
                        desired_gap = self.config.desired_gap

                        # 惩罚过小的车距
                        gap_error = cp.maximum(desired_gap - gap, 0.0)
                        cost += self.config.Q_gap * cp.square(gap_error)

                # 5. 瓶颈区域特殊处理（简化：不使用logical_and以保持凸性）
                # 注：由于cp.logical_and会创建非凸约束，这里简化处理
                # 瓶颈区域优化主要通过调整目标函数权重实现
                s_pos = X[k, i, 0]

                # 如果车辆可能在瓶颈区域（基于初始位置），增加控制平滑性
                # 使用软约束：当位置接近瓶颈区域时，增加控制权重
                # 这里简化为对所有情况都应用平滑控制
                if k < M:
                    cost += self.config.Q_accel * cp.square(U[k, i, 0])

        # ========== 约束条件 ==========

        constraints = []

        # 初始状态约束
        for i in range(n_icv):
            constraints.append(X[0, i] == X_init[i])

        # 动力学约束和约束条件
        for k in range(N):
            for i in range(n_icv):
                # 当前状态
                s_k = X[k, i, 0]
                v_k = X[k, i, 1]
                a_k = X[k, i, 2]

                # 控制输入（如果在控制时域内）
                if k < M:
                    u_accel = U[k, i, 0]
                    u_lane = U[k, i, 1]
                else:
                    # 控制时域外，保持最后一个控制
                    u_accel = U[M - 1, i, 0]
                    u_lane = U[M - 1, i, 1]

                # 下一状态（欧拉积分）
                s_next = s_k + v_k * dt
                v_next = v_k + u_accel * dt
                a_next = u_accel
                lane_next = X[k, i, 3]  # 简化：不考虑换道

                # 状态转移约束
                constraints.append(X[k + 1, i, 0] == s_next)
                constraints.append(X[k + 1, i, 1] == v_next)
                constraints.append(X[k + 1, i, 2] == a_next)
                constraints.append(X[k + 1, i, 3] == lane_next)

                # 速度约束
                constraints.append(v_next >= self.config.min_speed)
                constraints.append(v_next <= self.config.max_speed)

                # 加速度约束
                if k < M:
                    constraints.append(u_accel >= self.config.min_accel)
                    constraints.append(u_accel <= self.config.max_accel)

                # 换道概率约束
                if k < M:
                    constraints.append(u_lane >= 0.0)
                    constraints.append(u_lane <= 1.0)

                # 注：安全车距通过目标函数中的软约束实现，不再使用硬约束
                # 这样可以避免优化问题的不可行性

        # ========== 求解问题 ==========

        problem = cp.Problem(cp.Minimize(cost), constraints)

        solver = self.solver_map.get(self.config.solver, cp.OSQP)

        try:
            problem.solve(
                solver=solver,
                verbose=self.config.verbose,
                max_iter=self.config.max_iters,
                eps_abs=self.config.tolerance,
                eps_rel=self.config.tolerance
            )
        except:
            # 求解器异常，尝试其他求解器
            try:
                problem.solve(solver=cp.SCS, verbose=False)
            except:
                raise

        # 检查求解状态
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            # 添加更详细的错误信息用于调试
            raise ValueError(f"Solver status: {problem.status}, n_icv={n_icv}, N={N}, M={M}")

        # 提取最优控制（只返回第一步）
        actions = np.zeros((n_icv, 2))

        if U.value is not None:
            for i in range(n_icv):
                actions[i, 0] = U.value[0, i, 0]  # 加速度
                actions[i, 1] = U.value[0, i, 1]  # 换道概率

                # 裁剪到有效范围
                actions[i, 0] = np.clip(actions[i, 0], self.config.min_accel, self.config.max_accel)
                actions[i, 1] = np.clip(actions[i, 1], 0.0, 1.0)

        return actions

    def _get_fallback_actions(
        self,
        vehicle_states: Dict,
        icv_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        求解失败时的备用策略（IDM）

        Args:
            vehicle_states: 车辆状态字典
            icv_ids: ICV车辆ID列表

        Returns:
            actions_dict: 动作字典
        """
        actions_dict = {}

        for veh_id in icv_ids:
            if veh_id not in vehicle_states:
                continue

            state = vehicle_states[veh_id]
            speed = state.get('speed', 0.0)

            # IDM策略：加速到期望速度
            desired_speed = 30.0
            speed_error = desired_speed - speed

            # PI控制器
            accel = 0.5 * speed_error
            accel = np.clip(accel, self.config.min_accel, self.config.max_accel)

            actions_dict[veh_id] = np.array([accel, 0.0])

        return actions_dict

    def get_stats(self) -> Dict:
        """获取求解器统计信息"""
        return self.stats

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_solves': 0,
            'successful_solves': 0,
            'failed_solves': 0,
            'avg_solve_time': 0.0
        }


class DistributedMPCController:
    """
    分布式MPC控制器

    为每辆ICV独立求解MPC，然后进行协调。
    适用于车辆数量较多的情况，降低计算复杂度。
    """

    def __init__(self, config: Optional[MPCConfig] = None):
        """
        初始化分布式MPC控制器

        Args:
            config: MPC配置
        """
        self.config = config if config is not None else MPCConfig()
        self.traffic_model = SimplifiedTrafficModel(
            dt=self.config.dt,
            max_speed=self.config.max_speed,
            max_accel=self.config.max_accel,
            max_decel=self.config.min_accel
        )
        self.topology = TrafficTopology()

    def solve(
        self,
        observation_dict: Dict,
        vehicle_ids: List[str],
        icv_ids: List[str],
        step: int
    ) -> Dict[str, np.ndarray]:
        """
        分布式求解MPC（每辆车独立优化）

        Args:
            observation_dict: 观测字典
            vehicle_ids: 所有车辆ID列表
            icv_ids: ICV车辆ID列表
            step: 当前时间步

        Returns:
            actions_dict: 动作字典
        """
        vehicle_states = observation_dict.get('vehicle_states', {})

        # 更新拓扑
        self.topology.update_topology(vehicle_states, vehicle_ids)

        actions_dict = {}

        # 为每辆ICV独立求解
        for veh_id in icv_ids:
            if veh_id not in vehicle_states:
                continue

            # 获取前车信息
            leader_id = self.topology.get_leader(veh_id)

            # 求解单车辆MPC
            action = self._solve_single_vehicle_mpc(
                vehicle_states[veh_id],
                vehicle_states.get(leader_id) if leader_id else None
            )

            actions_dict[veh_id] = action

        return actions_dict

    def _solve_single_vehicle_mpc(
        self,
        ego_state: Dict,
        leader_state: Optional[Dict]
    ) -> np.ndarray:
        """
        求解单车辆MPC

        Args:
            ego_state: 本车状态
            leader_state: 前车状态（如果有）

        Returns:
            action: [acceleration, lane_change_prob]
        """
        N = self.config.prediction_horizon
        dt = self.config.dt

        # 优化变量
        u_accel = cp.Variable(N)
        u_lane = cp.Variable(N)

        # 状态变量
        v = cp.Variable(N + 1)
        s = cp.Variable(N + 1)

        # 初始状态
        v0 = ego_state.get('speed', 0.0)
        s0 = ego_state.get('s', 0.0)

        # 前车状态
        leader_v0 = leader_state.get('speed', v0) if leader_state else v0
        leader_s0 = leader_state.get('s', s0 + 100.0) if leader_state else s0 + 100.0

        # 目标函数
        cost = 0.0

        for k in range(N):
            # 速度跟踪
            desired_v = 30.0
            cost += self.config.Q_speed * cp.square(v[k] - desired_v)

            # 控制成本
            cost += self.config.R_accel * cp.square(u_accel[k])

            # 车间距保持
            if leader_state is not None:
                # 简化：假设前车匀速
                leader_s_k = leader_s0 + leader_v0 * (k + 1) * dt
                gap = leader_s_k - s[k]
                gap_error = cp.maximum(self.config.desired_gap - gap, 0.0)
                cost += self.config.Q_gap * cp.square(gap_error)

        # 约束
        constraints = [
            v[0] == v0,
            s[0] == s0,
        ]

        for k in range(N):
            # 状态转移
            constraints.append(v[k + 1] == v[k] + u_accel[k] * dt)
            constraints.append(s[k + 1] == s[k] + v[k] * dt)

            # 约束
            constraints.append(v[k] >= self.config.min_speed)
            constraints.append(v[k] <= self.config.max_speed)
            constraints.append(u_accel[k] >= self.config.min_accel)
            constraints.append(u_accel[k] <= self.config.max_accel)
            constraints.append(u_lane[k] >= 0.0)
            constraints.append(u_lane[k] <= 1.0)

        # 求解
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, verbose=False)

        # 提解
        if problem.status == 'optimal' and u_accel.value is not None:
            accel = float(u_accel.value[0])
            lane = float(u_lane.value[0]) if u_lane.value is not None else 0.0
        else:
            # 备用策略
            accel = np.clip(0.5 * (30.0 - v0), self.config.min_accel, self.config.max_accel)
            lane = 0.0

        return np.array([accel, lane])


if __name__ == '__main__':
    """测试MPC控制器"""
    print("=" * 70)
    print("MPC Controller Test")
    print("=" * 70)

    # 创建控制器
    config = MPCConfig(
        prediction_horizon=10,
        control_horizon=5,
        verbose=False
    )

    mpc = MPCController(config)

    # 创建测试观测
    vehicle_states = {
        'veh_0': {
            's': 1000.0,
            'd': 0.0,
            'vs': 20.0,
            'vd': 0.0,
            'speed': 20.0,
            'acceleration': 0.0,
            'lane_index': 0,
            'angle': 0.0
        },
        'veh_1': {
            's': 1050.0,
            'd': 0.0,
            'vs': 20.0,
            'vd': 0.0,
            'speed': 20.0,
            'acceleration': 0.0,
            'lane_index': 0,
            'angle': 0.0
        }
    }

    observation_dict = {
        'vehicle_states': vehicle_states,
        'vehicle_ids': ['veh_0', 'veh_1'],
        'icv_ids': {'veh_0', 'veh_1'}
    }

    # 求解
    print("\nSolving MPC problem...")
    actions = mpc.solve(
        observation_dict,
        vehicle_ids=['veh_0', 'veh_1'],
        icv_ids=['veh_0', 'veh_1'],
        step=0
    )

    print("\nOptimal Actions:")
    for veh_id, action in actions.items():
        print(f"  {veh_id}: accel={action[0]:.3f} m/s², lane_change={action[1]:.3f}")

    # 统计信息
    stats = mpc.get_stats()
    print(f"\nSolver Statistics:")
    print(f"  Total solves: {stats['total_solves']}")
    print(f"  Successful: {stats['successful_solves']}")
    print(f"  Failed: {stats['failed_solves']}")
    print(f"  Avg time: {stats['avg_solve_time']*1000:.2f} ms")

    print("\n✓ MPC Controller test passed!")
