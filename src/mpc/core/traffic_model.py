"""
简化的交通流模型用于MPC预测

该模型提供车辆在给定控制输入下的状态转移预测，
用于MPC的有限时域优化。

模型基于简化的物理方程：
- 纵向运动：受加速度和阻力影响
- 横向运动：简化为换道决策
- 跟驰模型：IDM（Intelligent Driver Model）
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class SimplifiedTrafficModel:
    """
    简化的交通流模型

    用于MPC预测车辆在给定控制输入下的未来状态。
    考虑了：
    1. 车辆动力学（加速/减速）
    2. 跟驰行为（IDM）
    3. 换道影响
    """

    def __init__(
        self,
        dt: float = 0.1,  # 时间步长（秒）
        max_speed: float = 30.0,  # 最大速度（m/s）
        max_accel: float = 2.0,  # 最大加速度（m/s²）
        max_decel: float = -4.5,  # 最大减速度（m/s²）
    ):
        """
        初始化交通流模型

        Args:
            dt: 时间步长
            max_speed: 最大速度
            max_accel: 最大加速度
            max_decel: 最大减速度
        """
        self.dt = dt
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_decel = max_decel

        # IDM参数（用于跟驰模型）
        self.idm_desired_speed = 30.0  # v0: 期望速度（m/s）
        self.idm_min_gap = 2.0  # s0: 最小车间距（m）
        self.idm_time_headway = 1.5  # T: 车头时距（s）
        self.idm_accel = 0.73  # a_max: 最大加速度（m/s²）
        self.idm_decel = 1.67  # b_max: 舒适减速度（m/s²）

    def predict_single_vehicle(
        self,
        current_state: np.ndarray,
        control_input: np.ndarray,
        leader_state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        预测单个车辆在给定控制输入下的下一状态

        Args:
            current_state: [s, v, a, lane] 当前状态
                - s: 纵向位置（m）
                - v: 纵向速度（m/s）
                - a: 纵向加速度（m/s²）
                - lane: 车道索引
            control_input: [accel, lane_change_prob]
                - accel: 加速度指令（m/s²）
                - lane_change_prob: 换道概率
            leader_state: [s_leader, v_leader] 前车状态（如果有）

        Returns:
            next_state: [s, v, a, lane] 下一状态
        """
        s, v, a, lane = current_state
        accel_cmd, lane_change_prob = control_input

        # 1. 更新速度（考虑车辆动力学和IDM跟驰）
        if leader_state is not None:
            # 有前车：使用IDM模型
            s_leader, v_leader = leader_state

            # 计算车间距
            gap = max(s_leader - s - 5.0, 0.1)  # 假设车长5m

            # 计算速度差
            delta_v = v - v_leader

            # IDM期望加速度
            desired_gap = (self.idm_min_gap +
                           v * self.idm_time_headway +
                           (v * delta_v) / (2 * np.sqrt(self.idm_accel * self.idm_decel)))

            idm_accel = self.idm_accel * (1 - (v / self.idm_desired_speed)**4 -
                                         (desired_gap / gap)**2)

            # 组合用户指令和IDM（加权平均）
            alpha = 0.7  # 用户指令权重
            next_accel = alpha * accel_cmd + (1 - alpha) * idm_accel
        else:
            # 无前车：直接使用控制输入
            next_accel = accel_cmd

        # 限制加速度范围
        next_accel = np.clip(next_accel, self.max_decel, self.max_accel)

        # 更新速度
        next_v = v + next_accel * self.dt
        next_v = np.clip(next_v, 0.0, self.max_speed)

        # 更新位置
        next_s = s + next_v * self.dt

        # 更新加速度（用于输出）
        next_a = next_accel

        # 2. 处理换道（简化：probabilistic）
        # 在MPC中我们主要优化纵向控制，换道作为辅助
        next_lane = lane  # 简化：不频繁换道

        next_state = np.array([next_s, next_v, next_a, next_lane])

        return next_state

    def predict_multi_vehicle(
        self,
        current_states: Dict[str, np.ndarray],
        control_inputs: Dict[str, np.ndarray],
        vehicle_ids: List[str],
        topology: Optional[Dict] = None
    ) -> Dict[str, np.ndarray]:
        """
        预测多车辆系统的下一状态

        Args:
            current_states: {veh_id: state} 当前状态
            control_inputs: {veh_id: [accel, lane_change]} 控制输入
            vehicle_ids: 车辆ID列表（按车道位置排序）
            topology: 拓扑信息（用于确定前车关系）

        Returns:
            next_states: {veh_id: state} 下一状态
        """
        next_states = {}

        # 如果没有提供拓扑信息，简化处理（每辆车独立）
        if topology is None:
            for veh_id in vehicle_ids:
                state = current_states.get(veh_id)
                control = control_inputs.get(veh_id, np.array([0.0, 0.0]))

                if state is not None:
                    next_state = self.predict_single_vehicle(state, control, None)
                    next_states[veh_id] = next_state
        else:
            # 有拓扑信息：考虑前车-跟随关系
            for veh_id in vehicle_ids:
                state = current_states.get(veh_id)
                control = control_inputs.get(veh_id, np.array([0.0, 0.0]))

                if state is not None:
                    # 获取前车
                    leader_id = topology.get(veh_id, {}).get('leader')
                    leader_state = None

                    if leader_id and leader_id in current_states:
                        leader_state_data = current_states[leader_id]
                        leader_state = np.array([
                            leader_state_data[0],  # s
                            leader_state_data[1]   # v
                        ])

                    next_state = self.predict_single_vehicle(state, control, leader_state)
                    next_states[veh_id] = next_state

        return next_states


class IDMController:
    """
    IDM（Intelligent Driver Model）控制器

    作为MPC的baseline和组件，用于生成自然的纵向控制。
    """

    def __init__(
        self,
        desired_speed: float = 30.0,
        min_gap: float = 2.0,
        time_headway: float = 1.5,
        max_accel: float = 0.73,
        max_decel: float = 1.67
    ):
        """
        初始化IDM控制器

        Args:
            desired_speed: 期望速度（m/s）
            min_gap: 最小车间距（m）
            time_headway: 车头时距（s）
            max_accel: 最大加速度（m/s²）
            max_decel: 最大减速度（m/s²）
        """
        self.desired_speed = desired_speed
        self.min_gap = min_gap
        self.time_headway = time_headway
        self.max_accel = max_accel
        self.max_decel = max_decel

    def compute_acceleration(
        self,
        current_speed: float,
        gap: float,
        leader_speed: float
    ) -> float:
        """
        计算IDM加速度

        Args:
            current_speed: 当前速度（m/s）
            gap: 与前车距离（m）
            leader_speed: 前车速度（m/s）

        Returns:
            acceleration: 加速度（m/s²）
        """
        # 确保gap为正
        gap = max(gap, 0.1)

        # 速度差
        delta_v = current_speed - leader_speed

        # 期望车间距
        desired_gap = (self.min_gap +
                       current_speed * self.time_headway +
                       (current_speed * delta_v) / (2 * np.sqrt(self.max_accel * self.max_decel)))

        # IDM加速度公式
        accel = self.max_accel * (1 - (current_speed / self.desired_speed)**4 -
                                   (desired_gap / gap)**2)

        return accel


class TrafficTopology:
    """
    交通拓扑管理器

    用于管理车辆之间的空间关系（前车-跟随、车道相邻等）。
    """

    def __init__(self):
        """初始化拓扑管理器"""
        self.vehicle_relations = {}

    def update_topology(
        self,
        vehicle_states: Dict[str, np.ndarray],
        vehicle_ids: List[str]
    ):
        """
        更新车辆拓扑关系

        Args:
            vehicle_states: {veh_id: [s, v, a, lane]} 车辆状态
            vehicle_ids: 车辆ID列表
        """
        self.vehicle_relations = {}

        # 按车道分组
        lanes = {}
        for veh_id in vehicle_ids:
            if veh_id in vehicle_states:
                state = vehicle_states[veh_id]
                # ✅ 修复：state是字典，不是numpy数组
                # 从字典中提取lane_index和s坐标
                if isinstance(state, dict):
                    lane = int(state.get('lane_index', 0))
                    s = state.get('s', 0.0)
                else:
                    # 兼容numpy数组格式 [s, v, a, lane]
                    lane = int(state[3])
                    s = state[0]

                if lane not in lanes:
                    lanes[lane] = []
                lanes[lane].append((veh_id, s))  # (veh_id, s)

        # 对每条车道，按s坐标排序，建立前车-跟随关系
        for lane, vehicles in lanes.items():
            # 按s坐标降序排序
            vehicles_sorted = sorted(vehicles, key=lambda x: x[1], reverse=True)

            for i, (veh_id, s) in enumerate(vehicles_sorted):
                if veh_id not in self.vehicle_relations:
                    self.vehicle_relations[veh_id] = {}

                # 前车（同一车道，s坐标更大的车）
                if i > 0:
                    leader_id, _ = vehicles_sorted[i - 1]
                    self.vehicle_relations[veh_id]['leader'] = leader_id
                    self.vehicle_relations[veh_id]['leader_gap'] = vehicles_sorted[i - 1][1] - s
                else:
                    self.vehicle_relations[veh_id]['leader'] = None
                    self.vehicle_relations[veh_id]['leader_gap'] = 1000.0  # 无前车

    def get_leader(self, veh_id: str) -> Optional[str]:
        """获取前车ID"""
        if veh_id in self.vehicle_relations:
            return self.vehicle_relations[veh_id].get('leader')
        return None

    def get_leader_gap(self, veh_id: str) -> float:
        """获取前车距离"""
        if veh_id in self.vehicle_relations:
            return self.vehicle_relations[veh_id].get('leader_gap', 1000.0)
        return 1000.0


if __name__ == '__main__':
    """测试交通流模型"""
    # 测试单车辆预测
    model = SimplifiedTrafficModel()

    # 初始状态：s=100m, v=20m/s, a=0, lane=0
    state = np.array([100.0, 20.0, 0.0, 0.0])

    # 控制输入：加速1 m/s²，不换道
    control = np.array([1.0, 0.0])

    # 预测
    next_state = model.predict_single_vehicle(state, control, None)

    print(f"初始状态: s={state[0]:.1f}m, v={state[1]:.1f}m/s")
    print(f"控制输入: accel={control[0]:.1f} m/s²")
    print(f"下一状态: s={next_state[0]:.1f}m, v={next_state[1]:.1f}m/s, a={next_state[2]:.2f} m/s²")

    print("\n✓ 交通流模型测试通过")
