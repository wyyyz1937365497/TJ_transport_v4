"""
比赛专用的SUMO环境改进版

根据赛题要求进行优化：
1. 使用车道自然坐标系（Frenet坐标）
2. 添加干预成本跟踪
3. 优化奖励函数以匹配比赛评价标准
4. 添加交通流预测所需的全局统计
"""

import numpy as np
import traci
from typing import Dict, List, Set, Any, Tuple
from collections import defaultdict

from .sumo_env import SumoEnvironment


class CompetitionSumoEnv(SumoEnvironment):
    """
    比赛专用SUMO环境

    改进点：
    1. 车辆状态使用Frenet坐标系（s, d）
    2. 添加干预成本跟踪
    3. 奖励函数匹配比赛评价标准
    4. 添加吞吐量和通行时间统计
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # 干预统计
        self.intervention_stats = {
            'controlled_vehicles': 0,  # 受控车辆数量
            'total_accel_changes': 0,  # 加速度变化次数
            'total_lane_changes': 0,   # 换道次数
            'energy_consumption': 0.0,  # 能耗
            'control_magnitude': 0.0    # 控制幅度累计
        }

        # 车辆旅行时间跟踪
        self.vehicle_travel_times = {}  # {vehicle_id: {'start': step, 'end': step}}
        self.vehicle_departure_times = {}  # {vehicle_id: departure_step}

        # 性能指标
        self.performance_metrics = {
            'avg_speed': 0.0,
            'speed_std': 0.0,
            'throughput': 0.0,      # 吞吐量（到达车辆数/时间）
            'avg_travel_time': 0.0,  # 平均旅行时间
            'stopped_ratio': 0.0,    # 停车车辆比例
            'congestion_level': 0.0  # 拥堵水平
        }

    def reset(self) -> Dict[str, Any]:
        """重置环境"""
        obs = super().reset()

        # 重置统计
        self.intervention_stats = {
            'controlled_vehicles': 0,
            'total_accel_changes': 0,
            'total_lane_changes': 0,
            'energy_consumption': 0.0,
            'control_magnitude': 0.0
        }
        self.vehicle_travel_times = {}
        self.vehicle_departure_times = {}

        return obs

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行一步仿真

        Args:
            actions: {vehicle_id: [acceleration, lane_change]}

        Returns:
            observation, reward, done, info
        """
        # 记录干预统计
        self._track_interventions(actions)

        # 执行动作
        observation = super().step(actions)

        # 更新性能指标
        self._update_performance_metrics(observation)

        # 计算奖励（使用比赛标准）
        reward = self._compute_competition_reward(observation)

        # 检查结束
        done = self._is_done()

        # 获取额外信息
        info = self._get_competition_info()

        return observation, reward, done, info

    def _get_observation(self) -> Dict[str, Any]:
        """
        获取观测（使用Frenet坐标系）

        返回的车辆状态包含：
        - s: 沿车道中心线的距离
        - d: 横向偏移（相对于车道中心）
        - vs: 纵向速度
        - vd: 横向速度（近似为0，因为车辆主要沿车道行驶）
        - lane_index: 车道索引
        """
        all_vehicle_ids = traci.vehicle.getIDList()
        vehicle_states = {}
        valid_vehicle_ids = []
        icv_ids = set()

        # 选择ICV
        control_ratio = self.config.get('control_ratio', 0.25)
        num_icv = max(1, int(len(all_vehicle_ids) * control_ratio))

        if len(all_vehicle_ids) > 0:
            icv_indices = np.random.choice(
                len(all_vehicle_ids),
                size=min(num_icv, len(all_vehicle_ids)),
                replace=False
            )
            icv_ids = {all_vehicle_ids[i] for i in icv_indices}

        # 收集车辆状态（Frenet坐标系）
        for veh_id in all_vehicle_ids:
            try:
                # 获取基本状态
                speed = traci.vehicle.getSpeed(veh_id)
                acceleration = traci.vehicle.getAcceleration(veh_id)
                angle = traci.vehicle.getAngle(veh_id)
                lane_id = traci.vehicle.getLaneID(veh_id)
                lane_index = traci.vehicle.getLaneIndex(veh_id)

                # Frenet坐标系
                s = traci.vehicle.getLanePosition(veh_id)  # 沿车道位置
                d = traci.vehicle.getLateralLanePosition(veh_id)  # 横向偏移

                # 速度分解到Frenet坐标系
                # 纵向速度 vs ≈ speed (主要沿车道方向)
                # 横向速度 vd ≈ 0 (直道情况)
                vs = speed * np.cos(np.radians(angle - self._get_lane_angle(lane_id)))
                vd = speed * np.sin(np.radians(angle - self._get_lane_angle(lane_id)))

                # 记录出发时间
                if veh_id not in self.vehicle_departure_times:
                    self.vehicle_departure_times[veh_id] = self.current_step

                vehicle_states[veh_id] = {
                    'id': veh_id,
                    's': s,              # 纵向位置（沿车道）
                    'd': d,              # 横向偏移
                    'vs': vs,            # 纵向速度
                    'vd': vd,            # 横向速度
                    'speed': speed,      # 总速度
                    'acceleration': acceleration,
                    'lane_id': lane_id,
                    'lane_index': lane_index,
                    'angle': angle,
                    # 保留部分笛卡尔坐标用于全局理解
                    'x': traci.vehicle.getPosition(veh_id)[0],
                    'y': traci.vehicle.getPosition(veh_id)[1]
                }

                valid_vehicle_ids.append(veh_id)

            except Exception as e:
                continue

        # 全局统计
        global_stats = self._compute_competition_global_stats(vehicle_states)

        observation = {
            'vehicle_states': vehicle_states,
            'vehicle_ids': valid_vehicle_ids,
            'icv_ids': icv_ids,
            'global_stats': global_stats,
            'step': self.current_step
        }

        return observation

    def _get_lane_angle(self, lane_id: str) -> float:
        """获取车道的航向角"""
        try:
            edge_id = lane_id.split('_')[0]
            angle = traci.edge.getAngle(edge_id)
            return angle
        except:
            return 0.0

    def _compute_competition_global_stats(self, vehicle_states: Dict[str, Dict]) -> np.ndarray:
        """
        计算比赛需要的全局统计特征

        返回32维向量：
        [0-15]: 原有统计
        [16-23]: 效率相关（吞吐量、平均旅行时间等）
        [24-31]: 稳定性相关（拥堵指标等）
        """
        if not vehicle_states:
            return np.zeros(32)

        stats = np.zeros(32)

        # [0-15]: 原有统计（继承自父类）
        base_stats = super()._compute_global_stats(vehicle_states)
        stats[:16] = base_stats

        speeds = [v['speed'] for v in vehicle_states.values()]
        accelerations = [v['acceleration'] for v in vehicle_states.values()]

        # [16-19]: 效率指标
        stats[16] = len(speeds) / max(self.current_step, 1)  # 吞吐量（车辆数/时间）
        stats[17] = np.mean(speeds) if speeds else 0.0        # 平均速度
        stats[18] = sum(1 for s in speeds if s > 5.0) / max(len(speeds), 1)  # 高速车辆比例
        stats[19] = np.percentile(speeds, 50) if speeds else 0.0  # 速度中位数

        # [20-23]: 拥堵指标
        stats[20] = sum(1 for s in speeds if s < 1.0) / max(len(speeds), 1)  # 慢速车辆比例（拥堵）
        stats[21] = np.std(speeds) if len(speeds) > 1 else 0.0  # 速度标准差
        stats[22] = sum(1 for a in accelerations if a < -2.0) / max(len(accelerations), 1)  # 急减速比例
        stats[23] = sum(1 for s in speeds if s < 0.1) / max(len(speeds), 1)  # 停车比例

        # [24-27]: 车道利用率
        lane_counts = defaultdict(int)
        for v in vehicle_states.values():
            lane_counts[v['lane_index']] += 1

        if lane_counts:
            stats[24] = max(lane_counts.values()) / len(vehicle_states)  # 最拥挤车道占有率
            stats[25] = len(lane_counts)  # 使用车道数
            stats[26] = np.std(list(lane_counts.values()))  # 车道分布标准差
        else:
            stats[24:27] = 0.0

        stats[27] = sum(1 for v in vehicle_states.values() if abs(v['d']) > 1.0) / max(len(vehicle_states), 1)  # 偏离车道中心的比例

        # [28-31]: 预留空间（可添加更多指标）
        stats[28:32] = 0.0

        return stats

    def _track_interventions(self, actions: Dict[str, np.ndarray]):
        """跟踪干预统计"""
        if not actions:
            return

        for veh_id, action in actions.items():
            if veh_id not in traci.vehicle.getIDList():
                continue

            try:
                old_accel = traci.vehicle.getAcceleration(veh_id)
                old_lane = traci.vehicle.getLaneIndex(veh_id)

                # 加速度变化
                accel_change = abs(action[0] - old_accel)
                self.intervention_stats['total_accel_changes'] += accel_change

                # 换道
                lane_change = action[1]  # -1: 左, 0: 不变, 1: 右
                if lane_change != 0:
                    self.intervention_stats['total_lane_changes'] += 1

                # 控制幅度
                self.intervention_stats['control_magnitude'] += abs(action[0]) + abs(lane_change)

                # 能耗估计（简化版）
                # 能耗 ≈ 速度 × 加速度（正加速度消耗能量）
                speed = traci.vehicle.getSpeed(veh_id)
                if action[0] > 0:
                    self.intervention_stats['energy_consumption'] += speed * action[0] * 0.1  # dt=0.1s

            except:
                continue

        self.intervention_stats['controlled_vehicles'] = len(actions)

    def _update_performance_metrics(self, observation: Dict[str, Any]):
        """更新性能指标"""
        vehicle_states = observation.get('vehicle_states', {})

        if not vehicle_states:
            return

        speeds = [v['speed'] for v in vehicle_states.values()]

        # 更新速度统计
        self.performance_metrics['avg_speed'] = np.mean(speeds) if speeds else 0.0
        self.performance_metrics['speed_std'] = np.std(speeds) if len(speeds) > 1 else 0.0

        # 更新停车比例
        stopped = sum(1 for s in speeds if s < 0.1)
        self.performance_metrics['stopped_ratio'] = stopped / len(speeds) if speeds else 0.0

        # 更新吞吐量
        current_time = self.current_step * self.config.get('step_length', 0.1)
        self.performance_metrics['throughput'] = len(self.stats.get('arrived_vehicles', [])) / max(current_time, 1.0)

        # 更新拥堵水平（速度<1m/s的车辆比例）
        self.performance_metrics['congestion_level'] = sum(1 for s in speeds if s < 1.0) / len(speeds) if speeds else 0.0

        # 更新旅行时间
        for veh_id in self.stats.get('arrived_vehicles', []):
            if veh_id in self.vehicle_departure_times:
                departure_step = self.vehicle_departure_times.pop(veh_id, self.current_step)
                travel_time = self.current_step - departure_step
                self.vehicle_travel_times[veh_id] = travel_time

        if self.vehicle_travel_times:
            self.performance_metrics['avg_travel_time'] = np.mean(list(self.vehicle_travel_times.values()))

    def _compute_competition_reward(self, observation: Dict[str, Any]) -> float:
        """
        计算比赛标准奖励

        奖励 = 效率得分 + 稳定性得分 - 干预成本惩罚

        初赛权重：效率为主
        """
        vehicle_states = observation.get('vehicle_states', {})

        if not vehicle_states:
            return 0.0

        speeds = [v['speed'] for v in vehicle_states.values()]

        # ========== 1. 效率得分 Sefficiency ==========
        # 1.1 速度得分（鼓励高速）
        avg_speed = np.mean(speeds) if speeds else 0.0
        speed_score = avg_speed / 30.0  # 归一化到0-30m/s

        # 1.2 吞吐量得分（鼓励高到达率）
        arrived_count = len(self.stats.get('arrived_vehicles', []))
        current_time = self.current_step * self.config.get('step_length', 0.1)
        throughput_score = arrived_count / max(current_time, 1.0) * 10.0  # 放大权重

        # 1.3 完成率得分
        departed_count = len(self.stats.get('departed_vehicles', []))
        completion_score = (arrived_count / max(departed_count, 1)) if departed_count > 0 else 0.0

        # 综合效率得分
        efficiency_score = (
            5.0 * speed_score +      # 速度权重5
            3.0 * throughput_score +  # 吞吐量权重3
            2.0 * completion_score    # 完成率权重2
        )

        # ========== 2. 稳定性得分 Sstability ==========
        # 2.1 速度方差惩罚
        if len(speeds) > 1:
            speed_std = np.std(speeds)
            stability_score = -0.5 * (speed_std / 10.0)  # 惩罚速度波动
        else:
            stability_score = 0.0

        # 2.2 拥堵惩罚
        congestion_ratio = sum(1 for s in speeds if s < 1.0) / len(speeds) if speeds else 0.0
        congestion_penalty = -2.0 * congestion_ratio

        # 2.3 停车惩罚
        stopped_ratio = sum(1 for s in speeds if s < 0.1) / len(speeds) if speeds else 0.0
        stopped_penalty = -1.0 * stopped_ratio

        # 综合稳定性得分
        stability_total = stability_score + congestion_penalty + stopped_penalty

        # ========== 3. 干预成本惩罚 Pint ==========
        # 3.1 受控车辆数量惩罚（初赛中ICV比例固定为25%，这里只惩罚控制幅度）
        control_magnitude = self.intervention_stats.get('control_magnitude', 0.0)
        magnitude_penalty = 0.01 * control_magnitude / max(len(speeds), 1)

        # 3.2 换道惩罚（换道是高风险操作）
        lane_changes = self.intervention_stats.get('total_lane_changes', 0)
        lane_change_penalty = 0.1 * lane_changes / max(len(speeds), 1)

        # 干预成本惩罚因子（范围：0-1，越小惩罚越大）
        intervention_cost = magnitude_penalty + lane_change_penalty
        Pint = max(0.1, 1.0 - intervention_cost)  # 最低为0.1

        # ========== 总奖励 ==========
        # 初赛：效率为主，稳定性为辅，干预成本次之
        total_reward = (efficiency_score + stability_total) * Pint

        return float(total_reward)

    def _get_competition_info(self) -> Dict[str, Any]:
        """获取比赛所需的信息"""
        base_info = self._get_info()

        competition_info = {
            **base_info,
            'intervention_stats': self.intervention_stats.copy(),
            'performance_metrics': self.performance_metrics.copy(),
            'efficiency_score': self.performance_metrics.get('avg_speed', 0.0),
            'stability_score': -self.performance_metrics.get('speed_std', 0.0),
            'intervention_penalty': self.intervention_stats.get('control_magnitude', 0.0)
        }

        return competition_info
