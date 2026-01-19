"""
比赛专用的SUMO环境改进版

根据赛题要求进行优化：
1. 使用车道自然坐标系（Frenet坐标）
2. 添加干预成本跟踪
3. 优化奖励函数以匹配比赛评价标准
4. 添加交通流预测所需的全局统计
"""

import numpy as np
import torch
from typing import Dict, List, Set, Any, Tuple, Optional
from collections import defaultdict
from pathlib import Path

from .gpu_sumo_env_optimized import GPUSumoEnvironmentOptimized as GPUSumoEnvironment

# 导入优化的Frenet工具
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.frenet_utils import get_frenet_system


class CompetitionSumoEnv(GPUSumoEnvironment):
    """
    比赛专用SUMO环境（GPU加速版）

    改进点：
    1. 车辆状态使用Frenet坐标系（s, d）
    2. 添加干预成本跟踪
    3. 奖励函数匹配比赛评价标准
    4. 添加吞吐量和通行时间统计
    5. 使用Libsumo + GPU加速
    """

    def __init__(
        self,
        config: Dict[str, Any],
        use_gui: bool = False,
        device: str = 'cuda'
    ):
        super().__init__(config, use_gui=use_gui, device=device)

        # ✅ 修复：从正确的配置路径读取奖励权重
        # 配置文件中使用的是 'rewards' 而不是 'competition.reward_weights'
        reward_config = config.get('rewards', {})
        competition_config = config.get('competition', {})
        reward_weights = competition_config.get('reward_weights', {})

        # 优先使用 reward_weights，否则使用 reward_config，最后使用默认值
        self.speed_norm_factor = reward_weights.get('speed_norm',
                                                     reward_config.get('speed_weight', 1.0) / 30.0)
        self.throughput_weight = reward_weights.get('throughput',
                                                    reward_config.get('efficiency_weight', 2.0) * 5.0)
        self.efficiency_weights = reward_weights.get('efficiency_weights',
                                                      competition_config.get('efficiency_weights', [5.0, 3.0, 2.0]))
        self.stability_weight = reward_weights.get('stability_weight',
                                                    reward_config.get('safety_weight', 3.0) * 0.2)
        self.congestion_penalty_weight = reward_weights.get('congestion_penalty_weight', 2.0)
        self.stopped_penalty_weight = reward_weights.get('stopped_penalty_weight', 1.0)

        # 初始化优化的Frenet坐标系(基于固定路网)
        net_xml_path = config.get('net_file', '仿真环境_初赛_1.0/仿真环境-初赛/net.xml')
        if Path(net_xml_path).exists():
            try:
                self.frenet_system = get_frenet_system(net_xml_path)
                self.use_accurate_frenet = True
                print(f"[OK] Frenet坐标系初始化成功: {net_xml_path}")
            except Exception as e:
                # Frenet系统是比赛性能的关键，不应该静默失败
                raise RuntimeError(
                    f"[X] Frenet坐标系初始化失败（这会严重降低模型性能）: {e}\n"
                    f"   请确保net.xml文件路径正确且格式有效: {net_xml_path}\n"
                    f"   Frenet坐标系对于准确的位置表示至关重要，不能使用简化版本。"
                )
        else:
            raise FileNotFoundError(
                f"[X] Frenet系统所需的net.xml文件不存在: {net_xml_path}\n"
                f"   精确的Frenet坐标系对于比赛性能至关重要。\n"
                f"   请检查路径配置或在config中指定正确的net_file路径。"
            )

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

        # ✅ 新增：智能ICV管理器（Top-K机制）
        # 读取智能ICV配置
        smart_icv_config = config.get('smart_icv', {})
        use_smart_icv = smart_icv_config.get('enabled', False)

        if use_smart_icv:
            from src.env.smart_icv_manager import SmartICVManager

            max_vehicles = config.get('max_vehicles', 512)

            self.smart_icv_manager = SmartICVManager(
                max_vehicles=max_vehicles,
                default_top_k=smart_icv_config.get('default_top_k', 5),
                emergency_top_k=smart_icv_config.get('emergency_top_k', 15),
                elevated_top_k=smart_icv_config.get('elevated_top_k', 10),
                intervention_threshold=smart_icv_config.get('intervention_threshold', 0.25),
                decision_interval=smart_icv_config.get('decision_interval', 10),
                ttc_threshold=smart_icv_config.get('ttc_threshold', 2.0),
                thw_threshold=smart_icv_config.get('thw_threshold', 1.5),
            )

            print(f"[OK] 智能ICV管理器已启用（Top-K机制）")
            print(f"     默认控制: {self.smart_icv_manager.default_top_k} 辆")
            print(f"     紧急控制: {self.smart_icv_manager.emergency_top_k} 辆")
        else:
            self.smart_icv_manager = None

        # 动态ICV管理状态（兼容旧版）
        self.current_icv_ids = set()  # 当前ICV车辆集合
        self.icv_scores = {}  # 当前ICV的重要性评分 {veh_id: score}
        self.icv_selection_step = 0  # 上次ICV选择的step
        self.icv_reevaluate_interval = 10  # 每10步重新评估一次ICV组成

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

        # ✅ 重置动态ICV状态
        self.current_icv_ids = set()
        self.icv_scores = {}
        self.icv_selection_step = 0

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

        # 执行动作（父类返回tuple: observation, reward, done, info）
        obs, _, done, base_info = super().step(actions)

        # 更新性能指标
        self._update_performance_metrics(obs)

        # 计算奖励（使用比赛标准）
        reward = self._compute_competition_reward(obs)

        # 获取额外信息
        info = self._get_competition_info()

        return obs, reward, done, info

    def _get_observation(self) -> Dict[str, Any]:
        """
        获取观测（使用优化的Frenet坐标系 + GPU加速）

        返回的车辆状态包含：
        - s: 沿车道中心线的距离
        - d: 横向偏移（相对于车道中心）
        - vs: 纵向速度
        - vd: 横向速度
        - lane_index: 车道索引
        - in_bottleneck: 是否在瓶颈区域
        - edge_id: 边ID
        """
        # 导入traci（可能是libsumo）
        try:
            import libsumo as traci_lib
        except ImportError:
            import traci as traci_lib

        all_vehicle_ids = traci_lib.vehicle.getIDList()
        vehicle_states = {}
        valid_vehicle_ids = []

        # ✅ 智能ICV选择机制（基于规则的影响力评分 + 动态释放）
        # ✅ 修复：使用icv_ratio而不是control_ratio（配置文件中使用icv_ratio）
        control_ratio = self.config.get('icv_ratio', self.config.get('control_ratio', 0.25))
        num_icv = max(1, int(len(all_vehicle_ids) * control_ratio))

        icv_ids = set()
        if len(all_vehicle_ids) > 0:
            # ✅ 动态ICV管理：定期重新评估ICV组成
            should_reevaluate = (
                self.current_step == 0 or  # episode开始
                (self.current_step - self.icv_selection_step) >= self.icv_reevaluate_interval  # 超过间隔
            )

            if should_reevaluate and len(self.current_icv_ids) > 0:
                # 动态更新：重新评估并可能释放部分ICV
                icv_ids = self._dynamic_update_icv(
                    all_vehicle_ids,
                    traci_lib,
                    num_icv
                )
            else:
                # 初始选择或保持不变
                if len(self.current_icv_ids) == 0:
                    # 初始选择
                    icv_ids = self._intelligent_select_icv(
                        all_vehicle_ids,
                        traci_lib,
                        num_icv
                    )
                    self.current_icv_ids = icv_ids
                else:
                    # 保持当前ICV集合（但需要检查车辆是否还在路网中）
                    icv_ids = self.current_icv_ids & set(all_vehicle_ids)

                    # 如果ICV数量不足，补充新的
                    if len(icv_ids) < num_icv:
                        additional_needed = min(num_icv - len(icv_ids), len(set(all_vehicle_ids) - icv_ids))
                        if additional_needed > 0:
                            remaining_vehicles = set(all_vehicle_ids) - icv_ids
                            # 从剩余车辆中选择重要性最高的
                            additional_icv = self._intelligent_select_icv(
                                list(remaining_vehicles),
                                traci_lib,
                                additional_needed
                            )
                            icv_ids.update(additional_icv)
                            self.current_icv_ids = icv_ids

                    # ✅ 关键修复：确保ICV数量不超过目标值（硬约束）
                    if len(icv_ids) > num_icv:
                        # 保留评分最高的num_icv个ICV
                        icv_scores = {veh_id: self._compute_vehicle_score(veh_id, traci_lib, all_vehicle_ids)
                                      for veh_id in icv_ids if veh_id in all_vehicle_ids}
                        sorted_icvs = sorted(icv_scores.items(), key=lambda x: x[1], reverse=True)
                        icv_ids = set([veh_id for veh_id, _ in sorted_icvs[:num_icv]])
                        self.current_icv_ids = icv_ids

        # ========== 优化: 使用Libsumo批量获取车辆状态 ==========
        # 批量订阅所有车辆的关键属性（减少IPC调用）
        if len(all_vehicle_ids) > 0:
            # 定义要订阅的变量（使用traci.constants中的常量ID）
            var_list = [
                traci_lib.constants.VAR_SPEED,        # 0x40 - 速度
                traci_lib.constants.VAR_ACCELERATION,  # 0x72 - 加速度
                traci_lib.constants.VAR_ANGLE,        # 0x43 - 角度
                traci_lib.constants.VAR_LANE_INDEX,   # 0x52 - 车道索引
                traci_lib.constants.VAR_POSITION,     # 0x42 - 位置
                traci_lib.constants.VAR_LANE_ID       # 0x51 - 车道ID
            ]

            # 批量订阅（Libsumo直接调用，无TCP开销）
            for veh_id in all_vehicle_ids:
                traci_lib.vehicle.subscribe(veh_id, var_list)

            # 一次性获取所有车辆的订阅数据
            all_subscription_results = traci_lib.vehicle.getAllSubscriptionResults()
        else:
            all_subscription_results = {}

        # 收集车辆状态（优化的Frenet坐标系）
        for veh_id in all_vehicle_ids:
            try:
                # 从订阅结果中获取数据（避免单独的TraCI调用）
                if veh_id in all_subscription_results:
                    sub_data = all_subscription_results[veh_id]
                    speed = sub_data.get(traci_lib.constants.VAR_SPEED, 0.0)                     # VAR_SPEED (0x40)
                    acceleration = sub_data.get(traci_lib.constants.VAR_ACCELERATION, 0.0)       # VAR_ACCELERATION (0x72)
                    angle = sub_data.get(traci_lib.constants.VAR_ANGLE, 0.0)                     # VAR_ANGLE (0x43)
                    lane_index = sub_data.get(traci_lib.constants.VAR_LANE_INDEX, 0)             # VAR_LANE_INDEX (0x52)
                    x, y = sub_data.get(traci_lib.constants.VAR_POSITION, (0.0, 0.0))           # VAR_POSITION (0x42)
                    lane_id = sub_data.get(traci_lib.constants.VAR_LANE_ID, "")                  # VAR_LANE_ID (0x51)
                else:
                    # Libsumo订阅失败不应该发生
                    raise RuntimeError(
                        f"[X] Libsumo批量订阅失败，车辆 {veh_id} 不在订阅结果中。\n"
                        f"   这会导致性能严重下降。\n"
                        f"   当前订阅车辆数: {len(all_vehicle_ids)}\n"
                        f"   订阅结果车辆数: {len(all_subscription_results)}\n"
                        f"   请检查Libsumo配置和SUMO版本兼容性。"
                    )

                # 提取edge_id
                edge_id = lane_id.split('_')[0] if '_' in lane_id else lane_id

                # 优化的Frenet坐标系计算
                # 使用精确的Frenet坐标系统(基于预计算的车道中心线)
                s, d = self.frenet_system.cartesian_to_frenet(x, y, edge_id, lane_id)

                # 获取车道在该位置的航向角(用于速度分解)
                lane_heading = self.frenet_system.lanes.get(lane_id)
                if lane_heading is not None:
                    heading_at_s = lane_heading.get_heading_at_s(s)
                else:
                    heading_at_s = np.radians(angle)

                    # 检查是否在瓶颈区域
                in_bottleneck = self.frenet_system.is_in_bottleneck(s, edge_id)

                # 速度分解到Frenet坐标系
                vs = speed * np.cos(angle * np.pi / 180.0 - heading_at_s)
                vd = speed * np.sin(angle * np.pi / 180.0 - heading_at_s)

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
                    'edge_id': edge_id,
                    'in_bottleneck': in_bottleneck,  # 是否在瓶颈区域
                    # 保留部分笛卡尔坐标用于调试
                    'x': x,
                    'y': y
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
            # 导入traci（可能是libsumo）
            try:
                import libsumo as traci_lib
            except ImportError:
                import traci as traci_lib

            edge_id = lane_id.split('_')[0]
            angle = traci_lib.edge.getAngle(edge_id)
            return angle
        except:
            return 0.0

    def _compute_vehicle_score(
        self,
        veh_id: str,
        traci_lib,
        all_vehicle_ids: List[str]
    ) -> float:
        """
        计算单辆车的重要性评分

        评分标准：
        1. 位置权重：瓶颈区域车辆优先（最多25分）
        2. 速度权重：速度异常车辆（最多10分）
        3. 车道权重：关键车道车辆（最多5分）
        4. 加速度权重：急加减速车辆（5分）
        5. 跟驰距离权重：跟驰风险（最多8分）

        Args:
            veh_id: 车辆ID
            traci_lib: TraCI库实例
            all_vehicle_ids: 所有车辆ID列表（用于前车查询）

        Returns:
            重要性评分（0-53分）
        """
        score = 0.0

        try:
            # 获取车辆基本信息
            lane_id = traci_lib.vehicle.getLaneID(veh_id)
            lane_index = traci_lib.vehicle.getLaneIndex(veh_id)
            speed = traci_lib.vehicle.getSpeed(veh_id)
            position = traci_lib.vehicle.getPosition(veh_id)  # (x, y)

            # ========== 1. 位置权重：优先选择瓶颈区域（最多25分） ==========
            # ✅ 修复：使用真实的s坐标判断是否在瓶颈区域
            if hasattr(self, 'frenet_system') and self.frenet_system is not None:
                try:
                    # 计算Frenet坐标
                    x, y = position
                    edge_id = lane_id.split('_')[0] if '_' in lane_id else lane_id

                    # 获取真实的s坐标
                    s, d = self.frenet_system.cartesian_to_frenet(x, y, edge_id, lane_id)

                    # 判断是否在瓶颈区域
                    in_bottleneck = self.frenet_system.is_in_bottleneck(
                        s=s,
                        edge_id=edge_id
                    )

                    if in_bottleneck:
                        score += 15.0  # 瓶颈区域车辆优先

                    # ✅ 额外：距离瓶颈越近，权重越高
                    # 获取瓶颈区域的s范围（如果有的话）
                    if hasattr(self.frenet_system, 'bottleneck_s_range'):
                        s_min, s_max = self.frenet_system.bottleneck_s_range.get(edge_id, (0, 0))
                        if s_max > s_min:
                            # 计算到瓶颈中心的距离
                            bottleneck_center = (s_min + s_max) / 2.0
                            dist_to_bottleneck = abs(s - bottleneck_center)
                            # 距离越近，分数越高（最高10分）
                            proximity_score = max(0, 10.0 - dist_to_bottleneck / 100.0)
                            score += proximity_score

                except Exception as e:
                    # 如果Frenet坐标计算失败，使用简化的边缘判断
                    edge_id = lane_id.split('_')[0] if '_' in lane_id else lane_id
                    # 简化判断：某些edge_id可能包含瓶颈关键字
                    is_bottleneck_edge = any(keyword in edge_id.lower()
                                             for keyword in ['bottleneck', 'ramp', 'merge', 'junction'])
                    if is_bottleneck_edge:
                        score += 10.0  # 降低权重，因为不够精确
            else:
                # 如果没有Frenet系统，使用简化的边缘判断
                edge_id = lane_id.split('_')[0] if '_' in lane_id else lane_id
                is_bottleneck_edge = any(keyword in edge_id.lower()
                                         for keyword in ['bottleneck', 'ramp', 'merge', 'junction'])
                if is_bottleneck_edge:
                    score += 10.0

            # ========== 2. 速度权重：优先选择速度异常的车辆（最多10分） ==========
            if speed < 5.0:
                score += 10.0  # 慢速车（可能是拥堵源）
            elif speed > 20.0:
                score += 5.0   # 快速车（需要协调）

            # ========== 3. 车道权重：优先选择关键车道（最多5分） ==========
            # 最外侧车道（index=0）通常是汇流车道
            if lane_index == 0:
                score += 5.0
            elif lane_index == 1:
                score += 3.0
            # 中间车道权重较低

            # ========== 4. 加速度权重：优先选择急加减速的车辆（5分） ==========
            acceleration = traci_lib.vehicle.getAcceleration(veh_id)
            if abs(acceleration) > 2.0:
                score += 5.0  # 急加减速（不稳定因素）

            # ========== 5. 跟驰距离权重：优先选择跟驰距离近的车辆（最多8分） ==========
            # 获取前车信息
            leader_id = traci_lib.vehicle.getLeader(veh_id, 100.0)
            if leader_id and leader_id in all_vehicle_ids:
                leader_speed = traci_lib.vehicle.getSpeed(leader_id)
                speed_diff = speed - leader_speed
                if speed_diff < -5.0:  # 比前车慢很多（可能是瓶颈）
                    score += 8.0
                elif speed_diff > 5.0:  # 比前车快很多（可能追尾风险）
                    score += 6.0

        except Exception as e:
            # 如果获取车辆信息失败，给予最低分数
            score = 0.0

        return score

    def _dynamic_update_icv(
        self,
        all_vehicle_ids: List[str],
        traci_lib,
        num_icv: int
    ) -> set:
        """
        ✅ 动态更新ICV集合：释放低重要性车辆，招募高重要性车辆

        工作流程：
        1. 重新评估当前ICV的重要性评分
        2. 识别低重要性ICV（评分低于阈值）
        3. 释放低重要性ICV的名额
        4. 从非ICV车辆池中招募高重要性车辆
        5. 更新ICV集合和评分缓存

        Args:
            all_vehicle_ids: 当前所有车辆ID列表
            traci_lib: TraCI库实例
            num_icv: 目标ICV数量

        Returns:
            更新后的ICV集合
        """
        # ========== 1. 重新评估当前ICV的重要性 ==========
        current_scores = {}
        icv_still_in_network = set()

        for veh_id in self.current_icv_ids:
            if veh_id in all_vehicle_ids:
                # 车辆仍在路网中，重新计算评分
                score = self._compute_vehicle_score(veh_id, traci_lib, all_vehicle_ids)
                current_scores[veh_id] = score
                icv_still_in_network.add(veh_id)
            # else: 车辆已经离开路网，不需要保留

        # ========== 2. 识别低重要性ICV ==========
        # 阈值设置：如果评分<10分，认为重要性不够，需要释放
        release_threshold = 10.0
        low_importance_icvs = {
            veh_id for veh_id, score in current_scores.items()
            if score < release_threshold
        }

        # ========== 3. 释放低重要性ICV ==========
        remaining_icvs = icv_still_in_network - low_importance_icvs

        # ========== 4. 从非ICV车辆池中招募新高重要性车辆 ==========
        num_to_replace = num_icv - len(remaining_icvs)

        if num_to_replace > 0:
            # 候选车辆池：当前不是ICV的车辆
            non_icv_vehicles = set(all_vehicle_ids) - remaining_icvs

            if non_icv_vehicles:
                # 计算候选车辆的评分
                candidate_scores = {}
                for veh_id in non_icv_vehicles:
                    score = self._compute_vehicle_score(veh_id, traci_lib, all_vehicle_ids)
                    candidate_scores[veh_id] = score

                # 选择评分最高的K辆车辆
                sorted_candidates = sorted(
                    candidate_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                new_icvs = set([veh_id for veh_id, score in sorted_candidates[:num_to_replace]])
                remaining_icvs.update(new_icvs)

        # ========== 5. 更新状态 ==========
        # ✅ 关键修复：确保ICV数量不超过目标值（硬约束）
        remaining_icvs = set(list(remaining_icvs)[:num_icv])  # 强制裁剪到num_icv

        self.current_icv_ids = remaining_icvs
        self.icv_scores = {
            veh_id: current_scores.get(veh_id, 0)
            for veh_id in remaining_icvs
        }
        self.icv_selection_step = self.current_step

        # ========== 调试信息（可选） ==========
        if self.current_step % 100 == 0:
            print(f"[ICV Update] Step {self.current_step}:")
            print(f"  - Released {len(low_importance_icvs)} low-importance ICVs")
            print(f"  - Current ICV count: {len(remaining_icvs)}/{num_icv} (hard constraint applied)")

            if len(self.icv_scores) > 0:
                scores_list = list(self.icv_scores.values())
                print(f"  - ICV scores: min={min(scores_list):.1f}, max={max(scores_list):.1f}, "
                      f"mean={sum(scores_list)/len(scores_list):.1f}")

        return remaining_icvs

    def _intelligent_select_icv(
        self,
        all_vehicle_ids: List[str],
        traci_lib,
        num_icv: int
    ) -> set:
        """
        ✅ 智能选择ICV车辆（基于影响力评分）

        优先选择关键车辆：
        1. 瓶颈区域的车辆（汇流区、减速区）
        2. 速度异常的车辆（过慢或过快）
        3. 关键位置的车辆（最外侧车道、上游瓶颈）

        Args:
            all_vehicle_ids: 所有车辆ID列表
            traci_lib: TraCI库实例
            num_icv: 需要选择的ICV数量

        Returns:
            选中车辆ID的集合
        """
        if len(all_vehicle_ids) <= num_icv:
            # 如果车辆总数少于需要选择的数量，全部选择
            return set(all_vehicle_ids)

        # 计算每辆车的关键性评分
        vehicle_scores = {}
        for veh_id in all_vehicle_ids:
            score = self._compute_vehicle_score(veh_id, traci_lib, all_vehicle_ids)
            vehicle_scores[veh_id] = score

        # 选择评分最高的K辆车辆
        sorted_vehicles = sorted(
            vehicle_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 取前K辆
        selected_vehicles = set([veh_id for veh_id, score in sorted_vehicles[:num_icv]])

        # 调试信息（可选）
        if self.current_step % 100 == 0:  # 每100步打印一次
            top_scores = [score for _, score in sorted_vehicles[:5]]
            print(f"[ICV Selection] Selected {len(selected_vehicles)} ICVs out of {len(all_vehicle_ids)} vehicles")
            print(f"  Top scores: {top_scores}")

        return selected_vehicles

    def _compute_competition_global_stats(self, vehicle_states: Dict[str, Dict]) -> np.ndarray:
        """
        计算比赛需要的全局统计特征（向量化优化版本）

        返回32维向量：
        [0-15]: 原有统计
        [16-23]: 效率相关（吞吐量、平均旅行时间等）
        [24-31]: 稳定性相关（拥堵指标等）
        """
        if not vehicle_states:
            return np.zeros(32)

        stats = np.zeros(32)
        num_vehicles = len(vehicle_states)

        # [0-15]: 原有统计（使用父类的GPU统计方法）
        try:
            # 向量化提取数据
            vehicle_data = [[
                state.get('s', 0.0),
                state.get('d', 0.0),
                state.get('vs', 0.0),
                state.get('vd', 0.0),
                state.get('speed', 0.0),
                state.get('acceleration', 0.0),
                state.get('lane_index', 0.0),
                state.get('angle', 0.0),
                1.0 if vid in self.icv_ids else 0.0
            ] for vid, state in vehicle_states.items()]

            if vehicle_data:
                states_tensor = torch.tensor(vehicle_data, dtype=torch.float32, device=self.device)
                base_stats_tensor = self._compute_global_stats_gpu(states_tensor)
                stats[:16] = base_stats_tensor.cpu().numpy()
            else:
                stats[:16] = 0.0
        except Exception as e:
            # 如果GPU统计失败，使用零向量
            stats[:16] = 0.0

        # 向量化提取所有速度和加速度
        speeds = np.array([v['speed'] for v in vehicle_states.values()])
        accelerations = np.array([v['acceleration'] for v in vehicle_states.values()])
        d_values = np.array([abs(v['d']) for v in vehicle_states.values()])

        # [16-19]: 效率指标（向量化计算）
        stats[16] = num_vehicles / max(self.current_step, 1)  # 吞吐量
        stats[17] = np.mean(speeds)  # 平均速度
        stats[18] = np.mean(speeds > 5.0)  # 高速车辆比例
        stats[19] = np.median(speeds)  # 速度中位数

        # [20-23]: 拥堵指标（向量化计算）
        stats[20] = np.mean(speeds < 1.0)  # 慢速车辆比例
        stats[21] = np.std(speeds) if num_vehicles > 1 else 0.0  # 速度标准差
        stats[22] = np.mean(accelerations < -2.0)  # 急减速比例
        stats[23] = np.mean(speeds < 0.1)  # 停车比例

        # [24-27]: 车道利用率（向量化计算）
        lane_indices = np.array([v['lane_index'] for v in vehicle_states.values()])
        unique_lanes, lane_counts = np.unique(lane_indices, return_counts=True)

        if len(unique_lanes) > 0:
            stats[24] = np.max(lane_counts) / num_vehicles  # 最拥挤车道占有率
            stats[25] = len(unique_lanes)  # 使用车道数
            stats[26] = np.std(lane_counts) if len(unique_lanes) > 1 else 0.0  # 车道分布标准差
        else:
            stats[24:27] = 0.0

        stats[27] = np.mean(d_values > 1.0)  # 偏离车道中心的比例

        # [28-31]: 预留空间（可添加更多指标）
        stats[28:32] = 0.0

        return stats

    def _track_interventions(self, actions: Dict[str, np.ndarray]):
        """跟踪干预统计"""
        if not actions:
            return

        # 导入traci（可能是libsumo）
        try:
            import libsumo as traci_lib
        except ImportError:
            import traci as traci_lib

        for veh_id, action in actions.items():
            if veh_id not in traci_lib.vehicle.getIDList():
                continue

            try:
                old_accel = traci_lib.vehicle.getAcceleration(veh_id)
                old_lane = traci_lib.vehicle.getLaneIndex(veh_id)

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
                speed = traci_lib.vehicle.getSpeed(veh_id)
                if action[0] > 0:
                    self.intervention_stats['energy_consumption'] += speed * action[0] * 0.1  # dt=0.1s

            except:
                continue

        self.intervention_stats['controlled_vehicles'] = len(actions)

    def _update_performance_metrics(self, observation: Dict[str, Any]):
        """更新性能指标（向量化优化版本）"""
        vehicle_states = observation.get('vehicle_states', {})

        if not vehicle_states:
            return

        # 向量化提取速度
        speeds = np.array([v['speed'] for v in vehicle_states.values()])
        num_vehicles = len(speeds)

        # 更新速度统计
        self.performance_metrics['avg_speed'] = np.mean(speeds)
        self.performance_metrics['speed_std'] = np.std(speeds) if num_vehicles > 1 else 0.0

        # 更新停车比例（向量化）
        self.performance_metrics['stopped_ratio'] = np.mean(speeds < 0.1)

        # 更新吞吐量
        current_time = self.current_step * self.config.get('step_length', 0.1)
        self.performance_metrics['throughput'] = len(self.stats.get('arrived_vehicles', [])) / max(current_time, 1.0)

        # 更新拥堵水平（向量化）
        self.performance_metrics['congestion_level'] = np.mean(speeds < 1.0)

        # 更新旅行时间（向量化处理）
        arrived_vehicles = self.stats.get('arrived_vehicles', [])
        if arrived_vehicles:
            # 批量处理到达车辆的旅行时间
            for veh_id in arrived_vehicles:
                if veh_id in self.vehicle_departure_times:
                    departure_step = self.vehicle_departure_times.pop(veh_id, self.current_step)
                    self.vehicle_travel_times[veh_id] = self.current_step - departure_step

        if self.vehicle_travel_times:
            self.performance_metrics['avg_travel_time'] = np.mean(list(self.vehicle_travel_times.values()))

    def _compute_competition_reward(self, observation: Dict[str, Any]) -> float:
        """
        计算比赛标准奖励（向量化优化版本）

        奖励 = 效率得分 + 稳定性得分 - 干预成本惩罚

        初赛权重：效率为主
        """
        vehicle_states = observation.get('vehicle_states', {})

        if not vehicle_states:
            return 0.0

        # 向量化提取速度
        speeds = np.array([v['speed'] for v in vehicle_states.values()])
        num_vehicles = len(speeds)

        # ========== 1. 效率得分 Sefficiency ==========
        # 1.1 速度得分（鼓励高速）
        avg_speed = np.mean(speeds)
        speed_score = avg_speed * self.speed_norm_factor  # 使用配置的归一化因子

        # 1.2 吞吐量得分（鼓励高到达率）
        arrived_count = len(self.stats.get('arrived_vehicles', []))
        current_time = self.current_step * self.config.get('step_length', 0.1)
        throughput_score = arrived_count / max(current_time, 1.0) * self.throughput_weight  # 使用配置的权重

        # 1.3 完成率得分
        departed_count = len(self.stats.get('departed_vehicles', []))
        completion_score = (arrived_count / max(departed_count, 1)) if departed_count > 0 else 0.0

        # 综合效率得分（使用配置的权重）
        w_speed, w_throughput, w_completion = self.efficiency_weights
        efficiency_score = (
            w_speed * speed_score +
            w_throughput * throughput_score +
            w_completion * completion_score
        )

        # ========== 2. 稳定性得分 Sstability ==========
        # 2.1 速度方差惩罚
        if num_vehicles > 1:
            speed_std = np.std(speeds)
            stability_score = -self.stability_weight * (speed_std / 10.0)  # 使用配置的权重
        else:
            stability_score = 0.0

        # 2.2 拥堵惩罚（向量化）
        congestion_ratio = np.mean(speeds < 1.0)
        congestion_penalty = -self.congestion_penalty_weight * congestion_ratio

        # 2.3 停车惩罚（向量化）
        stopped_ratio = np.mean(speeds < 0.1)
        stopped_penalty = -self.stopped_penalty_weight * stopped_ratio

        # 综合稳定性得分
        stability_total = stability_score + congestion_penalty + stopped_penalty

        # ========== 3. 干预成本惩罚 Pint ==========
        # 3.1 受控车辆数量惩罚（初赛中ICV比例固定为25%，这里只惩罚控制幅度）
        control_magnitude = self.intervention_stats.get('control_magnitude', 0.0)
        magnitude_penalty = 0.01 * control_magnitude / max(num_vehicles, 1)

        # 3.2 换道惩罚（换道是高风险操作）
        lane_changes = self.intervention_stats.get('total_lane_changes', 0)
        lane_change_penalty = 0.1 * lane_changes / max(num_vehicles, 1)

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
