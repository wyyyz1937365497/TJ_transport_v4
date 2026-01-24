#!/usr/bin/env python3
"""
OCR-MAX模型提交脚本

功能：
1. 集成SimplifiedICVPolicy模型到官方评测框架
2. 每个仿真步调用模型获取控制指令
3. 生成符合要求的submit.xlsx文件

使用方法：
    python submit_solution.py

输出：
    - submit.xlsx (提交文件)
"""

import os
import sys
import traci
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
import json
import numpy as np
import torch
from pathlib import Path
from openpyxl.styles import Font, Alignment, PatternFill

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模型
from src.models.simplified_icv_policy import SimplifiedICVPolicy


class OCRMAXSubmission:
    """
    OCR-MAX模型提交类

    集成SimplifiedICVPolicy到官方评测框架
    """

    def __init__(self, sumo_cfg_path, checkpoint_path, device='cuda'):
        """
        Args:
            sumo_cfg_path: SUMO配置文件路径
            checkpoint_path: 模型权重路径
            device: 运行设备
        """
        self.sumo_cfg_path = sumo_cfg_path
        self.checkpoint_path = checkpoint_path
        self.device = device

        # 数据存储（与官方框架一致）
        self.vehicle_data = []
        self.step_data = []
        self.route_data = {}
        self.vehicle_od_data = {}

        # 累计统计
        self.cumulative_departed = 0
        self.cumulative_arrived = 0
        self.all_departed_vehicles = set()
        self.all_arrived_vehicles = set()

        # 仿真参数
        self.flow_rate = 0
        self.simulation_time = 0
        self.step_length = 1.0
        self.total_demand = 0

        # 红绿灯监控（与官方一致）
        self.traffic_lights = ['J5', 'J14', 'J15', 'J17']
        self.available_traffic_lights = []

        # ========== OCR-MAX模型相关 ==========
        self.policy = None
        self.max_vehicles = 32
        self.node_dim = 9
        self.obs_dim = 321  # 32*9 + 32 + 1

        # ICV跟踪
        self.icv_ratio = 0.10
        self.icv_ids = set()

        print("=" * 70)
        print("OCR-MAX模型提交框架")
        print("=" * 70)
        print(f"模型权重: {checkpoint_path}")
        print(f"设备: {device}")
        print("=" * 70)

    # ========================================================================
    # 第一部分: 环境初始化 (与官方框架一致)
    # ========================================================================

    def parse_config(self):
        """解析SUMO配置文件"""
        print("\n[第一部分] 正在初始化环境...")

        tree = ET.parse(self.sumo_cfg_path)
        root = tree.getroot()

        config_dir = os.path.dirname(self.sumo_cfg_path)

        # 获取路网和路径文件
        for input_elem in root.findall('.//input'):
            net_file = input_elem.find('net-file')
            if net_file is not None:
                net_file_path = net_file.get('value')
                if not os.path.isabs(net_file_path):
                    net_file_path = os.path.join(config_dir, net_file_path)
                self.net_file = net_file_path

            route_files = input_elem.find('route-files')
            if route_files is not None:
                route_file_path = route_files.get('value')
                if not os.path.isabs(route_file_path):
                    route_file_path = os.path.join(config_dir, route_file_path)
                self.routes_file = route_file_path

        # 获取时间步长
        time_step = root.find('.//step-length')
        if time_step is not None:
            self.step_length = float(time_step.get('value', 1.0))

        print(f"✓ 配置解析完成:")
        print(f"  - 网络文件: {self.net_file}")
        print(f"  - 路径文件: {self.routes_file}")
        print(f"  - 时间步长: {self.step_length}s")

    def parse_routes(self):
        """解析路径文件"""
        if not self.routes_file or not os.path.exists(self.routes_file):
            print("⚠️  路径文件不存在")
            return

        try:
            tree = ET.parse(self.routes_file)
            root = tree.getroot()

            total_vehs_per_hour = 0
            max_end_time = 0
            total_demand = 0

            for flow in root.findall('flow'):
                vehs_per_hour = float(flow.get('vehsPerHour', 0))
                begin_time = float(flow.get('begin', 0))
                end_time = float(flow.get('end', 0))

                duration_hours = (end_time - begin_time) / 3600.0
                flow_demand = vehs_per_hour * duration_hours
                total_demand += flow_demand

                total_vehs_per_hour += vehs_per_hour
                max_end_time = max(max_end_time, end_time)

            trip_count = len(root.findall('trip'))
            total_demand += trip_count

            self.simulation_time = max_end_time
            self.flow_rate = total_vehs_per_hour / 3600.0
            self.total_demand = total_demand

            print(f"✓ 交通需求分析:")
            print(f"  - 流量率: {self.flow_rate:.4f} veh/s")
            print(f"  - 仿真时长: {self.simulation_time:.2f} s")
            print(f"  - 理论总需求: {self.total_demand:.0f} 车辆")

        except Exception as e:
            print(f"❌ 路径文件解析失败: {e}")

    def initialize_traffic_lights(self):
        """初始化红绿灯监控"""
        try:
            all_tls = traci.trafficlight.getIDList()

            for tl_id in self.traffic_lights:
                if tl_id in all_tls:
                    self.available_traffic_lights.append(tl_id)

            print(f"✓ 红绿灯监控设置:")
            print(f"  - 可用红绿灯: {self.available_traffic_lights}")

        except Exception as e:
            print(f"❌ 红绿灯初始化失败: {e}")

    def load_model(self):
        """加载OCR-MAX模型"""
        print("\n[第二部分] 正在加载OCR-MAX模型...")

        try:
            # 创建模型
            self.policy = SimplifiedICVPolicy(
                obs_dim=self.obs_dim,
                node_dim=self.node_dim,
                hidden_dim=128,
                num_layers=3,
                num_vehicles=self.max_vehicles,
                device=self.device,
                use_safety_shield=True  # 推理时启用
            )

            # 加载权重
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False
            )
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.policy.eval_mode()

            print(f"✓ 模型加载成功:")
            print(f"  - Checkpoint: {self.checkpoint_path}")
            print(f"  - 模型参数: {sum(p.numel() for p in self.policy.parameters()):,}")

            return True

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def initialize_environment(self, use_gui=False, max_steps=3600):
        """初始化环境"""
        print("\n[第一部分] 正在启动SUMO仿真...")

        # 解析配置
        self.parse_config()
        self.parse_routes()

        # 启动SUMO
        sumo_binary = "sumo-gui" if use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_cfg_path,
            "--no-warnings", "true",
            "--duration-log.statistics", "true"
        ]

        try:
            traci.start(sumo_cmd)
            print(f"✓ SUMO启动成功 (模式: {'GUI' if use_gui else 'CLI'})")
        except Exception as e:
            print(f"❌ SUMO启动失败: {e}")
            return False

        # 初始化红绿灯
        self.initialize_traffic_lights()

        # 加载模型
        if not self.load_model():
            return False

        print("✓ 环境初始化完成!\n")
        return True

    # ========================================================================
    # 第二部分: 观测处理与控制算法实现
    # ========================================================================

    def get_observation_dict(self):
        """
        从SUMO获取观测字典

        Returns:
            obs_dict: {
                'vehicle_ids': list,
                'vehicle_states': {veh_id: {...}},
                'icv_ids': set,
                'global_stats': np.array
            }
        """
        vehicle_ids = traci.vehicle.getIDList()
        vehicle_states = {}

        # 选择ICV（前10%车辆）
        num_icv = max(1, int(len(vehicle_ids) * self.icv_ratio))

        # 更新ICV集合（保持已有ICV，添加新的）
        if len(self.icv_ids) < num_icv:
            # 按照ID排序选择前N辆
            sorted_vehicles = sorted(vehicle_ids)
            self.icv_ids = set(sorted_vehicles[:num_icv])

        # 只保留当前存在的车辆
        self.icv_ids = self.icv_ids.intersection(set(vehicle_ids))

        # 收集车辆状态
        for veh_id in vehicle_ids:
            try:
                # 获取位置和速度
                position = traci.vehicle.getLanePosition(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                acceleration = traci.vehicle.getAcceleration(veh_id)
                lane_index = traci.vehicle.getLaneIndex(veh_id)
                angle = traci.vehicle.getAngle(veh_id)

                # 获取Frenet坐标（简化版）
                road_id = traci.vehicle.getRoadID(veh_id)
                route = traci.vehicle.getRoute(veh_id)
                route_index = traci.vehicle.getRouteIndex(veh_id)

                # 简化的s, d坐标（相对于edge）
                s = position
                d = lane_index * 3.5  # 假设车道宽度3.5m

                # 简化的vs, vd（基于速度和角度）
                vs = speed * np.cos(np.radians(angle))
                vd = speed * np.sin(np.radians(angle))

                vehicle_states[veh_id] = {
                    's': s,
                    'd': d,
                    'vs': vs,
                    'vd': vd,
                    'speed': speed,
                    'acceleration': acceleration,
                    'lane_index': lane_index,
                    'angle': angle
                }
            except Exception as e:
                continue

        # 全局统计（32维）
        global_stats = np.zeros(32, dtype=np.float32)
        if len(vehicle_ids) > 0:
            speeds = [vehicle_states.get(vid, {}).get('speed', 0) for vid in vehicle_ids]
            accels = [vehicle_states.get(vid, {}).get('acceleration', 0) for vid in vehicle_ids]

            # 前16维：速度统计
            if len(speeds) > 0:
                global_stats[0] = np.mean(speeds)
                global_stats[1] = np.std(speeds) if len(speeds) > 1 else 0
                global_stats[2] = np.max(speeds)
                global_stats[3] = np.min(speeds)

            # 第16-31维：加速度统计
            if len(accels) > 0:
                global_stats[16] = np.mean(accels)
                global_stats[17] = np.std(accels) if len(accels) > 1 else 0
                global_stats[18] = np.max(accels)
                global_stats[19] = np.min(accels)

        return {
            'vehicle_ids': vehicle_ids,
            'vehicle_states': vehicle_states,
            'icv_ids': self.icv_ids,
            'global_stats': global_stats
        }

    def flatten_observation(self, obs_dict):
        """
        将观测字典转换为扁平化张量

        Args:
            obs_dict: 观测字典

        Returns:
            flattened_obs: [obs_dim] 扁平化观测
        """
        vehicle_states = obs_dict.get('vehicle_states', {})
        vehicle_ids = obs_dict.get('vehicle_ids', [])
        icv_ids = obs_dict.get('icv_ids', set())
        global_stats = obs_dict.get('global_stats', np.zeros(32))

        num_vehicles = len(vehicle_ids)

        # 提取9维车辆特征（归一化）
        vehicle_features = np.zeros((self.max_vehicles, 9), dtype=np.float32)

        for i, veh_id in enumerate(vehicle_ids[:self.max_vehicles]):
            if veh_id in vehicle_states:
                state = vehicle_states[veh_id]
                vehicle_features[i, 0] = state.get('s', 0.0) / 1000.0
                vehicle_features[i, 1] = state.get('d', 0.0) / 10.0
                vehicle_features[i, 2] = state.get('vs', 0.0) / 30.0
                vehicle_features[i, 3] = state.get('vd', 0.0) / 10.0
                vehicle_features[i, 4] = state.get('speed', 0.0) / 30.0
                vehicle_features[i, 5] = state.get('acceleration', 0.0) / 3.0
                vehicle_features[i, 6] = state.get('lane_index', 0.0) / 10.0
                vehicle_features[i, 7] = state.get('angle', 0.0) / 360.0
                vehicle_features[i, 8] = 1.0 if veh_id in icv_ids else 0.0

        # 确保global_stats是32维
        global_stats_flat = global_stats.flatten()
        if len(global_stats_flat) < 32:
            global_stats_flat = np.concatenate([
                global_stats_flat,
                np.zeros(32 - len(global_stats_flat), dtype=np.float32)
            ])
        elif len(global_stats_flat) > 32:
            global_stats_flat = global_stats_flat[:32]

        # 扁平化并拼接
        vehicle_features_flat = vehicle_features.flatten()
        num_vehicles_array = np.array([num_vehicles], dtype=np.float32)

        flattened_obs = np.concatenate([
            vehicle_features_flat,
            global_stats_flat,
            num_vehicles_array
        ])

        return flattened_obs

    def convert_actions_to_dict(self, action_array, vehicle_ids, icv_ids):
        """
        将扁平动作数组转换为字典格式

        Args:
            action_array: [max_vehicles * 2] 扁平动作数组
            vehicle_ids: 车辆ID列表
            icv_ids: ICV ID集合

        Returns:
            actions_dict: {vehicle_id: [acceleration, lane_change]}
        """
        actions_dict = {}

        # 重塑为 [max_vehicles, 2]
        actions_reshaped = action_array.reshape(self.max_vehicles, 2)

        for i, veh_id in enumerate(vehicle_ids):
            if veh_id in icv_ids and i < self.max_vehicles:
                accel = actions_reshaped[i, 0].item()
                lane_change = actions_reshaped[i, 1].item()

                # 反归一化加速度
                accel = accel * 5.0 - 1.0  # 映射到合理范围

                # 限制范围
                accel = np.clip(accel, -4.5, 2.0)
                lane_change = np.clip(lane_change, 0.0, 1.0)

                actions_dict[veh_id] = np.array([accel, lane_change])

        return actions_dict

    def apply_control_actions(self, actions_dict):
        """
        应用控制动作到SUMO

        Args:
            actions_dict: {vehicle_id: [acceleration, lane_change]}
        """
        for veh_id, action in actions_dict.items():
            try:
                accel = action[0]
                lane_change_prob = action[1]

                # 设置加速度（通过设置速度实现）
                current_speed = traci.vehicle.getSpeed(veh_id)
                new_speed = max(0, current_speed + accel * 0.1)  # 0.1s步长
                traci.vehicle.setSpeed(veh_id, new_speed)

                # 换道决策（概率性）
                if lane_change_prob > 0.7:  # 阈值可调
                    current_lane = traci.vehicle.getLaneIndex(veh_id)
                    road_id = traci.vehicle.getRoadID(veh_id)

                    # 获取车道数量
                    lane_count = traci.edge.getLaneNumber(road_id)

                    # 尝试向左或向右换道
                    if current_lane < lane_count - 1:
                        try:
                            traci.vehicle.changeLane(veh_id, current_lane + 1, 2.0)
                        except:
                            pass
                    elif current_lane > 0:
                        try:
                            traci.vehicle.changeLane(veh_id, current_lane - 1, 2.0)
                        except:
                            pass

            except Exception as e:
                continue

    def apply_control_algorithm(self, step):
        """
        应用OCR-MAX控制算法

        参数:
            step: 当前仿真步数
        """
        # 获取观测
        obs_dict = self.get_observation_dict()

        # 转换为扁平化观测
        obs_flat = self.flatten_observation(obs_dict)

        # 转换为张量
        obs_tensor = torch.from_numpy(obs_flat).unsqueeze(0).float().to(self.device)

        # 模型推理（确定性）
        with torch.no_grad():
            outputs = self.policy(obs_tensor, deterministic=True)
            action = outputs['actions'][0].cpu().numpy()  # [max_vehicles * 2]

        # 转换为动作字典
        vehicle_ids = obs_dict['vehicle_ids']
        icv_ids = obs_dict['icv_ids']
        actions_dict = self.convert_actions_to_dict(action, vehicle_ids, icv_ids)

        # 应用动作到SUMO
        self.apply_control_actions(actions_dict)

    # ========================================================================
    # 第三部分: 数据收集与统计（与官方框架一致）
    # ========================================================================

    def get_traffic_light_states(self):
        """获取红绿灯状态"""
        tl_states = {}

        for tl_id in self.available_traffic_lights:
            try:
                state = traci.trafficlight.getRedYellowGreenState(tl_id)
                phase = traci.trafficlight.getPhase(tl_id)
                remaining_time = traci.trafficlight.getNextSwitch(tl_id) - traci.simulation.getTime()

                tl_states[f'{tl_id}_state'] = state
                tl_states[f'{tl_id}_phase'] = phase
                tl_states[f'{tl_id}_remaining_time'] = remaining_time

            except Exception as e:
                tl_states[f'{tl_id}_state'] = 'unknown'
                tl_states[f'{tl_id}_phase'] = -1
                tl_states[f'{tl_id}_remaining_time'] = -1

        return tl_states

    def get_vehicle_od(self, veh_id):
        """获取车辆OD信息"""
        if veh_id in self.vehicle_od_data:
            return self.vehicle_od_data[veh_id]

        try:
            route = traci.vehicle.getRoute(veh_id)
            if len(route) >= 2:
                origin = route[0]
                destination = route[-1]
            elif len(route) == 1:
                origin = route[0]
                destination = route[0]
            else:
                origin = "unknown"
                destination = "unknown"

            od_info = {
                'origin': origin,
                'destination': destination,
                'route_length': len(route)
            }

            self.vehicle_od_data[veh_id] = od_info
            return od_info

        except:
            od_info = {
                'origin': "unknown",
                'destination': "unknown",
                'route_length': 0
            }
            self.vehicle_od_data[veh_id] = od_info
            return od_info

    def get_route_length(self, edges):
        """计算路径总长度"""
        total_length = 0
        for edge_id in edges:
            try:
                edge_length = traci.edge.getLength(edge_id)
                total_length += edge_length
            except:
                try:
                    lane_id = f"{edge_id}_0"
                    edge_length = traci.lane.getLength(lane_id)
                    total_length += edge_length
                except:
                    total_length += 100
        return total_length

    def calculate_traveled_distance(self, veh_id, route_info):
        """计算车辆已行驶距离"""
        try:
            current_edge = traci.vehicle.getRoadID(veh_id)
            current_position = traci.vehicle.getLanePosition(veh_id)
            route_edges = route_info['route_edges']

            traveled = 0
            for edge in route_edges:
                if edge == current_edge:
                    traveled += current_position
                    break
                else:
                    try:
                        edge_length = traci.edge.getLength(edge)
                        traveled += edge_length
                    except:
                        traveled += 100

            return min(traveled, route_info['route_length'])
        except:
            return 0

    def collect_step_data(self, step):
        """收集每个时间步的数据"""
        current_time = step * self.step_length

        # 获取当前活跃车辆
        current_vehicle_ids = set(traci.vehicle.getIDList())

        # 更新累计统计
        current_arrived_ids = set(traci.simulation.getArrivedIDList())
        current_departed_ids = set(traci.simulation.getDepartedIDList())

        new_arrivals = current_arrived_ids - self.all_arrived_vehicles
        self.all_arrived_vehicles.update(new_arrivals)
        self.cumulative_arrived = len(self.all_arrived_vehicles)

        new_departures = current_departed_ids - self.all_departed_vehicles
        self.all_departed_vehicles.update(new_departures)
        self.cumulative_departed = len(self.all_departed_vehicles)

        # 获取红绿灯状态
        traffic_light_states = self.get_traffic_light_states()

        # 记录时间步级数据
        step_record = {
            'step': step,
            'time': current_time,
            'active_vehicles': len(current_vehicle_ids),
            'arrived_vehicles': self.cumulative_arrived,
            'departed_vehicles': self.cumulative_departed,
            'current_arrivals': len(new_arrivals),
            'current_departures': len(new_departures)
        }
        step_record.update(traffic_light_states)
        self.step_data.append(step_record)

        # 收集车辆级数据
        for veh_id in current_vehicle_ids:
            try:
                speed = traci.vehicle.getSpeed(veh_id)
                position = traci.vehicle.getLanePosition(veh_id)
                edge_id = traci.vehicle.getRoadID(veh_id)
                route_index = traci.vehicle.getRouteIndex(veh_id)

                od_info = self.get_vehicle_od(veh_id)

                if veh_id not in self.route_data:
                    route_edges = traci.vehicle.getRoute(veh_id)
                    route_length = self.get_route_length(route_edges)
                    self.route_data[veh_id] = {
                        'route_edges': route_edges,
                        'route_length': route_length
                    }

                route_info = self.route_data[veh_id]
                traveled_distance = self.calculate_traveled_distance(veh_id, route_info)
                completion_rate = min(traveled_distance / max(route_info['route_length'], 1), 1.0)

                vehicle_record = {
                    'step': step,
                    'time': current_time,
                    'vehicle_id': veh_id,
                    'speed': speed,
                    'position': position,
                    'edge_id': edge_id,
                    'route_index': route_index,
                    'traveled_distance': traveled_distance,
                    'route_length': route_info['route_length'],
                    'completion_rate': completion_rate,
                    'origin': od_info['origin'],
                    'destination': od_info['destination'],
                    'route_edges_count': od_info['route_length']
                }
                self.vehicle_data.append(vehicle_record)

            except Exception as e:
                continue

        # 进度报告
        if step % 100 == 0:
            print(f"[步骤 {step}] 活跃: {len(current_vehicle_ids)}, "
                  f"累计出发: {self.cumulative_departed}, "
                  f"累计到达: {self.cumulative_arrived}, "
                  f"ICV数量: {len(self.icv_ids)}")

    def save_to_excel(self, output_dir="competition_results"):
        """保存数据到Excel文件"""
        print(f"\n[第三部分] 正在保存数据到Excel...")

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 生成Excel文件名
        excel_file = os.path.join(output_dir, "submit.xlsx")

        # 准备仿真参数
        params = {
            'flow_rate': self.flow_rate,
            'simulation_time': self.simulation_time,
            'step_length': self.step_length,
            'total_steps': len(self.step_data),
            'total_demand': self.total_demand,
            'final_departed': self.cumulative_departed,
            'final_arrived': self.cumulative_arrived,
            'unique_vehicles': len(self.route_data),
            'monitored_traffic_lights': self.traffic_lights,
            'available_traffic_lights': self.available_traffic_lights,
            'collection_timestamp': timestamp,
            'model_checkpoint': self.checkpoint_path
        }

        # 创建Excel写入器
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:

            # 1. 写入数据汇总 Sheet
            print(f"✓ 正在创建数据汇总...")
            summary_data = [
                {'指标': '理论总需求', '数值': f"{params['total_demand']:.0f} 车辆"},
                {'指标': '实际累计出发', '数值': f"{params['final_departed']} 车辆"},
                {'指标': '实际累计到达', '数值': f"{params['final_arrived']} 车辆"},
                {'指标': 'OD完成率(OCR)', '数值': f"{params['final_arrived']/max(params['final_departed'], 1):.4f}"},
                {'指标': '唯一车辆数', '数值': f"{params['unique_vehicles']} 车辆"},
                {'指标': '总时间步数', '数值': params['total_steps']},
                {'指标': '仿真时长', '数值': f"{params['simulation_time']:.2f} 秒"},
                {'指标': '流量率', '数值': f"{params['flow_rate']:.4f} veh/s"},
                {'指标': '时间步长', '数值': f"{params['step_length']:.2f} 秒"},
                {'指标': '监控红绿灯', '数值': ', '.join(params['available_traffic_lights'])},
                {'指标': '数据收集时间', '数值': timestamp},
                {'指标': '模型权重', '数值': params['model_checkpoint']}
            ]

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='数据汇总', index=False)

            # 格式化汇总sheet
            worksheet = writer.sheets['数据汇总']
            worksheet.column_dimensions['A'].width = 25
            worksheet.column_dimensions['B'].width = 40

            for cell in worksheet[1]:
                cell.font = Font(bold=True, size=11, color="FFFFFF")
                cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                cell.alignment = Alignment(horizontal='center', vertical='center')

            # 2. 写入仿真参数 Sheet
            print(f"✓ 正在写入仿真参数...")
            params_data = []
            for key, value in params.items():
                if isinstance(value, list):
                    value = ', '.join(map(str, value))
                params_data.append({'参数名称': key, '参数值': str(value)})

            params_df = pd.DataFrame(params_data)
            params_df.to_excel(writer, sheet_name='仿真参数', index=False)

            # 格式化参数sheet
            worksheet = writer.sheets['仿真参数']
            worksheet.column_dimensions['A'].width = 30
            worksheet.column_dimensions['B'].width = 50

            for cell in worksheet[1]:
                cell.font = Font(bold=True, size=11, color="FFFFFF")
                cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                cell.alignment = Alignment(horizontal='center', vertical='center')

            # 3. 写入时间步数据 Sheet
            if self.step_data:
                print(f"✓ 正在写入时间步数据...")
                step_df = pd.DataFrame(self.step_data)
                step_df.to_excel(writer, sheet_name='时间步数据', index=False)

                # 格式化
                worksheet = writer.sheets['时间步数据']
                for cell in worksheet[1]:
                    cell.font = Font(bold=True, size=11, color="FFFFFF")
                    cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                    cell.alignment = Alignment(horizontal='center', vertical='center')

                print(f"  - 记录数: {len(step_df):,}")

            # 4. 写入车辆数据 Sheet
            if self.vehicle_data:
                print(f"✓ 正在写入车辆数据...")
                vehicle_df = pd.DataFrame(self.vehicle_data)
                vehicle_df.to_excel(writer, sheet_name='车辆数据', index=False)

                # 格式化
                worksheet = writer.sheets['车辆数据']
                for cell in worksheet[1]:
                    cell.font = Font(bold=True, size=11, color="FFFFFF")
                    cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                    cell.alignment = Alignment(horizontal='center', vertical='center')

                print(f"  - 记录数: {len(vehicle_df):,}")
                print(f"  - 唯一车辆: {vehicle_df['vehicle_id'].nunique()}")

        # 数据统计报告
        print(f"\n{'=' * 70}")
        print(f"数据收集统计报告")
        print(f"{'=' * 70}")
        print(f"理论总需求:     {self.total_demand:.0f} 车辆")
        print(f"实际累计出发:   {self.cumulative_departed} 车辆")
        print(f"实际累计到达:   {self.cumulative_arrived} 车辆")
        print(f"OD完成率(OCR):  {self.cumulative_arrived/max(self.cumulative_departed, 1):.4f}")
        print(f"{'=' * 70}")
        print(f"\n✅ Excel文件已保存: {excel_file}")
        print(f"\nExcel包含以下sheets:")
        print(f"  1. 数据汇总 - 关键指标统计")
        print(f"  2. 仿真参数 - 完整的仿真配置参数")
        print(f"  3. 时间步数据 - 每个时间步的系统状态")
        print(f"  4. 车辆数据 - 每辆车每个时间步的详细数据")

        return {'excel_file': excel_file}

    def run(self, max_steps=3600, use_gui=False):
        """运行完整的仿真流程"""
        print("\n开始运行OCR-MAX提交仿真...")
        print(f"最大步数: {max_steps}\n")

        # 第一部分: 初始化环境
        if not self.initialize_environment(use_gui=use_gui, max_steps=max_steps):
            print("❌ 环境初始化失败")
            return False

        # 仿真主循环
        print(f"\n{'=' * 70}")
        print("[第二部分] 开始运行OCR-MAX控制算法...")
        print(f"{'=' * 70}\n")

        step = 0
        try:
            while step < max_steps:
                # 执行仿真步
                traci.simulationStep()

                # 第二部分: 应用OCR-MAX控制算法
                self.apply_control_algorithm(step)

                # 第三部分: 收集数据
                self.collect_step_data(step)

                step += 1

                # 检查仿真是否结束
                if traci.simulation.getMinExpectedNumber() <= 0 and step > 100:
                    print(f"\n仿真自然结束于步骤 {step}")
                    break

        except Exception as e:
            print(f"\n❌ 仿真过程中发生错误: {e}")
            import traceback
            traceback.print_exc()

        finally:
            traci.close()

        # 第三部分: 保存数据到Excel
        print(f"\n{'=' * 70}")
        result = self.save_to_excel()

        print(f"\n✅ 仿真完成!")
        print(f"\n可使用此Excel文件进行评测提交: {result['excel_file']}")

        return True


def main():
    """主函数"""

    # ========================================================================
    # 配置区域
    # ========================================================================

    # SUMO配置文件
    sumo_cfg = "仿真环境_初赛_1.0/仿真环境-初赛/sumo_train.sumocfg"

    # 模型权重路径
    checkpoint_path = "checkpoints/ocr_max/stage2_best.pth"

    # 仿真参数
    MAX_STEPS = 3600
    USE_GUI = False  # 提交时设为False
    DEVICE = 'cuda'  # 使用cuda或cpu

    # ========================================================================

    # 检查文件是否存在
    if not os.path.exists(sumo_cfg):
        print(f"❌ 配置文件不存在: {sumo_cfg}")
        return

    if not os.path.exists(checkpoint_path):
        print(f"❌ 模型权重不存在: {checkpoint_path}")
        return

    try:
        # 创建提交实例
        submission = OCRMAXSubmission(
            sumo_cfg_path=sumo_cfg,
            checkpoint_path=checkpoint_path,
            device=DEVICE
        )

        # 运行仿真
        submission.run(max_steps=MAX_STEPS, use_gui=USE_GUI)

    except Exception as e:
        print(f"\n❌ 程序运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
