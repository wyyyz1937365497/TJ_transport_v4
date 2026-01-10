"""
SUMO环境接口 - Windows兼容版本
功能：与SUMO仿真环境交互
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import time
import pickle
import json

try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    print("⚠️  警告: TraCI未安装，SUMO功能将不可用")


class SumoEnvironment:
    """
    SUMO仿真环境基类

    特性：
    - Windows路径兼容
    - 自动启动/关闭SUMO
    - 高效数据收集
    - 异常处理
    """

    def __init__(
        self,
        config: Dict[str, Any],
        use_gui: bool = False
    ):
        self.config = config
        self.use_gui = use_gui

        # SUMO配置
        self.sumo_cfg = config.get('sumo_cfg', '')
        self.net_file = config.get('net_file', '')
        self.route_file = config.get('route_file', '')
        self.step_length = config.get('step_length', 0.1)

        # 检查路径
        self._validate_paths()

        # 构建SUMO命令
        self.sumo_cmd = self._build_sumo_command()

        # TraCI状态
        self.is_connected = False

        # 统计信息
        self.current_step = 0
        self.stats = {
            'total_steps': 0,
            'total_vehicles': 0,
            'departed_vehicles': set(),
            'arrived_vehicles': set()
        }

    def _validate_paths(self):
        """验证SUMO文件路径"""
        if not os.path.exists(self.sumo_cfg):
            raise FileNotFoundError(f"SUMO配置文件不存在: {self.sumo_cfg}")

        if self.net_file and not os.path.exists(self.net_file):
            print(f"⚠️  警告: 路网文件不存在: {self.net_file}")

        if self.route_file and not os.path.exists(self.route_file):
            print(f"⚠️  警告: 路径文件不存在: {self.route_file}")

    def _build_sumo_command(self) -> List[str]:
        """构建SUMO命令"""
        sumo_binary = "sumo-gui.exe" if self.use_gui else "sumo.exe"

        # 如果不在PATH中，尝试使用绝对路径
        if not os.path.exists(sumo_binary):
            # 尝试常见的SUMO安装路径
            possible_paths = [
                r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe",
                r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe",
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    sumo_binary = path
                    break

        cmd = [
            sumo_binary,
            "-c", self.sumo_cfg,
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--step-length", str(self.step_length),
            "--seed", str(self.config.get('seed', 42))
        ]

        # 添加远程端口（避免冲突）
        cmd.extend(["--remote-port", str(self.config.get('port', 8813))])

        return cmd

    def start(self):
        """启动SUMO仿真"""
        if not TRACI_AVAILABLE:
            raise RuntimeError("TraCI未安装，无法启动SUMO")

        if self.is_connected:
            return

        try:
            traci.start(self.sumo_cmd)
            self.is_connected = True
            self.current_step = 0
            print(f"✅ SUMO已启动 (GUI: {self.use_gui})")
        except Exception as e:
            raise RuntimeError(f"SUMO启动失败: {e}")

    def close(self):
        """关闭SUMO仿真"""
        if self.is_connected and TRACI_AVAILABLE:
            try:
                traci.close()
            except:
                pass
            self.is_connected = False
            print("✅ SUMO已关闭")

    def reset(self) -> Dict[str, Any]:
        """重置环境"""
        if self.is_connected:
            self.close()

        self.start()
        self.current_step = 0
        self.stats = {
            'total_steps': 0,
            'total_vehicles': 0,
            'departed_vehicles': set(),
            'arrived_vehicles': set()
        }

        return self._get_observation()

    def step(self, actions: Optional[Dict[str, np.ndarray]] = None) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行一步仿真

        Args:
            actions: {vehicle_id: [acceleration, lane_change]}

        Returns:
            observation, reward, done, info
        """
        if not self.is_connected:
            raise RuntimeError("SUMO未连接，请先调用reset()")

        # 应用控制动作
        if actions:
            self._apply_actions(actions)

        # 推进仿真
        traci.simulationStep()
        self.current_step += 1

        # 获取观测
        observation = self._get_observation()

        # 计算奖励
        reward = self._compute_reward(observation)

        # 检查是否结束
        done = self._is_done()

        # 额外信息
        info = self._get_info()

        return observation, reward, done, info

    def _apply_actions(self, actions: Dict[str, np.ndarray]):
        """应用控制动作到车辆"""
        for veh_id, action in actions.items():
            try:
                if veh_id not in traci.vehicle.getIDList():
                    continue

                # 解析动作
                acceleration = action[0]  # [-3, 2] m/s²
                lane_change = action[1]  # [0, 1] 概率

                # 应用加速度
                current_speed = traci.vehicle.getSpeed(veh_id)
                target_speed = max(0, current_speed + acceleration * self.step_length)
                traci.vehicle.setSpeed(veh_id, target_speed)

                # 应用换道
                if lane_change > 0.5:
                    current_lane = traci.vehicle.getLaneIndex(veh_id)
                    road_id = traci.vehicle.getRoadID(veh_id)

                    # 随机选择左右车道
                    lane_change_direction = np.random.choice([-1, 1])

                    try:
                        traci.vehicle.changeLane(
                            veh_id,
                            current_lane + lane_change_direction,
                            2.0  # 持续时间
                        )
                    except:
                        pass  # 换道失败（可能是边界）

            except Exception as e:
                # 忽略单个车辆的控制失败
                pass

    def _get_observation(self) -> Dict[str, Any]:
        """获取当前观测"""
        # 获取所有车辆
        vehicle_ids = traci.vehicle.getIDList()

        vehicle_states = {}
        icv_ids = set()

        # 根据配置选择ICV
        control_ratio = self.config.get('control_ratio', 0.25)
        num_icv = max(1, int(len(vehicle_ids) * control_ratio))

        if len(vehicle_ids) > 0:
            icv_indices = np.random.choice(
                len(vehicle_ids),
                size=min(num_icv, len(vehicle_ids)),
                replace=False
            )
            icv_ids = {vehicle_ids[i] for i in icv_indices}

        # 收集车辆状态
        for veh_id in vehicle_ids:
            try:
                position = traci.vehicle.getPosition(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                angle = traci.vehicle.getAngle(veh_id)
                lane_id = traci.vehicle.getLaneID(veh_id)
                lane_index = traci.vehicle.getLaneIndex(veh_id)
                acceleration = traci.vehicle.getAcceleration(veh_id)

                vehicle_states[veh_id] = {
                    'id': veh_id,
                    'x': position[0],
                    'y': position[1],
                    'z': 0.0,
                    'speed': speed,
                    'vx': speed * np.cos(np.radians(angle)),
                    'vy': speed * np.sin(np.radians(angle)),
                    'ax': acceleration * np.cos(np.radians(angle)),
                    'ay': acceleration * np.sin(np.radians(angle)),
                    'angle': angle,
                    'lane_id': lane_id,
                    'lane_index': lane_index,
                    'acceleration': acceleration,
                    'position': traci.vehicle.getLanePosition(veh_id)
                }
            except:
                continue

        # 全局统计
        global_stats = self._compute_global_stats(vehicle_states)

        observation = {
            'vehicle_states': vehicle_states,
            'vehicle_ids': vehicle_ids,
            'icv_ids': icv_ids,
            'global_stats': global_stats,
            'step': self.current_step
        }

        return observation

    def _compute_global_stats(self, vehicle_states: Dict[str, Dict]) -> np.ndarray:
        """计算全局统计特征（16维）"""
        if not vehicle_states:
            return np.zeros(16)

        speeds = [v['speed'] for v in vehicle_states.values()]
        accelerations = [v['acceleration'] for v in vehicle_states.values()]

        stats = np.zeros(16)

        # 速度统计
        stats[0] = np.mean(speeds) if speeds else 0.0
        stats[1] = np.std(speeds) if len(speeds) > 1 else 0.0
        stats[2] = np.max(speeds) if speeds else 0.0
        stats[3] = np.min(speeds) if speeds else 0.0

        # 加速度统计
        stats[4] = np.mean(accelerations) if accelerations else 0.0
        stats[5] = np.std(accelerations) if len(accelerations) > 1 else 0.0

        # 车辆数
        stats[6] = len(vehicle_states)

        # 时间信息
        stats[7] = self.current_step * self.step_length

        # 车道分布
        lanes = [v['lane_index'] for v in vehicle_states.values()]
        stats[8] = np.mean(lanes) if lanes else 0.0

        # 碰撞检测
        stats[9] = traci.simulation.getCollidingVehiclesNumber()

        # 已到达/已出发
        stats[10] = len(self.stats['arrived_vehicles'])
        stats[11] = len(self.stats['departed_vehicles'])

        # 剩余车辆
        stats[12] = traci.simulation.getMinExpectedNumber()

        # 平均车头时距
        stats[13] = 0.0  # 简化

        # 网络负载
        stats[14] = stats[6] / max(stats[12] + stats[6], 1)

        # 仿真进度
        stats[15] = self.current_step / self.config.get('max_steps', 36000)

        return stats

    def _compute_reward(self, observation: Dict[str, Any]) -> float:
        """计算奖励"""
        # 简化版奖励：基于速度和流量
        vehicle_states = observation['vehicle_states']

        if not vehicle_states:
            return 0.0

        speeds = [v['speed'] for v in vehicle_states.values()]
        avg_speed = np.mean(speeds)

        # 奖励：平均速度
        reward = avg_speed / 30.0  # 归一化

        # 惩罚：速度标准差（稳定性）
        if len(speeds) > 1:
            speed_std = np.std(speeds)
            reward -= 0.1 * speed_std / 10.0

        return float(reward)

    def _is_done(self) -> bool:
        """检查是否结束"""
        if self.current_step >= self.config.get('max_steps', 36000):
            return True

        if traci.simulation.getMinExpectedNumber() <= 0 and self.current_step > 1000:
            return True

        return False

    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        info = {
            'step': self.current_step,
            'vehicles': traci.vehicle.getIDCount(),
            'collisions': traci.simulation.getCollidingVehiclesNumber(),
            'arrived': traci.simulation.getArrivedNumber(),
            'departed': traci.simulation.getDepartedNumber()
        }

        return info

    def __enter__(self):
        """上下文管理器入口"""
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
