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
import logging

# 配置日志
logger = logging.getLogger(__name__)

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
        use_gui: bool = False,
        port: Optional[int] = None,
        disable_port_retry: bool = False  # 禁用端口重试（用于并行环境）
    ):
        self.config = config
        self.use_gui = use_gui
        self.port = port if port is not None else 8813  # 默认端口
        self.disable_port_retry = disable_port_retry  # 是否禁用端口重试

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
        import platform
        is_windows = platform.system() == "Windows"

        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        if is_windows:
            sumo_binary += ".exe"

        # 如果不在PATH中，尝试使用绝对路径
        if not os.path.exists(sumo_binary):
            if is_windows:
                # Windows路径
                possible_paths = [
                    r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe",
                    r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe",
                ]
            else:
                # Linux/WSL路径
                possible_paths = [
                    "/usr/bin/sumo",
                    "/usr/local/bin/sumo",
                    os.path.expanduser("~/sumo/bin/sumo"),
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
            # 注意：不要添加--remote-port参数，traci.start()会自动处理
        ]

        # 只有在配置中明确指定了seed才添加
        if 'seed' in self.config:
            cmd.extend(["--seed", str(self.config['seed'])])

        return cmd

    def start(self):
        """启动SUMO仿真"""
        if not TRACI_AVAILABLE:
            raise RuntimeError("TraCI未安装，无法启动SUMO")

        if self.is_connected:
            return

        # 尝试连接
        if self.disable_port_retry:
            # 并行模式：不重试，直接使用分配的端口
            try:
                traci.start(self.sumo_cmd, port=self.port)
                self.is_connected = True
                self.current_step = 0
                print(f"✅ SUMO已启动 (GUI: {self.use_gui}, Port: {self.port})")
            except Exception as e:
                raise RuntimeError(f"SUMO启动失败 (Port: {self.port}): {e}")
        else:
            # 单机模式：尝试连接，最多重试3次
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    traci.start(self.sumo_cmd, port=self.port)
                    self.is_connected = True
                    self.current_step = 0
                    print(f"✅ SUMO已启动 (GUI: {self.use_gui}, Port: {self.port})")
                    return
                except Exception as e:
                    if attempt < max_attempts - 1:
                        print(f"⚠️  端口 {self.port} 占用，尝试新端口...")
                        # 尝试新端口
                        self.port += 10
                        self.sumo_cmd = self._build_sumo_command()
                    else:
                        raise RuntimeError(f"SUMO启动失败（已尝试 {max_attempts} 次）: {e}")

    def close(self):
        """关闭SUMO仿真"""
        if self.is_connected and TRACI_AVAILABLE:
            try:
                # 使用 close(wait=False) 避免阻塞
                # 如果不支持 wait 参数，则使用强制关闭
                try:
                    traci.close(wait=False)
                except TypeError:
                    # 旧版本 TraCI 不支持 wait 参数，强制关闭连接
                    traci.close()
            except Exception as e:
                # 忽略关闭错误，确保进程退出
                pass

            # 强制终止 SUMO 进程（防止僵尸进程）
            try:
                import signal
                import psutil
                # 查找并终止 SUMO 进程
                current_process = psutil.Process()
                for child in current_process.children(recursive=True):
                    try:
                        if 'sumo' in child.name().lower():
                            child.terminate()
                    except:
                        pass
            except ImportError:
                # psutil 不可用时，使用 os.system 强制终止
                os.system("pkill -9 sumo 2>/dev/null || true")
            except Exception:
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

        # 更新到达车辆统计
        if TRACI_AVAILABLE:
            try:
                newly_arrived = traci.simulation.getArrivedIDList()
                self.stats['arrived_vehicles'].update(newly_arrived)
            except Exception:
                pass

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

                # 应用换道（安全版本）
                if lane_change > 0.5:
                    try:
                        current_lane = traci.vehicle.getLaneIndex(veh_id)
                        road_id = traci.vehicle.getRoadID(veh_id)

                        # 获取道路的车道数量
                        lane_count = traci.edge.getLaneNumber(road_id)

                        # 检查可以换到哪个车道
                        possible_directions = []
                        if current_lane > 0:
                            possible_directions.append(-1)  # 可以向左
                        if current_lane < lane_count - 1:
                            possible_directions.append(1)   # 可以向右

                        if possible_directions:
                            # 随机选择一个有效的换道方向
                            lane_change_direction = np.random.choice(possible_directions)
                            target_lane = current_lane + lane_change_direction

                            # 二次验证：确保目标车道在有效范围内
                            if 0 <= target_lane < lane_count:
                                traci.vehicle.changeLane(
                                    veh_id,
                                    target_lane,
                                    2.0  # 持续时间
                                )
                    except Exception as lane_error:
                        # 换道失败，记录但不中断训练（可能是车辆离开路网等）
                        logger.debug(f"车辆 {veh_id} 换道失败: {lane_error}")

            except Exception as e:
                # 车辆控制失败，记录但不影响其他车辆
                logger.debug(f"车辆 {veh_id} 控制失败: {e}")

    def _get_observation(self) -> Dict[str, Any]:
        """获取当前观测"""
        # 获取所有车辆
        all_vehicle_ids = traci.vehicle.getIDList()

        vehicle_states = {}
        valid_vehicle_ids = []  # 只保留成功获取状态的车辆
        icv_ids = set()

        # 根据配置选择ICV（从所有车辆中选择）
        control_ratio = self.config.get('control_ratio', 0.25)
        num_icv = max(1, int(len(all_vehicle_ids) * control_ratio))

        if len(all_vehicle_ids) > 0:
            icv_indices = np.random.choice(
                len(all_vehicle_ids),
                size=min(num_icv, len(all_vehicle_ids)),
                replace=False
            )
            icv_ids = {all_vehicle_ids[i] for i in icv_indices}

        # 收集车辆状态（只保留成功的）
        for veh_id in all_vehicle_ids:
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

                # 添加到有效车辆列表
                valid_vehicle_ids.append(veh_id)

            except Exception as e:
                # 获取状态失败，跳过此车辆
                continue

        # 全局统计
        global_stats = self._compute_global_stats(vehicle_states)

        observation = {
            'vehicle_states': vehicle_states,
            'vehicle_ids': valid_vehicle_ids,  # 使用有效车辆列表
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

        # 平均车头时距（完整实现）
        # 计算所有车辆与前车之间的平均时间距离
        time_headways = []
        for veh_id in traci.vehicle.getIDList():
            try:
                leader_info = traci.vehicle.getLeader(veh_id, 100.0)  # 100米范围内
                if leader_info:
                    # getLeader返回 (leader_id, distance) 元组
                    leader_id = leader_info[0]  # 提取vehicle_id
                    distance = leader_info[1]    # 提取distance

                    # 获取速度
                    leader_speed = traci.vehicle.getSpeed(leader_id)
                    ego_speed = traci.vehicle.getSpeed(veh_id)

                    # 避免除零
                    if ego_speed > 0.1:
                        # 车头时距 = 距离 / 速度
                        thw = distance / ego_speed
                        time_headways.append(thw)
            except:
                continue

        stats[13] = np.mean(time_headways) if time_headways else 2.0  # 默认2秒

        # 网络负载
        stats[14] = stats[6] / max(stats[12] + stats[6], 1)

        # 仿真进度
        stats[15] = self.current_step / self.config.get('max_steps', 36000)

        return stats

    def _compute_reward(self, observation: Dict[str, Any]) -> float:
        """
        计算奖励 - 完整实现

        综合考虑多个指标：
        1. 平均速度（交通效率）
        2. 速度标准差（交通稳定性）
        3. OD完成率（到达目的地的车辆比例）
        4. 碰撞惩罚（安全性）
        5. 停车次数（流畅性）
        """
        vehicle_states = observation['vehicle_states']

        if not vehicle_states:
            return 0.0

        # ========== 1. 交通效率奖励 ==========
        speeds = [v['speed'] for v in vehicle_states.values()]
        avg_speed = np.mean(speeds)

        # 速度奖励：鼓励高平均速度（归一化到0-30m/s）
        speed_reward = avg_speed / 30.0

        # ========== 2. 交通稳定性奖励 ==========
        # 速度标准差越小越好（交通流更稳定）
        if len(speeds) > 1:
            speed_std = np.std(speeds)
            stability_reward = -0.1 * (speed_std / 10.0)  # 惩罚速度波动
        else:
            stability_reward = 0.0

        # ========== 3. OD完成率奖励 ==========
        # 统计到达的车辆数和总出发数
        arrived_count = len(self.stats.get('arrived_vehicles', []))
        departed_count = len(self.stats.get('departed_vehicles', []))

        if departed_count > 0:
            completion_rate = arrived_count / departed_count
            # 完成率越高越好
            completion_reward = 10.0 * completion_rate  # 较大权重
        else:
            completion_reward = 0.0

        # ========== 4. 停车惩罚 ==========
        # 统计停车（速度接近0）的车辆比例
        stopped_vehicles = sum(1 for s in speeds if s < 0.1)
        stopped_ratio = stopped_vehicles / len(speeds)
        stopped_penalty = -0.5 * stopped_ratio  # 惩罚停车

        # ========== 5. 碰撞/紧急刹车惩罚 ==========
        # 统计急减速的车辆（加速度 < -3 m/s²）
        emergency_braking = 0
        for veh_id in vehicle_states.keys():
            try:
                accel = traci.vehicle.getDecel(veh_id)
                if accel > 3.0:  # 减速度>3m/s²认为是急刹车
                    emergency_braking += 1
            except:
                continue

        emergency_penalty = -0.2 * (emergency_braking / max(len(speeds), 1))

        # ========== 总奖励 ==========
        # 权重设置：
        # - 速度效率: 1.0
        # - 稳定性: 0.1
        # - 完成率: 10.0（最重要）
        # - 停车: 0.5
        # - 急刹车: 0.2
        total_reward = (
            1.0 * speed_reward +
            stability_reward +
            completion_reward +
            stopped_penalty +
            emergency_penalty
        )

        return float(total_reward)

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
            'arrived': list(self.stats.get('arrived_vehicles', set())),
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
