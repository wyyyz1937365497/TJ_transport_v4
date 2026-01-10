"""SUMO环境接口 - 简化且高效"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False

from ..utils.dataclass import Observation, VehicleState, Action, StepResult
from ..utils.logging import get_logger
from ..utils.sumo_port_manager import find_free_port, check_port_in_use

logger = get_logger()

# 全局端口分配器（用于多进程）
_next_port = 8813
_port_lock = None


class SumoEnvironment:
    """
    SUMO仿真环境

    简化版环境接口，专注于核心功能
    """

    def __init__(
        self,
        config: Dict[str, Any],
        use_gui: bool = False,
        port: Optional[int] = None,
    ):
        self.config = config
        self.use_gui = use_gui

        # 路径
        self.sumo_cfg = config.get("sumo_cfg", "")
        self.net_file = config.get("net_file", "")
        self.route_file = config.get("route_file", "")

        # 参数
        self.step_length = config.get("step_length", 0.1)
        self.max_steps = config.get("max_steps", 3600)
        self.control_ratio = config.get("control_ratio", 0.25)

        # 状态
        self.is_connected = False
        self.current_step = 0

        # 端口
        self.port = port if port is not None else find_free_port()

        # 统计
        self.stats = {
            "departed": set(),
            "arrived": set(),
        }

        # 构建命令
        self.sumo_cmd = self._build_sumo_command()

    def _build_sumo_command(self) -> List[str]:
        """构建SUMO命令"""
        # WSL/Linux使用sumo而不是sumo.exe
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"

        cmd = [
            sumo_binary,
            "-c", self.sumo_cfg,
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--step-length", str(self.step_length),
            "--remote-port", str(self.port),  # 指定端口
        ]

        if "seed" in self.config:
            cmd.extend(["--seed", str(self.config["seed"])])

        return cmd

    def reset(self) -> Observation:
        """重置环境"""
        if self.is_connected:
            self.close()

        self._start()
        self.current_step = 0
        self.stats = {"departed": set(), "arrived": set()}

        return self._get_observation()

    def step(self, actions: Optional[Dict[str, np.ndarray]] = None) -> StepResult:
        """执行一步"""
        if not self.is_connected:
            raise RuntimeError("SUMO未连接，请先调用reset()")

        # 应用动作
        if actions:
            self._apply_actions(actions)

        # 推进仿真
        traci.simulationStep()
        self.current_step += 1

        # 更新统计
        self._update_stats()

        # 获取观测
        observation = self._get_observation()

        # 计算奖励
        reward = self._compute_reward(observation)

        # 检查结束
        done = self._is_done()

        # 额外信息
        info = self._get_info()

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )

    def _start(self):
        """启动SUMO"""
        if not TRACI_AVAILABLE:
            raise RuntimeError("TraCI未安装")

        # 尝试连接，最多重试3次
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # 使用指定端口启动
                traci.start(self.sumo_cmd, port=self.port)
                self.is_connected = True
                logger.info(f"SUMO已启动 (GUI: {self.use_gui}, Port: {self.port})")
                return
            except Exception as e:
                logger.warning(f"启动SUMO失败 (尝试 {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    # 端口冲突，尝试新端口
                    self.port = find_free_port(self.port + 1)
                    self.sumo_cmd = self._build_sumo_command()
                    logger.info(f"尝试使用新端口: {self.port}")
                    import time
                    time.sleep(0.5)
                else:
                    raise RuntimeError(f"SUMO启动失败（已尝试 {max_attempts} 次）: {e}")

    def close(self):
        """关闭SUMO"""
        if self.is_connected and TRACI_AVAILABLE:
            try:
                traci.close()
            except:
                pass
            self.is_connected = False
            logger.info("SUMO已关闭")

    def _apply_actions(self, actions: Dict[str, np.ndarray]):
        """应用控制动作"""
        for veh_id, action in actions.items():
            try:
                if veh_id not in traci.vehicle.getIDList():
                    continue

                acceleration = float(action[0])
                lane_change = float(action[1])

                # 应用加速度
                current_speed = traci.vehicle.getSpeed(veh_id)
                target_speed = max(0, current_speed + acceleration * self.step_length)
                traci.vehicle.setSpeed(veh_id, target_speed)

                # 应用换道
                if lane_change > 0.5:
                    self._try_lane_change(veh_id)

            except Exception:
                continue

    def _try_lane_change(self, veh_id: str):
        """尝试换道"""
        try:
            current_lane = traci.vehicle.getLaneIndex(veh_id)
            road_id = traci.vehicle.getRoadID(veh_id)
            lane_count = traci.edge.getLaneNumber(road_id)

            # 可以换道的方向
            directions = []
            if current_lane > 0:
                directions.append(-1)
            if current_lane < lane_count - 1:
                directions.append(1)

            if directions:
                direction = random.choice(directions)
                traci.vehicle.changeLane(
                    veh_id,
                    current_lane + direction,
                    2.0,
                )
        except Exception:
            pass

    def _get_observation(self) -> Observation:
        """获取观测"""
        all_vehicle_ids = traci.vehicle.getIDList()

        vehicle_states = {}
        valid_ids = []

        # 选择ICV
        num_icv = max(1, int(len(all_vehicle_ids) * self.control_ratio))
        icv_ids = (
            set(random.sample(all_vehicle_ids, num_icv))
            if len(all_vehicle_ids) >= num_icv
            else set(all_vehicle_ids)
        )

        # 收集车辆状态
        for veh_id in all_vehicle_ids:
            try:
                position = traci.vehicle.getPosition(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                angle = traci.vehicle.getAngle(veh_id)
                lane_id = traci.vehicle.getLaneID(veh_id)
                lane_index = traci.vehicle.getLaneIndex(veh_id)
                acceleration = traci.vehicle.getAcceleration(veh_id)

                vehicle_states[veh_id] = VehicleState(
                    id=veh_id,
                    x=position[0],
                    y=position[1],
                    speed=speed,
                    acceleration=acceleration,
                    lane_id=lane_id,
                    lane_index=lane_index,
                    position=traci.vehicle.getLanePosition(veh_id),
                    angle=angle,
                )
                valid_ids.append(veh_id)
            except Exception:
                continue

        # 全局统计
        global_stats = self._compute_global_stats(vehicle_states)

        return Observation(
            vehicle_states=vehicle_states,
            vehicle_ids=valid_ids,
            icv_ids=list(icv_ids),
            global_stats=torch.tensor(global_stats, dtype=torch.float32),
            step=self.current_step,
        )

    def _compute_global_stats(self, vehicle_states: Dict[str, VehicleState]) -> List[float]:
        """计算全局统计特征（16维）"""
        if not vehicle_states:
            return [0.0] * 16

        speeds = [v.speed for v in vehicle_states.values()]
        accels = [v.acceleration for v in vehicle_states.values()]

        stats = [0.0] * 16

        # 速度统计
        stats[0] = float(np.mean(speeds))
        stats[1] = float(np.std(speeds) if len(speeds) > 1 else 0.0)
        stats[2] = float(np.max(speeds))
        stats[3] = float(np.min(speeds))

        # 加速度统计
        stats[4] = float(np.mean(accels))
        stats[5] = float(np.std(accels) if len(accels) > 1 else 0.0)

        # 车辆数
        stats[6] = float(len(vehicle_states))

        # 时间
        stats[7] = float(self.current_step * self.step_length)

        # 车道分布
        lanes = [v.lane_index for v in vehicle_states.values()]
        stats[8] = float(np.mean(lanes) if lanes else 0.0)

        # 碰撞
        stats[9] = float(traci.simulation.getCollidingVehiclesNumber())

        # 已到达/已出发
        stats[10] = float(len(self.stats["arrived"]))
        stats[11] = float(len(self.stats["departed"]))

        # 剩余
        stats[12] = float(traci.simulation.getMinExpectedNumber())

        # 车头时距
        thws = []
        for veh_id in traci.vehicle.getIDList():
            try:
                leader = traci.vehicle.getLeader(veh_id, 100.0)
                if leader:
                    distance = leader[1]
                    ego_speed = traci.vehicle.getSpeed(veh_id)
                    if ego_speed > 0.1:
                        thws.append(distance / ego_speed)
            except Exception:
                continue
        stats[13] = float(np.mean(thws) if thws else 2.0)

        # 负载
        stats[14] = stats[6] / max(stats[12] + stats[6], 1)

        # 进度
        stats[15] = self.current_step / self.max_steps

        return stats

    def _compute_reward(self, observation: Observation) -> float:
        """计算奖励"""
        if not observation.vehicle_states:
            return 0.0

        states = observation.vehicle_states
        speeds = [v.speed for v in states.values()]

        # 速度奖励
        speed_reward = float(np.mean(speeds) / 30.0)

        # 稳定性奖励
        if len(speeds) > 1:
            stability_reward = -0.1 * float(np.std(speeds) / 10.0)
        else:
            stability_reward = 0.0

        # 完成率奖励
        if self.stats["departed"]:
            completion_reward = 10.0 * len(self.stats["arrived"]) / len(self.stats["departed"])
        else:
            completion_reward = 0.0

        # 停车惩罚
        stopped = sum(1 for s in speeds if s < 0.1)
        stopped_penalty = -0.5 * stopped / max(len(speeds), 1)

        return speed_reward + stability_reward + completion_reward + stopped_penalty

    def _is_done(self) -> bool:
        """检查是否结束"""
        if self.current_step >= self.max_steps:
            return True
        if traci.simulation.getMinExpectedNumber() <= 0 and self.current_step > 1000:
            return True
        return False

    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        return {
            "step": self.current_step,
            "vehicles": traci.vehicle.getIDCount(),
            "collisions": traci.simulation.getCollidingVehiclesNumber(),
            "arrived": traci.simulation.getArrivedNumber(),
            "departed": traci.simulation.getDepartedNumber(),
        }

    def _update_stats(self):
        """更新统计信息"""
        departed = set(traci.simulation.getDepartedIDList())
        arrived = set(traci.simulation.getArrivedIDList())

        self.stats["departed"].update(departed)
        self.stats["arrived"].update(arrived)

    def __enter__(self):
        """上下文管理器"""
        self.reset()
        return self

    def __exit__(self, *args):
        """上下文管理器"""
        self.close()
