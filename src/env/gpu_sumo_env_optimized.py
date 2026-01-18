"""
SUMO订阅机制优化版本 - 使用批处理API加速

关键优化：
1. 使用traci.vehicle.subscribe批量获取单个车辆数据
2. 使用traci.junction.subscribeContext批量获取所有车辆数据
3. 减少API调用次数：160次 → 1次（每步）

性能提升预期：
- 数据收集速度：3-5倍提升
- 总体训练速度：2-3倍提升
"""

import os
import numpy as np
from typing import Dict, List, Optional, Any
import logging

# GPU和深度学习
import torch

# 尝试导入Libsumo
try:
    import libsumo as traci
    LIBSUMO_AVAILABLE = True
except ImportError:
    try:
        import traci
        LIBSUMO_AVAILABLE = False
    except ImportError:
        traci = None
        LIBSUMO_AVAILABLE = False

# TraCI常量
try:
    import traci.constants as tc
except ImportError:
    # 定义常量（如果traci.constants不可用）
    class tc:
        VAR_POSITION = 0x40
        VAR_SPEED = 0x40
        VAR_LANE_ID = 0x31
        VAR_LANE_INDEX = 0x33
        VAR_LANEPOSITION = 0x34
        VAR_ROAD_ID = 0x31
        VAR_ANGLE = 0x41
        VAR_ACCELERATION = 0x47
        CMD_GET_VEHICLE_VARIABLE = 0xa4
        VAR_WAITING_TIME = 0x7f

logger = logging.getLogger(__name__)


class GPUSumoEnvironmentOptimized:
    """
    GPU加速+订阅机制优化的SUMO环境

    核心优化：
    1. 使用订阅机制批量获取数据
    2. GPU加速矩阵运算
    3. 减少API调用次数
    """

    def __init__(
        self,
        config: Dict[str, Any],
        use_gui: bool = False,
        device: str = 'cuda',
        use_subscription: bool = True  # 是否启用订阅优化
    ):
        self.config = config
        self.use_gui = use_gui

        # 确保使用单GPU（cuda:0）以避免多GPU通信开销
        if torch.cuda.is_available():
            if device == 'cuda':
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')

        self.use_subscription = use_subscription and LIBSUMO_AVAILABLE

        # SUMO配置
        self.sumo_cfg = config.get('sumo_config', config.get('sumo_cfg', ''))
        self.step_length = config.get('step_length', 0.1)

        # 构建命令
        self.sumo_cmd = self._build_sumo_command()

        # 状态
        self.is_connected = False
        self.current_step = 0

        # 统计信息
        self.stats = {
            'arrived_vehicles': set(),
            'departed_vehicles': set(),
            'started_vehicles': set(),
            'ended_vehicles': set()
        }

        # 订阅相关
        self._subscription_enabled = False
        self._junction_id = None

        # 缓存（GPU tensors）
        self._vehicle_states_cache = None
        self._global_stats_cache = None

        # 性能统计
        self._api_call_count = 0
        self._subscription_call_count = 0

    def _build_sumo_command(self):
        """构建SUMO命令"""
        if self.use_gui:
            # GUI模式：使用sumo-gui
            sumo_binary = "sumo-gui"
        else:
            # 非GUI模式：libsumo直接调用或使用sumo命令行
            sumo_binary = "sumo"  # libsumo会自动处理

        cmd = [
            sumo_binary,
            "-c", self.sumo_cfg,
            "--no-step-log",  # 禁用步骤日志
            "--no-warnings",   # 禁用警告
        ]

        # 如果使用libsumo，添加额外优化选项
        if LIBSUMO_AVAILABLE:
            cmd.extend([
                "--collision.check-junctions", "false",  # 禁用路口碰撞检测（加速）
                "--collision.action", "warn",            # 碰撞时仅警告
            ])

        return cmd

    def start(self):
        """启动SUMO仿真"""
        if not LIBSUMO_AVAILABLE and traci is None:
            raise RuntimeError("TraCI/Libsumo未安装！")

        try:
            traci.start(self.sumo_cmd)
            self.is_connected = True

            # 启用订阅优化
            if self.use_subscription:
                self._enable_subscriptions()

            print(f"[OK] SUMO已启动 (订阅优化: {'启用' if self._subscription_enabled else '禁用'})")
        except Exception as e:
            raise RuntimeError(f"SUMO启动失败: {e}")

    def _enable_subscriptions(self):
        """启用批量订阅"""
        try:
            # 获取一个junction作为订阅锚点
            junction_ids = traci.junction.getIDList()
            if not junction_ids:
                logger.warning("没有找到junction，订阅优化不可用")
                return

            # 使用第一个junction
            self._junction_id = junction_ids[0]

            # 订阅大范围内的所有车辆（覆盖整个场景）
            # 1000000米 = 1000km，足够覆盖整个场景
            traci.junction.subscribeContext(
                self._junction_id,
                tc.CMD_GET_VEHICLE_VARIABLE,
                1000000,  # 大范围
                [
                    tc.VAR_POSITION,
                    tc.VAR_SPEED,
                    tc.VAR_LANE_ID,
                    tc.VAR_LANE_INDEX,
                    tc.VAR_LANEPOSITION,
                    tc.VAR_ROAD_ID,
                    tc.VAR_ANGLE,
                    tc.VAR_ACCELERATION,
                ]
            )

            self._subscription_enabled = True
            print(f"[OK] 订阅优化已启用 (junction: {self._junction_id})")

        except Exception as e:
            logger.warning(f"订阅启用失败: {e}，将使用普通API")
            self._subscription_enabled = False

    def close(self):
        """关闭SUMO仿真"""
        if self.is_connected and traci is not None:
            try:
                traci.close(wait=False)
            except:
                pass
            self.is_connected = False
            print(f"[OK] SUMO已关闭 (API调用: {self._api_call_count}, 订阅调用: {self._subscription_call_count})")

    def reset(self) -> Dict[str, Any]:
        """重置环境"""
        if self.is_connected:
            self.close()

        self.start()
        self.current_step = 0

        # 重置统计信息
        self.stats = {
            'arrived_vehicles': set(),
            'departed_vehicles': set(),
            'started_vehicles': set(),
            'ended_vehicles': set()
        }

        # 优先使用子类的_get_observation()（如果被覆盖），否则使用优化版本
        if hasattr(self, '_get_observation') and '_get_observation' in type(self).__dict__:
            # 子类覆盖了_get_observation()，使用它
            return self._get_observation()
        else:
            # 使用订阅优化版本
            return self._get_observation_optimized()

    def step(self, actions: Optional[Dict[str, torch.Tensor]] = None) -> tuple:
        """
        执行一步仿真（优化版）

        Args:
            actions: {vehicle_id: [acceleration, lane_change]}

        Returns:
            observation, reward, done, info
        """
        if not self.is_connected:
            raise RuntimeError("SUMO未连接，请先调用reset()")

        # 应用控制动作
        if actions:
            self._apply_actions_gpu(actions)

        # 推进仿真
        traci.simulationStep()
        self.current_step += 1

        # 更新到达/出发车辆统计
        try:
            newly_arrived = traci.simulation.getArrivedIDList()
            self.stats['arrived_vehicles'].update(newly_arrived)
        except Exception:
            pass

        try:
            newly_departed = traci.simulation.getDepartedIDList()
            self.stats['departed_vehicles'].update(newly_departed)
        except Exception:
            pass

        # 获取观测（优先使用子类覆盖的方法，否则使用订阅优化）
        if hasattr(self, '_get_observation') and '_get_observation' in type(self).__dict__:
            # 子类覆盖了_get_observation()，使用它
            observation = self._get_observation()
        else:
            # 使用订阅优化版本
            observation = self._get_observation_optimized()

        # 计算奖励
        reward = self._compute_reward_gpu(observation)

        # 检查是否结束
        done = self._is_done()

        # 额外信息
        info = self._get_info()

        return observation, reward, done, info

    def _get_observation_optimized(self) -> Dict[str, Any]:
        """
        获取观测（订阅优化版）

        优化前：32辆车 × 5次查询 = 160次API调用
        优化后：1次getContextSubscriptionResults调用
        """
        if not self._subscription_enabled:
            # 回退到原始方法
            return self._get_observation_fallback()

        try:
            # 使用订阅批量获取所有车辆数据（1次API调用）
            self._subscription_call_count += 1
            all_results = traci.junction.getContextSubscriptionResults(self._junction_id)

            if not all_results:
                # 没有车辆
                return self._create_empty_observation()

            # 解析订阅结果
            vehicle_states = {}
            for veh_id, vars_dict in all_results.items():
                vehicle_states[veh_id] = {
                    'position': vars_dict.get(tc.VAR_POSITION, (0, 0)),
                    'speed': vars_dict.get(tc.VAR_SPEED, 0.0),
                    'lane_id': vars_dict.get(tc.VAR_LANE_ID, ""),
                    'lane_index': vars_dict.get(tc.VAR_LANE_INDEX, 0),
                    'lane_position': vars_dict.get(tc.VAR_LANEPOSITION, 0.0),
                    'road_id': vars_dict.get(tc.VAR_ROAD_ID, ""),
                    'angle': vars_dict.get(tc.VAR_ANGLE, 0.0),
                    'acceleration': vars_dict.get(tc.VAR_ACCELERATION, 0.0),
                }

            # 转换为GPU tensor
            return self._convert_to_gpu_tensors(vehicle_states)

        except Exception as e:
            logger.warning(f"订阅获取失败: {e}，回退到普通API")
            return self._get_observation_fallback()

    def _get_observation_fallback(self) -> Dict[str, Any]:
        """回退方法：使用普通API获取观测"""
        try:
            vehicle_ids = traci.vehicle.getIDList()
            vehicle_states = {}

            self._api_call_count += len(vehicle_ids)

            for veh_id in vehicle_ids:
                vehicle_states[veh_id] = {
                    'position': traci.vehicle.getPosition(veh_id),
                    'speed': traci.vehicle.getSpeed(veh_id),
                    'lane_id': traci.vehicle.getLaneID(veh_id),
                    'lane_index': traci.vehicle.getLaneIndex(veh_id),
                    'lane_position': traci.vehicle.getLanePosition(veh_id),
                    'road_id': traci.vehicle.getRoadID(veh_id),
                    'angle': traci.vehicle.getAngle(veh_id),
                }

            return self._convert_to_gpu_tensors(vehicle_states)

        except Exception as e:
            logger.error(f"获取观测失败: {e}")
            return self._create_empty_observation()

    def _convert_to_gpu_tensors(self, vehicle_states: Dict[str, Dict]) -> Dict[str, Any]:
        """
        将车辆状态转换为GPU tensors

        为了保持与CompetitionSumoEnv的兼容性，返回dict格式的vehicle_states，
        而不是单一的GPU tensor。这样子类可以正常覆盖_get_observation()。
        """
        if not vehicle_states:
            return self._create_empty_observation()

        # 保持原始dict格式，不转换为tensor
        # 这样CompetitionSumoEnv可以正常覆盖处理
        return {
            'vehicle_states': vehicle_states,  # 保持dict格式
            'vehicle_ids': list(vehicle_states.keys()),
            'num_vehicles': len(vehicle_states),
        }

    def _create_empty_observation(self) -> Dict[str, Any]:
        """创建空观测"""
        return {
            'vehicle_ids': [],
            'num_vehicles': 0,
            'vehicle_states': {}  # 保持dict格式
        }

    def _apply_actions_gpu(self, actions: Dict[str, torch.Tensor]):
        """应用控制动作"""
        if not actions:
            return

        vehicle_ids = list(traci.vehicle.getIDList())
        valid_ids = [vid for vid in vehicle_ids if vid in actions]

        if not valid_ids:
            return

        # 批量获取当前速度（1次API调用）
        self._api_call_count += len(valid_ids)
        current_speeds = {vid: traci.vehicle.getSpeed(vid) for vid in valid_ids}

        # 批量应用控制
        for veh_id in valid_ids:
            try:
                action = actions[veh_id]

                if torch.is_tensor(action):
                    action = action.cpu().numpy()

                acceleration = action[0]
                lane_change = action[1]

                # 计算目标速度
                current_speed = current_speeds[veh_id]
                target_speed = max(0, current_speed + acceleration * self.step_length)

                # 应用加速度
                traci.vehicle.setSpeed(veh_id, target_speed)

                # 应用换道
                if lane_change > 0.5:
                    self._safe_change_lane(veh_id)

            except Exception as e:
                logger.debug(f"车辆 {veh_id} 控制失败: {e}")

    def _safe_change_lane(self, veh_id: str):
        """安全换道（简化版）"""
        try:
            current_lane = traci.vehicle.getLaneIndex(veh_id)
            road_id = traci.vehicle.getRoadID(veh_id)
            lane_count = traci.edge.getLaneNumber(road_id)

            # 简单策略：优先向右换道
            if current_lane < lane_count - 1:
                traci.vehicle.changeLane(veh_id, current_lane + 1, 5.0)
            elif current_lane > 0:
                traci.vehicle.changeLane(veh_id, current_lane - 1, 5.0)

        except Exception as e:
            logger.debug(f"车辆 {veh_id} 换道失败: {e}")

    def _compute_reward_gpu(self, observation: Dict[str, Any]) -> float:
        """计算奖励（简化版）"""
        num_vehicles = observation.get('num_vehicles', 0)
        if num_vehicles == 0:
            return 0.0

        vehicle_states = observation.get('vehicle_states', {})

        # 计算平均速度（从dict中提取）
        total_speed = 0.0
        count = 0
        for state in vehicle_states.values():
            if isinstance(state, dict):
                speed = state.get('speed', 0.0)
                total_speed += speed
                count += 1

        avg_speed = total_speed / count if count > 0 else 0.0
        return avg_speed

    def _is_done(self) -> bool:
        """检查episode是否结束"""
        try:
            max_steps = self.config.get('max_steps', 3600)
            if self.current_step >= max_steps:
                return True

            # 检查是否还有车辆
            min_expected = traci.simulation.getMinExpectedNumber()
            return min_expected == 0

        except Exception:
            return False

    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        return {
            'current_step': self.current_step,
            'arrived_count': len(self.stats['arrived_vehicles']),
            'departed_count': len(self.stats['departed_vehicles']),
        }


# 使用说明
"""
使用优化环境：

from src.env.gpu_sumo_env_optimized import GPUSumoEnvironmentOptimized

env = GPUSumoEnvironmentOptimized(
    config=config,
    device='cuda',
    use_subscription=True  # 启用订阅优化
)

obs = env.reset()
for _ in range(100):
    obs, reward, done, info = env.step(actions)

env.close()

预期效果：
- API调用次数：160次/步 → 1次/步
- 速度提升：3-5倍
"""
