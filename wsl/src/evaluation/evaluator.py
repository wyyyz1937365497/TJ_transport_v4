"""
评估器模块 - 完整实现
用于评估训练好的模型性能
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..models import TrafficController
from ..env import SumoEnvironment
from ..utils import Config, get_logger
from ..utils.dataclass import Observation
from .metrics import compute_metrics

logger = get_logger()


class Evaluator:
    """
    模型评估器 - 完整实现

    评估指标：
    1. OD完成率 (OCR): 车辆按时到达目的地的比例
    2. 速度标准差 (σv): 交通流稳定性指标
    3. 平均绝对加速度 (|a|avg): 舒适度指标
    4. 干预成本: 控制指令的总成本
    5. 平均速度: 交通效率指标
    6. 行程时间: 车辆平均行程时间
    7. 碰撞次数: 安全性指标
    """

    def __init__(
        self,
        model: TrafficController,
        config: Config,
    ):
        self.model = model
        self.config = config

        # 设备
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        # 评估配置
        eval_cfg = config.get("evaluation", {})
        self.num_episodes = eval_cfg.get("num_episodes", 5)

        # 统计存储
        self.all_trajectories = []
        self.all_metrics = []

    def evaluate(self, verbose: bool = True) -> Dict[str, Any]:
        """
        评估模型

        Args:
            verbose: 是否显示进度条

        Returns:
            评估指标字典
        """
        logger.info("=" * 70)
        logger.info("开始模型评估")
        logger.info(f"Episodes: {self.num_episodes}")
        logger.info("=" * 70)

        # 创建环境
        env = SumoEnvironment(
            self.config.environment,
            use_gui=False,
        )

        # 评估循环
        episode_metrics = []

        iterator = tqdm(range(self.num_episodes), desc="评估中") if verbose else range(self.num_episodes)

        for episode_idx in iterator:
            # 运行一个episode
            episode_data = self._run_episode(env, episode_idx)

            # 计算指标
            metrics = compute_metrics(episode_data)

            episode_metrics.append(metrics)
            self.all_trajectories.append(episode_data)

            if verbose:
                iterator.set_postfix({
                    "reward": f"{metrics['total_reward']:.2f}",
                    "cost": f"{metrics['avg_cost']:.4f}",
                    "speed": f"{metrics['avg_speed']:.2f}",
                })

                logger.info(
                    f"Episode {episode_idx + 1}/{self.num_episodes} | "
                    f"Reward: {metrics['total_reward']:.2f} | "
                    f"Cost: {metrics['avg_cost']:.4f} | "
                    f"Speed: {metrics['avg_speed']:.2f} m/s | "
                    f"OCR: {metrics['ocr']:.2%}"
                )

        env.close()

        # 汇总指标
        summary_metrics = self._aggregate_metrics(episode_metrics)

        logger.info("=" * 70)
        logger.info("评估完成")
        self._print_summary(summary_metrics)
        logger.info("=" * 70)

        return summary_metrics

    def _run_episode(self, env: SumoEnvironment, episode_idx: int) -> Dict[str, Any]:
        """
        运行一个评估episode

        Args:
            env: SUMO环境
            episode_idx: episode索引

        Returns:
            episode数据
        """
        # 重置环境
        observation = env.reset()
        done = False
        step = 0

        # 存储数据
        episode_data = {
            "episode_idx": episode_idx,
            "trajectories": {},  # 每个车辆的轨迹
            "actions": [],       # 每一步的动作
            "rewards": [],       # 每一步的奖励
            "costs": [],         # 每一步的成本
            "global_stats": [],  # 每一步的全局统计
        }

        # 记录车辆轨迹
        vehicle_trajectories = {}

        while not done and step < self.config.environment.get("max_steps", 3600):
            # 记录当前状态
            for veh_id, state in observation.vehicle_states.items():
                if veh_id not in vehicle_trajectories:
                    vehicle_trajectories[veh_id] = {
                        "id": veh_id,
                        "timestamps": [],
                        "positions": [],
                        "speeds": [],
                        "accelerations": [],
                        "lane_indices": [],
                        "actions": [],  # [acceleration, lane_change]
                    }

                vehicle_trajectories[veh_id]["timestamps"].append(step * self.config.environment.get("step_length", 0.1))
                vehicle_trajectories[veh_id]["positions"].append(state.position)
                vehicle_trajectories[veh_id]["speeds"].append(state.speed)
                vehicle_trajectories[veh_id]["accelerations"].append(state.acceleration)
                vehicle_trajectories[veh_id]["lane_indices"].append(state.lane_index)

            # 准备batch
            batch = self._prepare_batch(observation)

            # 获取动作
            with torch.no_grad():
                output = self.model(batch)

            # 转换为动作字典
            action_dict = self._convert_output_to_actions(output)

            # 记录动作
            for veh_id in observation.vehicle_ids:
                if veh_id in action_dict:
                    vehicle_trajectories[veh_id]["actions"].append(action_dict[veh_id].copy())
                else:
                    vehicle_trajectories[veh_id]["actions"].append([0.0, 0.0])

            # 记录全局统计
            episode_data["global_stats"].append(observation.global_stats.cpu().numpy().copy())

            # 计算成本
            cost = self._compute_cost(output, action_dict)
            episode_data["costs"].append(cost)

            # 执行动作
            next_observation, reward, done, info = env.step(action_dict)

            episode_data["rewards"].append(reward)

            observation = next_observation
            step += 1

        episode_data["trajectories"] = vehicle_trajectories

        return episode_data

    def _prepare_batch(self, observation: Observation) -> Dict[str, Any]:
        """准备训练batch"""
        vehicle_states = observation.vehicle_states
        vehicle_ids = observation.vehicle_ids
        icv_ids = observation.icv_ids

        batch = {
            "vehicle_states": vehicle_states,
            "vehicle_ids": vehicle_ids,
            "icv_ids": icv_ids,
            "global_metrics": observation.global_stats.to(self.device),
            "is_icv": torch.tensor(
                [1 if vid in icv_ids else 0 for vid in vehicle_ids],
                dtype=torch.float32,
                device=self.device,
            ),
        }

        # 构建图数据
        graph_data = self._build_graph(observation)
        batch["graph_data"] = graph_data

        return batch

    def _build_graph(self, observation: Observation) -> Any:
        """从观测构建图数据"""
        states = observation.vehicle_states
        vehicle_ids = observation.vehicle_ids

        if not vehicle_ids:
            return self.model.graph_builder.build_graph(
                positions=torch.zeros(0, 2),
                velocities=torch.zeros(0, 2),
                accelerations=torch.zeros(0, 2),
                lane_indices=torch.zeros(0, dtype=torch.long),
            )

        positions = []
        velocities = []
        accelerations = []
        lane_indices = []

        for vid in vehicle_ids:
            if vid not in states:
                continue
            state = states[vid]

            positions.append([state.x, state.y])
            velocities.append([state.speed * 0.707, state.speed * 0.707])
            accelerations.append([state.acceleration * 0.707, state.acceleration * 0.707])
            lane_indices.append(state.lane_index)

        positions = torch.tensor(positions, dtype=torch.float32)
        velocities = torch.tensor(velocities, dtype=torch.float32)
        accelerations = torch.tensor(accelerations, dtype=torch.float32)
        lane_indices = torch.tensor(lane_indices, dtype=torch.long)

        graph_data = self.model.graph_builder.build_graph(
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            lane_indices=lane_indices,
        )

        return graph_data

    def _convert_output_to_actions(self, output: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """将模型输出转换为动作字典"""
        import numpy as np

        action_dict = {}
        selected_ids = output.get("selected_vehicle_ids", [])
        safe_actions = output.get("safe_actions")

        if safe_actions is None or safe_actions.numel() == 0:
            return action_dict

        actions_np = safe_actions.cpu().numpy()

        for i, veh_id in enumerate(selected_ids):
            if i < len(actions_np):
                action_dict[veh_id] = actions_np[i]

        return action_dict

    def _compute_cost(self, output: Dict[str, Any], actions: Dict[str, np.ndarray]) -> float:
        """计算干预成本"""
        alpha = 1.0  # 加速度成本系数
        beta = 5.0   # 换道成本系数

        total_cost = 0.0

        for veh_id, action in actions.items():
            accel_cost = alpha * abs(action[0])
            lane_cost = beta if action[1] > 0.5 else 0.0
            total_cost += (accel_cost + lane_cost)

        # 归一化
        num_vehicles = len(actions) if actions else 1
        return total_cost / max(num_vehicles, 1)

    def _aggregate_metrics(self, episode_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总多个episode的指标"""
        aggregated = {}

        # 对每个指标计算均值和标准差
        metric_keys = episode_metrics[0].keys()

        for key in metric_keys:
            values = [m[key] for m in episode_metrics]

            if isinstance(values[0], (int, float)):
                aggregated[f"{key}_mean"] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))
                aggregated[f"{key}_min"] = float(np.min(values))
                aggregated[f"{key}_max"] = float(np.max(values))
            else:
                aggregated[key] = values[0]

        # 添加统计信息
        aggregated["num_episodes"] = len(episode_metrics)

        return aggregated

    def _print_summary(self, metrics: Dict[str, Any]):
        """打印评估摘要"""
        logger.info("评估指标摘要:")
        logger.info(f"  OD完成率 (OCR): {metrics.get('ocr_mean', 0):.2%} ± {metrics.get('ocr_std', 0):.2%}")
        logger.info(f"  平均速度: {metrics.get('avg_speed_mean', 0):.2f} ± {metrics.get('avg_speed_std', 0):.2f} m/s")
        logger.info(f"  速度标准差 (σv): {metrics.get('speed_std_mean', 0):.2f} ± {metrics.get('speed_std_std', 0):.2f} m/s")
        logger.info(f"  平均绝对加速度 (|a|avg): {metrics.get('avg_accel_mean', 0):.2f} ± {metrics.get('avg_accel_std', 0):.2f} m/s²")
        logger.info(f"  平均干预成本: {metrics.get('avg_cost_mean', 0):.4f} ± {metrics.get('avg_cost_std', 0):.4f}")
        logger.info(f"  总奖励: {metrics.get('total_reward_mean', 0):.2f} ± {metrics.get('total_reward_std', 0):.2f}")
        logger.info(f"  平均行程时间: {metrics.get('avg_travel_time_mean', 0):.1f} ± {metrics.get('avg_travel_time_std', 0):.1f} s")
        logger.info(f"  碰撞次数: {metrics.get('collisions_mean', 0):.1f}")
