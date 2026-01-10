"""
约束优化训练模块 - 完整实现
阶段4: 使用拉格朗日乘子法平衡性能与成本
"""

import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..models import TrafficController
from ..env import SumoEnvironment
from ..utils import Config, setup_logger, get_logger
from ..utils.dataclass import Observation

logger = get_logger()


class ConstrainedOptimizer:
    """
    约束优化器 - 完整实现

    使用拉格朗日乘子法进行约束优化：
    - 目标：最大化奖励
    - 约束：成本 ≤ cost_limit
    """

    def __init__(
        self,
        model: TrafficController,
        config: Config,
    ):
        self.model = model
        self.config = config

        # 提取配置
        opt_cfg = config.training.get("phase4", {})

        self.total_timesteps = opt_cfg.get("total_timesteps", 20000)
        self.cost_limit = opt_cfg.get("cost_limit", 0.1)
        self.learning_rate = opt_cfg.get("learning_rate", 1e-4)
        self.lambda_lr = opt_cfg.get("lambda_lr", 0.01)
        self.max_grad_norm = opt_cfg.get("max_grad_norm", 0.5)

        # 拉格朗日乘子
        self.register_buffer(
            "lagrange_multiplier",
            torch.tensor(opt_cfg.get("initial_lambda", 1.0)),
        )

        # 设备
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.lagrange_multiplier = self.lagrange_multiplier.to(self.device)

        # 冻结GNN和世界模型，只训练控制器
        self.model.freeze_component("gnn")
        self.model.freeze_component("world_model")
        self.model.unfreeze_component("controller")

        # 优化器（只优化控制器）
        self.optimizer = torch.optim.Adam(
            self.model.controller.parameters(),
            lr=self.learning_rate,
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_timesteps,
            eta_min=self.learning_rate * 0.01,
        )

        # 统计
        self.stats = {
            "episode_rewards": [],
            "episode_costs": [],
            "lambdas": [],
            "constraint_violations": [],
        }

        # 日志
        self.logger = setup_logger(
            log_file=config.log_dir / "constrained_training.log",
        )

        self.logger.info(f"成本约束上限: {self.cost_limit}")
        self.logger.info(f"初始拉格朗日乘子: {self.lagrange_multiplier.item():.3f}")

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """注册buffer"""
        setattr(self, name, tensor)

    def update_lagrange_multiplier(self, cost: float):
        """
        更新拉格朗日乘子

        Args:
            cost: 当前成本
        """
        # 自适应更新
        if cost > self.cost_limit:
            # 成本超标，增加惩罚
            self.lagrange_multiplier.data = torch.clamp(
                self.lagrange_multiplier.data * (1 + self.lambda_lr),
                min=0.1,
                max=10.0,
            )
        else:
            # 成本满足，减少惩罚
            self.lagrange_multiplier.data = torch.clamp(
                self.lagrange_multiplier.data * (1 - self.lambda_lr * 0.5),
                min=0.1,
                max=10.0,
            )

    def compute_constrained_loss(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算约束损失

        Args:
            reward: 奖励
            cost: 成本
            value: 价值估计

        Returns:
            (total_loss, loss_dict)
        """
        # 1. 策略损失（带拉格朗日乘子）
        # 目标：最大化 reward - lambda * max(0, cost - cost_limit)
        cost_violation = torch.relu(cost - self.cost_limit)
        policy_loss = -(reward - self.lagrange_multiplier * cost_violation)

        # 2. 价值损失
        value_loss = F.mse_loss(value, reward)

        # 3. 总损失
        total_loss = policy_loss + 0.5 * value_loss

        # 4. 拉格朗日损失（用于日志）
        lagrange_loss = self.lagrange_multiplier * cost_violation + 0.5 * self.lagrange_multiplier ** 2

        loss_dict = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "lagrange_loss": lagrange_loss,
            "cost_violation": cost_violation,
        }

        return total_loss, loss_dict

    def collect_episode(
        self,
        env: SumoEnvironment,
    ) -> Tuple[List[Dict], List[float], List[float], float, float]:
        """
        收集一个完整episode

        Args:
            env: SUMO环境

        Returns:
            (experiences, rewards, costs, total_reward, total_cost)
        """
        experiences = []
        rewards = []
        costs = []

        # 重置环境
        observation = env.reset()
        done = False
        step = 0

        total_reward = 0.0
        total_cost = 0.0

        while not done and step < self.config.environment.get("max_steps", 3600):
            # 准备batch
            batch = self._prepare_batch(observation, env)

            # 前向传播
            with torch.no_grad():
                output = self.model(batch)

            # 转换为动作字典
            action_dict = self._convert_output_to_actions(output)

            # 执行动作
            next_observation, reward, done, info = env.step(action_dict)

            # 计算成本
            cost = self._compute_cost(output, action_dict)

            # 存储经验
            experiences.append({
                "state": batch,
                "output": output,
            })
            rewards.append(reward)
            costs.append(cost)

            total_reward += reward
            total_cost += cost

            # 更新拉格朗日乘子
            self.update_lagrange_multiplier(cost)

            observation = next_observation
            step += 1

        return experiences, rewards, costs, total_reward, total_cost

    def update(
        self,
        experiences: List[Dict],
        rewards: List[float],
        costs: List[float],
    ) -> Dict[str, float]:
        """
        更新策略

        Args:
            experiences: 经验列表
            rewards: 奖励列表
            costs: 成本列表

        Returns:
            损失统计
        """
        if len(experiences) < 10:
            return {"loss": 0.0}

        losses = []
        policy_losses = []
        value_losses = []
        lagrange_losses = []

        # 计算returns
        gamma = 0.99
        returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        costs_tensor = torch.tensor(costs, dtype=torch.float32).to(self.device)

        # 多次更新
        batch_size = 32
        num_updates = max(1, len(experiences) // batch_size)

        for _ in range(num_updates):
            # 随机采样
            indices = np.random.permutation(len(experiences))[:batch_size]

            # 前向传播
            all_values = []

            for i in indices:
                exp = experiences[i]
                output = exp["output"]

                if output.get("value_estimates") is not None and output["value_estimates"].numel() > 0:
                    all_values.append(output["value_estimates"].mean())
                else:
                    all_values.append(torch.zeros(1, device=self.device)[0])

            if len(all_values) > 0:
                values = torch.stack(all_values)
                avg_return = returns[indices].mean()
                avg_cost = costs_tensor[indices].mean()

                # 计算损失
                total_loss, loss_dict = self.compute_constrained_loss(
                    reward=avg_return,
                    cost=avg_cost,
                    value=values.mean(),
                )

                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.controller.parameters(), self.max_grad_norm)
                self.optimizer.step()

                losses.append(total_loss.item())
                policy_losses.append(loss_dict["policy_loss"].item())
                value_losses.append(loss_dict["value_loss"].item())
                lagrange_losses.append(loss_dict["lagrange_loss"].item())

        # 更新学习率
        self.scheduler.step()

        return {
            "loss": np.mean(losses) if losses else 0.0,
            "policy_loss": np.mean(policy_losses) if policy_losses else 0.0,
            "value_loss": np.mean(value_losses) if value_losses else 0.0,
            "lagrange_loss": np.mean(lagrange_losses) if lagrange_losses else 0.0,
            "lr": self.scheduler.get_last_lr()[0],
        }

    def train(self) -> TrafficController:
        """
        完整训练流程

        Returns:
            训练好的模型
        """
        self.logger.info("=" * 70)
        self.logger.info("约束优化训练开始（拉格朗日乘子法）")
        self.logger.info("=" * 70)

        # 创建环境
        env = SumoEnvironment(
            self.config.environment,
            use_gui=False,
        )

        # 训练循环
        timestep = 0
        episode = 0

        with tqdm(total=self.total_timesteps, desc="约束优化训练") as pbar:
            while timestep < self.total_timesteps:
                # 收集一个episode
                experiences, rewards, costs, total_reward, total_cost = self.collect_episode(env)

                # 更新统计
                self.stats["episode_rewards"].append(total_reward)
                self.stats["episode_costs"].append(total_cost)
                self.stats["lambdas"].append(self.lagrange_multiplier.item())
                self.stats["constraint_violations"].append(max(0, total_cost - self.cost_limit))

                # 更新策略
                if len(experiences) >= 10:
                    update_stats = self.update(experiences, rewards, costs)
                else:
                    update_stats = {"loss": 0.0, "lr": self.scheduler.get_last_lr()[0]}

                # 更新时间步
                timestep += len(rewards)
                episode += 1
                pbar.update(len(rewards))

                # 打印进度
                if episode % 10 == 0:
                    avg_reward = np.mean(self.stats["episode_rewards"][-10:])
                    avg_cost = np.mean(self.stats["episode_costs"][-10:])
                    avg_lambda = np.mean(self.stats["lambdas"][-10:])

                    pbar.set_postfix({
                        "reward": f"{avg_reward:.2f}",
                        "cost": f"{avg_cost:.4f}",
                        "lambda": f"{avg_lambda:.3f}",
                        "lr": f"{update_stats.get('lr', 0):.2e}",
                    })

                    self.logger.info(
                        f"Episode {episode} | "
                        f"Avg Reward: {avg_reward:.2f} | "
                        f"Avg Cost: {avg_cost:.4f} | "
                        f"Lambda: {avg_lambda:.3f} | "
                        f"LR: {update_stats.get('lr', 0):.2e}"
                    )

        env.close()

        self.logger.info("=" * 70)
        self.logger.info("约束优化训练完成")
        if self.stats["episode_rewards"]:
            self.logger.info(f"最终平均奖励: {np.mean(self.stats['episode_rewards'][-10:]):.2f}")
        if self.stats["episode_costs"]:
            self.logger.info(f"最终平均成本: {np.mean(self.stats['episode_costs'][-10:]):.4f}")
        self.logger.info(f"最终拉格朗日乘子: {self.lagrange_multiplier.item():.3f}")
        self.logger.info(f"成本约束满足率: {self._compute_satisfaction_rate():.1%}")
        self.logger.info("=" * 70)

        # 保存检查点
        self._save_checkpoint()

        return self.model

    def _compute_satisfaction_rate(self) -> float:
        """计算成本约束满足率"""
        if not self.stats["episode_costs"]:
            return 0.0

        satisfied = sum(1 for cost in self.stats["episode_costs"] if cost <= self.cost_limit)
        return satisfied / len(self.stats["episode_costs"])

    def _prepare_batch(self, observation: Observation, env: SumoEnvironment) -> Dict[str, Any]:
        """准备训练batch"""
        vehicle_states = observation.vehicle_states
        vehicle_ids = observation.vehicle_ids
        icv_ids = observation.icv_ids

        batch = {
            "vehicle_states": vehicle_states,
            "vehicle_ids": vehicle_ids,
            "icv_ids": icv_ids,
            "global_metrics": observation.global_stats.unsqueeze(0),
            "is_icv": torch.tensor(
                [1 if vid in icv_ids else 0 for vid in vehicle_ids],
                dtype=torch.float32,
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

    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint_path = self.config.checkpoint_dir / "constrained_phase4.ckpt"

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "lagrange_multiplier": self.lagrange_multiplier.item(),
            "stats": self.stats,
            "config": self.config.to_dict(),
        }, checkpoint_path)

        self.logger.info(f"检查点已保存: {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: Path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.lagrange_multiplier.data = torch.tensor(
            checkpoint["lagrange_multiplier"],
            device=self.device,
        )
        self.stats = checkpoint["stats"]

        self.logger.info(f"检查点已加载: {checkpoint_path}")
