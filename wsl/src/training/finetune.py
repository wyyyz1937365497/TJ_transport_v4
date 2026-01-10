"""
端到端微调模块 - 完整实现
阶段3: 联合优化所有组件（GNN + 世界模型 + 控制器）
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


class EndToEndFineTuner:
    """
    端到端微调器 - 完整实现

    联合优化所有组件：
    - GNN (风险感知图神经网络)
    - World Model (渐进式世界模型)
    - Controller (影响力驱动的控制器)
    """

    def __init__(
        self,
        model: TrafficController,
        config: Config,
    ):
        self.model = model
        self.config = config

        # 提取微调配置
        finetune_cfg = config.training.get("phase3", {})

        self.total_timesteps = finetune_cfg.get("total_timesteps", 30000)
        self.learning_rate = finetune_cfg.get("learning_rate", 1e-5)
        self.freeze_bn = finetune_cfg.get("freeze_bn", True)
        self.weight_decay = finetune_cfg.get("weight_decay", 1e-5)
        self.max_grad_norm = finetune_cfg.get("max_grad_norm", 0.3)
        self.update_freq = finetune_cfg.get("update_freq", 32)
        self.batch_size = finetune_cfg.get("batch_size", 16)

        # 设备
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # 解冻所有组件
        self.model.unfreeze_component("gnn")
        self.model.unfreeze_component("world_model")
        self.model.unfreeze_component("controller")

        # 冻结BatchNorm（如果需要）
        if self.freeze_bn:
            self._freeze_batch_norm()

        # 优化器（所有参数）
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_timesteps,
            eta_min=self.learning_rate * 0.01,
        )

        # 经验回放缓冲区
        self.buffer = deque(maxlen=self.update_freq * 2)

        # 统计
        self.stats = {
            "episode_rewards": [],
            "episode_costs": [],
            "losses": [],
        }

        # 日志
        self.logger = setup_logger(
            log_file=config.log_dir / "finetune_training.log",
        )

        # 打印可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"可训练参数数: {trainable_params:,}")

    def _freeze_batch_norm(self):
        """冻结BatchNorm层"""
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()
        self.logger.info("BatchNorm层已冻结")

    def collect_experience(
        self,
        env: SumoEnvironment,
        num_steps: int,
    ) -> Tuple[List[Dict], List[float], List[float]]:
        """
        收集经验

        Args:
            env: SUMO环境
            num_steps: 收集步数

        Returns:
            (states, rewards, costs)
        """
        states = []
        rewards = []
        costs = []

        # 重置环境
        state = env.reset()

        for step in range(num_steps):
            # 准备batch
            batch = self._prepare_batch(state, env)

            # 获取动作
            with torch.no_grad():
                output = self.model(batch)

            # 转换为动作字典
            action_dict = self._convert_output_to_actions(output)

            # 执行动作
            next_state, reward, done, info = env.step(action_dict)

            # 计算成本
            cost = self._compute_cost(output, action_dict)

            # 存储经验
            states.append({
                "state": batch,
                "output": output,
            })
            rewards.append(reward)
            costs.append(cost)

            # 检查是否结束
            if done:
                state = env.reset()
            else:
                state = next_state

        return states, rewards, costs

    def update(
        self,
        experiences: List[Dict],
        rewards: List[float],
    ) -> Dict[str, float]:
        """
        端到端更新

        Args:
            experiences: 经验列表
            rewards: 奖励列表

        Returns:
            损失统计
        """
        if len(experiences) < self.batch_size:
            return {"loss": 0.0}

        losses = []

        # 多次更新
        num_updates = max(1, len(experiences) // self.batch_size)

        for _ in range(num_updates):
            # 随机采样mini-batch
            indices = np.random.permutation(len(experiences))[:self.batch_size]

            batch_experiences = [experiences[i] for i in indices]
            batch_rewards = [rewards[i] for i in indices]

            # 计算returns（折扣累计奖励）
            gamma = 0.99
            returns = []
            R = 0
            for reward in reversed(batch_rewards):
                R = reward + gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

            # 前向传播
            all_value_preds = []
            all_action_preds = []

            for exp in batch_experiences:
                batch = exp["state"]
                output = exp["output"]

                # 提取价值估计
                if output.get("value_estimates") is not None and output["value_estimates"].numel() > 0:
                    all_value_preds.append(output["value_estimates"].mean())
                else:
                    all_value_preds.append(torch.zeros(1, device=self.device)[0])

                # 提取动作
                if output.get("safe_actions") is not None and output["safe_actions"].numel() > 0:
                    all_action_preds.append(output["safe_actions"])
                else:
                    all_action_preds.append(torch.zeros(1, 2, device=self.device))

            # 计算损失
            if len(all_value_preds) > 0:
                value_preds = torch.stack(all_value_preds)
                value_loss = F.mse_loss(value_preds, returns)
            else:
                value_loss = torch.zeros(1, device=self.device)

            if len(all_action_preds) > 0:
                actions_tensor = torch.cat(all_action_preds, dim=0)
                # 策略损失：最大化奖励
                policy_loss = -(actions_tensor.mean() * returns.mean())
            else:
                policy_loss = torch.zeros(1, device=self.device)

            # 熵损失（鼓励探索）
            if len(all_action_preds) > 0:
                action_std = actions_tensor.std(dim=0).mean() + 1e-8
                entropy_loss = -0.01 * action_std
            else:
                entropy_loss = torch.zeros(1, device=self.device)

            # 总损失
            loss = value_loss + 0.1 * policy_loss + entropy_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            losses.append(loss.item())

        # 更新学习率
        self.scheduler.step()

        return {
            "loss": np.mean(losses),
            "lr": self.scheduler.get_last_lr()[0],
        }

    def train(self) -> TrafficController:
        """
        完整训练流程

        Returns:
            训练好的模型
        """
        self.logger.info("=" * 70)
        self.logger.info("端到端微调开始")
        self.logger.info("=" * 70)

        # 创建环境
        env = SumoEnvironment(
            self.config.environment,
            use_gui=False,
        )

        # 训练循环
        timestep = 0
        episode_rewards = []
        episode_costs = []

        with tqdm(total=self.total_timesteps, desc="端到端微调") as pbar:
            while timestep < self.total_timesteps:
                # 收集经验
                collect_steps = min(self.update_freq, self.total_timesteps - timestep)

                experiences, rewards, costs = self.collect_experience(env, collect_steps)

                # 存储到缓冲区
                for exp, reward, cost in zip(experiences, rewards, costs):
                    self.buffer.append((exp, reward, cost))

                # 更新
                if len(self.buffer) >= self.batch_size:
                    # 准备数据
                    batch_data = list(self.buffer)
                    batch_exp = [item[0] for item in batch_data]
                    batch_rew = [item[1] for item in batch_data]

                    update_stats = self.update(batch_exp, batch_rew)

                    # 记录统计
                    self.stats["losses"].append(update_stats["loss"])

                # 更新时间步
                timestep += collect_steps
                pbar.update(collect_steps)

                # 记录episode统计
                episode_rewards.extend(rewards)
                episode_costs.extend(costs)

                # 打印进度
                if timestep % 1000 == 0:
                    mean_reward = np.mean(rewards) if rewards else 0.0
                    mean_cost = np.mean(costs) if costs else 0.0

                    pbar.set_postfix({
                        "reward": f"{mean_reward:.2f}",
                        "cost": f"{mean_cost:.4f}",
                        "lr": f"{update_stats.get('lr', 0):.2e}",
                    })

                    self.logger.info(
                        f"Timestep {timestep}/{self.total_timesteps} | "
                        f"Reward: {mean_reward:.2f} | "
                        f"Cost: {mean_cost:.4f} | "
                        f"LR: {update_stats.get('lr', 0):.2e}"
                    )

        env.close()

        self.logger.info("=" * 70)
        self.logger.info("端到端微调完成")
        if episode_rewards:
            self.logger.info(f"最终平均奖励: {np.mean(episode_rewards[-100:]):.2f}")
        if episode_costs:
            self.logger.info(f"最终平均成本: {np.mean(episode_costs[-100:]):.4f}")
        self.logger.info("=" * 70)

        # 保存检查点
        self._save_checkpoint()

        return self.model

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
        checkpoint_path = self.config.checkpoint_dir / "finetune_phase3.ckpt"

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
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
        self.stats = checkpoint["stats"]

        self.logger.info(f"检查点已加载: {checkpoint_path}")
