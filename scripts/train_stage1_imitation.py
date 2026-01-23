#!/usr/bin/env python3
"""
Stage 1: 模仿学习训练（Imitation Learning）

功能：
1. 加载Stage 0收集的演示数据
2. 使用SimplifiedICVPolicy进行模仿学习
3. 最小化MSE损失拟合专家演示

使用方法：
    python scripts/train_stage1_imitation.py --config configs/ocr_max.yaml --demo_data data/demonstrations/demonstrations.pkl

输出：
    - checkpoints/ocr_max/stage1_imitation_best.pth
    - logs/ocr_max/stage1_imitation/
"""

import os
import sys
import argparse
import yaml
import pickle
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.simplified_icv_policy import SimplifiedICVPolicy


class ImitationLearningTrainer:
    """
    模仿学习训练器（Stage 1）
    """

    def __init__(
        self,
        policy,
        demonstrations_path: str,
        config: dict,
        device: str = 'cuda',
        log_dir: str = 'logs/ocr_max/stage1_imitation',
        checkpoint_dir: str = 'checkpoints/ocr_max'
    ):
        """
        Args:
            policy: SimplifiedICVPolicy
            demonstrations_path: 演示数据路径
            config: 配置字典
            device: 设备
            log_dir: 日志目录
            checkpoint_dir: 检查点目录
        """
        self.policy = policy.to(device)
        self.config = config
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        # 创建目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 加载演示数据
        print(f"加载演示数据: {demonstrations_path}")
        with open(demonstrations_path, 'rb') as f:
            demo_data = pickle.load(f)

        self.demonstrations = demo_data['episodes']
        self.metadata = demo_data['metadata']

        print(f"  Episodes: {len(self.demonstrations)}")
        print(f"  平均长度: {self.metadata['avg_length']:.1f}")
        print(f"  平均奖励: {self.metadata['avg_reward']:.2f}")

        # 训练配置
        stage1_config = config.get('stage1_imitation_learning', {})
        self.num_epochs = stage1_config.get('num_epochs', 50)
        self.batch_size = stage1_config.get('batch_size', 32)
        self.lr = stage1_config.get('lr', 1e-3)
        self.grad_clip = stage1_config.get('grad_clip', 1.0)

        # 优化器
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.lr
        )

        # 损失函数
        self.mse_loss = nn.MSELoss()

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # TensorBoard
        self.writer = SummaryWriter(log_dir)

        # 训练统计
        self.best_loss = float('inf')
        self.global_step = 0

    def prepare_batch(self, transitions: list) -> tuple:
        """
        准备训练批次

        Args:
            transitions: 转换列表

        Returns:
            obs_batch: [B, obs_dim] 观测批次
            actions_batch: [B, N*2] 动作批次
        """
        obs_list = []
        actions_list = []

        for transition in transitions:
            obs = transition['obs']  # [obs_dim]
            actions_dict = transition['actions']  # {veh_id: [accel, lane]}
            selected_vehicles = transition['selected_vehicles']

            # 构建动作数组 [N*2]
            # 假设selected_vehicles数量<=32，按照字典序排列
            num_vehicles = len(selected_vehicles)
            actions_array = np.zeros(32 * 2, dtype=np.float32)

            for i, veh_id in enumerate(selected_vehicles):
                if i < 32 and veh_id in actions_dict:
                    actions_array[i * 2] = actions_dict[veh_id][0]  # accel
                    actions_array[i * 2 + 1] = actions_dict[veh_id][1]  # lane

            obs_list.append(obs)
            actions_list.append(actions_array)

        obs_batch = torch.from_numpy(np.array(obs_list)).float().to(self.device)
        actions_batch = torch.from_numpy(np.array(actions_list)).float().to(self.device)

        return obs_batch, actions_batch

    def train_epoch(self, epoch: int) -> dict:
        """
        训练一个epoch

        Args:
            epoch: 当前epoch

        Returns:
            metrics: 训练指标
        """
        self.policy.train()

        # 收集所有转换
        all_transitions = []
        for episode in self.demonstrations:
            all_transitions.extend(episode['transitions'])

        # 随机打乱
        np.random.shuffle(all_transitions)

        # 分批
        num_batches = len(all_transitions) // self.batch_size
        if num_batches == 0:
            num_batches = 1

        total_loss = 0.0
        total_accel_loss = 0.0
        total_lane_loss = 0.0

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch_idx in pbar:
            # 采样批次
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(all_transitions))
            batch_transitions = all_transitions[start_idx:end_idx]

            # 准备数据
            obs_batch, actions_batch = self.prepare_batch(batch_transitions)

            # 前向传播
            self.optimizer.zero_grad()

            # 确定性模式：使用policy输出
            outputs = self.policy(obs_batch, deterministic=True)
            pred_actions = outputs['actions']  # [B, N*2]

            # 计算损失
            loss = self.mse_loss(pred_actions, actions_batch)

            # 分别计算加速度和换道损失
            pred_accel = pred_actions[:, :32]
            target_accel = actions_batch[:, :32]
            accel_loss = self.mse_loss(pred_accel, target_accel)

            pred_lane = pred_actions[:, 32:]
            target_lane = actions_batch[:, 32:]
            lane_loss = self.mse_loss(pred_lane, target_lane)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.grad_clip
            )

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            total_accel_loss += accel_loss.item()
            total_lane_loss += lane_loss.item()

            # TensorBoard
            self.global_step += 1
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/accel_loss', accel_loss.item(), self.global_step)
                self.writer.add_scalar('train/lane_loss', lane_loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'accel': f'{accel_loss.item():.4f}',
                'lane': f'{lane_loss.item():.4f}'
            })

        # 平均损失
        avg_loss = total_loss / num_batches
        avg_accel_loss = total_accel_loss / num_batches
        avg_lane_loss = total_lane_loss / num_batches

        metrics = {
            'loss': avg_loss,
            'accel_loss': avg_accel_loss,
            'lane_loss': avg_lane_loss
        }

        return metrics

    def evaluate(self) -> dict:
        """
        评估模型

        Returns:
            metrics: 评估指标
        """
        self.policy.eval()

        total_loss = 0.0
        total_accel_loss = 0.0
        total_lane_loss = 0.0

        num_batches = 0

        with torch.no_grad():
            for episode in self.demonstrations:
                transitions = episode['transitions']

                # 分批评估
                for i in range(0, len(transitions), self.batch_size):
                    batch_transitions = transitions[i:i+self.batch_size]

                    obs_batch, actions_batch = self.prepare_batch(batch_transitions)

                    # 前向传播
                    outputs = self.policy(obs_batch, deterministic=True)
                    pred_actions = outputs['actions']

                    # 计算损失
                    loss = self.mse_loss(pred_actions, actions_batch)

                    pred_accel = pred_actions[:, :32]
                    target_accel = actions_batch[:, :32]
                    accel_loss = self.mse_loss(pred_accel, target_accel)

                    pred_lane = pred_actions[:, 32:]
                    target_lane = actions_batch[:, 32:]
                    lane_loss = self.mse_loss(pred_lane, target_lane)

                    total_loss += loss.item()
                    total_accel_loss += accel_loss.item()
                    total_lane_loss += lane_loss.item()
                    num_batches += 1

        avg_loss = total_loss / num_batches
        avg_accel_loss = total_accel_loss / num_batches
        avg_lane_loss = total_lane_loss / num_batches

        metrics = {
            'loss': avg_loss,
            'accel_loss': avg_accel_loss,
            'lane_loss': avg_lane_loss
        }

        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict, filename: str = None):
        """
        保存检查点

        Args:
            epoch: 当前epoch
            metrics: 训练指标
            filename: 文件名（可选）
        """
        if filename is None:
            filename = f'stage1_imitation_epoch{epoch}.pth'

        checkpoint = {
            'epoch': epoch,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_loss': self.best_loss,
            'config': self.config
        }

        save_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, save_path)

        print(f"  检查点已保存: {save_path}")

    def train(self):
        """
        完整训练流程
        """
        print(f"\n{'='*80}")
        print("Stage 1: 模仿学习训练")
        print(f"{'='*80}\n")

        print(f"配置:")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.lr}")
        print(f"  Gradient clip: {self.grad_clip}")
        print(f"  Device: {self.device}\n")

        for epoch in range(self.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.num_epochs} ---")

            # 训练
            train_metrics = self.train_epoch(epoch)

            print(f"\n训练损失:")
            print(f"  Total: {train_metrics['loss']:.4f}")
            print(f"  Accel: {train_metrics['accel_loss']:.4f}")
            print(f"  Lane: {train_metrics['lane_loss']:.4f}")

            # 评估
            val_metrics = self.evaluate()

            print(f"\n验证损失:")
            print(f"  Total: {val_metrics['loss']:.4f}")
            print(f"  Accel: {val_metrics['accel_loss']:.4f}")
            print(f"  Lane: {val_metrics['lane_loss']:.4f}")

            # 学习率调度
            self.scheduler.step(val_metrics['loss'])

            # TensorBoard
            self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('val/accel_loss', val_metrics['accel_loss'], epoch)
            self.writer.add_scalar('val/lane_loss', val_metrics['lane_loss'], epoch)

            # 保存检查点
            if val_metrics['loss'] < self.best_loss:
                self.best_loss = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics, 'stage1_best.pth')
                print(f"  ✅ 新的最佳模型: loss={val_metrics['loss']:.4f}")

            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_metrics)

        print(f"\n{'='*80}")
        print("训练完成！")
        print(f"  最佳验证损失: {self.best_loss:.4f}")
        print(f"{'='*80}\n")

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Stage 1: 模仿学习训练')
    parser.add_argument('--config', type=str, default='configs/ocr_max.yaml',
                        help='配置文件路径')
    parser.add_argument('--demo_data', type=str, required=True,
                        help='演示数据路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 创建策略网络
    print("创建策略网络...")
    policy_config = config.get('policy', {})

    policy = SimplifiedICVPolicy(
        obs_dim=policy_config.get('obs_dim', 321),
        node_dim=policy_config.get('node_dim', 9),
        hidden_dim=policy_config.get('hidden_dim', 128),
        num_layers=policy_config.get('num_layers', 3),
        num_vehicles=policy_config.get('num_vehicles', 32),
        device=args.device,
        use_safety_shield=False  # 训练时不使用SafetyShield
    )

    # 创建训练器
    trainer = ImitationLearningTrainer(
        policy=policy,
        demonstrations_path=args.demo_data,
        config=config,
        device=args.device,
        log_dir=config.get('logging', {}).get('log_dir', 'logs/ocr_max') + '/stage1_imitation',
        checkpoint_dir=config.get('checkpoints', {}).get('dir', 'checkpoints/ocr_max')
    )

    # 恢复训练（可选）
    if args.resume is not None:
        print(f"从检查点恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.best_loss = checkpoint['best_loss']
        print(f"  恢复epoch: {checkpoint['epoch']}")
        print(f"  恢复最佳损失: {checkpoint['best_loss']:.4f}")

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
