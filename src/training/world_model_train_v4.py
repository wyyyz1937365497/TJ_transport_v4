"""
Phase 1 世界模型训练器 v4.0

训练目标：
- 学习车辆状态编码器（RiskSensitiveGNN）
- 学习交通流预测器（MultiScaleRSSM）
- 学习动态权重门控（EnhancedDynamicWeightGating）
- 预训练特征提取器，为Phase 2的PPO训练提供良好初始化

使用方法：
    from src.training.world_model_train_v4 import WorldModelTrainer

    trainer = WorldModelTrainer(config=config, device=device)
    trainer.train()
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import numpy as np

from src.models.ideal_policy_v4 import IdealTrafficPolicyV4
from src.env.competition_env import CompetitionSumoEnv


class WorldModelDataset(torch.utils.data.Dataset):
    """
    世界模型训练数据集

    收集真实交通数据用于监督学习预训练
    """

    def __init__(
        self,
        env_config: Dict[str, Any],
        num_episodes: int = 100,
        steps_per_episode: int = 1000,
        seed: int = 42
    ):
        """
        Args:
            env_config: 环境配置
            num_episodes: 收集的episode数量
            steps_per_episode: 每个episode的步数
            seed: 随机种子
        """
        self.env_config = env_config
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode
        self.seed = seed

        # 数据存储
        self.observations = []
        self.next_observations = []

        print(f"[DATASET] Collecting {num_episodes} episodes...")
        self._collect_data()
        print(f"[OK] Dataset collected: {len(self.observations)} samples")

    def _collect_data(self):
        """使用环境收集数据"""
        env = CompetitionSumoEnv(config=self.env_config)

        for episode in tqdm(range(self.num_episodes), desc="Collecting data"):
            obs, _ = env.reset(seed=self.seed + episode)

            for step in range(self.steps_per_episode):
                # 随机动作
                action = env.action_space.sample()

                next_obs, reward, done, truncated, info = env.step(action)

                # 存储数据
                self.observations.append(obs)
                self.next_observations.append(next_obs)

                if done or truncated:
                    break

                obs = next_obs

        env.close()

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.observations[idx]).float(),
            torch.from_numpy(self.next_observations[idx]).float()
        )


class WorldModelTrainer:
    """
    世界模型训练器（Phase 1）

    训练流程：
    1. 收集真实交通数据（使用随机策略）
    2. 监督学习预训练特征提取器
       - GNN: 车辆交互建模
       - RSSM: 交通流预测
       - 动态权重门控: 场景识别
    3. 保存预训练权重供Phase 2使用
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
    ):
        """
        Args:
            config: 配置字典
            device: 计算设备
        """
        self.config = config
        self.device = device

        # Phase 1配置
        phase1_config = config.get('training', {}).get('phase1', {})
        self.num_epochs = phase1_config.get('num_epochs', 100)
        self.batch_size = phase1_config.get('batch_size', 32)
        self.learning_rate = phase1_config.get('learning_rate', 1e-3)
        self.num_episodes = phase1_config.get('num_episodes', 100)
        self.steps_per_episode = phase1_config.get('steps_per_episode', 1000)

        # 环境配置（使用简化场景）
        env_config = config.get('environment', {}).copy()
        env_config.update({
            'max_vehicles': 10,  # Phase 1使用小规模场景
            'inflow_rate': 800,
            'icv_ratio': 0.3,
            'disturbance_level': 0.0,
        })
        self.env_config = env_config

        # 创建模型
        print("[MODEL] Creating policy network for pretraining...")
        self.model = IdealTrafficPolicyV4(
            obs_dim=321,
            action_dim=2,
            config=config
        ).to(device)

        # 创建优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

        # 损失函数
        self.mse_loss = nn.MSELoss()

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        print(f"[OK] WorldModelTrainer initialized")
        print(f"  - Device: {device}")
        print(f"  - Epochs: {self.num_epochs}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Learning rate: {self.learning_rate}")

    def collect_data(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """
        收集训练和验证数据

        Returns:
            train_dataset, val_dataset
        """
        print("\n[COLLECT] Collecting training data...")

        # 训练集（80%）
        train_dataset = WorldModelDataset(
            env_config=self.env_config,
            num_episodes=int(self.num_episodes * 0.8),
            steps_per_episode=self.steps_per_episode,
            seed=self.config.get('seed', 42)
        )

        # 验证集（20%）
        val_dataset = WorldModelDataset(
            env_config=self.env_config,
            num_episodes=int(self.num_episodes * 0.2),
            steps_per_episode=self.steps_per_episode,
            seed=self.config.get('seed', 42) + 1000
        )

        return train_dataset, val_dataset

    def compute_prediction_loss(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算预测损失

        目标：让模型学习预测下一个时刻的状态

        Args:
            obs: 当前观测 [batch_size, 321]
            next_obs: 下一个观测 [batch_size, 321]

        Returns:
            total_loss, loss_dict
        """
        # 提取当前观测的特征
        features_dict = self.model.extract_features(obs)

        # 提取下一个观测的特征（用于预测目标）
        with torch.no_grad():
            next_features_dict = self.model.extract_features(next_obs)

        # 预测损失：预测下一时刻的全局嵌入
        pred_global = features_dict['global_embedding']
        target_global = next_features_dict['global_embedding']

        # 清理NaN
        pred_global = torch.nan_to_num(pred_global, nan=0.0, posinf=0.0, neginf=0.0)
        target_global = torch.nan_to_num(target_global, nan=0.0, posinf=0.0, neginf=0.0)

        # MSE损失
        loss_global = self.mse_loss(pred_global, target_global)

        # 预测下一时刻的节点嵌入
        node_embeddings = features_dict['node_embeddings']
        next_node_embeddings = next_features_dict['node_embeddings']

        # 对齐维度（可能由于车辆数量不同）
        if node_embeddings.size(0) == next_node_embeddings.size(0):
            loss_node = self.mse_loss(
                node_embeddings,
                next_node_embeddings
            )
        else:
            # 如果车辆数量不同，使用均值
            loss_node = self.mse_loss(
                node_embeddings.mean(dim=0, keepdim=True).expand_as(next_node_embeddings),
                next_node_embeddings
            )

        # 总损失
        total_loss = loss_global + 0.5 * loss_node

        loss_dict = {
            'global': loss_global.item(),
            'node': loss_node.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int
    ) -> float:
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch

        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")

        for batch_idx, (obs, next_obs) in enumerate(pbar):
            obs = obs.to(self.device)
            next_obs = next_obs.to(self.device)

            # 前向传播
            loss, loss_dict = self.compute_prediction_loss(obs, next_obs)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 记录
            total_loss += loss.item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{total_loss/num_batches:.4f}'
            })

        return total_loss / num_batches

    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        验证

        Args:
            val_loader: 验证数据加载器

        Returns:
            平均损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for obs, next_obs in val_loader:
                obs = obs.to(self.device)
                next_obs = next_obs.to(self.device)

                loss, _ = self.compute_prediction_loss(obs, next_obs)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 80)
        print("[PHASE 1] World Model Training")
        print("=" * 80)

        # 1. 收集数据
        train_dataset, val_dataset = self.collect_data()

        # 2. 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        print(f"\n[DATA] Train samples: {len(train_dataset)}")
        print(f"[DATA] Val samples: {len(val_dataset)}")
        print(f"[DATA] Train batches: {len(train_loader)}")
        print(f"[DATA] Val batches: {len(val_loader)}")

        # 3. 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15

        start_time = time.time()

        for epoch in range(self.num_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)

            # 验证
            val_loss = self.validate(val_loader)

            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)

            # 打印进度
            print(f"\n[EPOCH {epoch+1}/{self.num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {current_lr:.6f}")
            print(f"  Time:       {time.time() - start_time:.1f}s")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # 保存检查点
                checkpoint_dir = Path(self.config.get('paths', {}).get('checkpoint_dir', 'checkpoints/competition'))
                checkpoint_dir = checkpoint_dir / 'phase1'
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                checkpoint_path = checkpoint_dir / 'world_model_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                    'config': self.config
                }, checkpoint_path)

                print(f"  [SAVE] Best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n[EARLY STOP] No improvement for {patience} epochs")
                    break

        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print("[DONE] Phase 1 Training Completed!")
        print(f"  - Total time: {total_time/60:.1f} minutes")
        print(f"  - Best val loss: {best_val_loss:.4f}")
        print("=" * 80)

    def save_checkpoint(self, filepath: str):
        """
        保存最终检查点

        Args:
            filepath: 保存路径
        """
        checkpoint_dir = Path(filepath).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, filepath)

        print(f"[SAVE] Final checkpoint saved: {filepath}")
