"""
ICV评分器训练脚本

两阶段训练流程：
1. Stage 1: 行为克隆（监督学习）- 从规则评分器学习
2. Stage 2: PPO微调（强化学习）- 直接优化OCR
"""

import sys
import os
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.env.competition_env import CompetitionSumoEnv
from src.env.rule_based_scorer import RuleBasedVehicleScorer
from src.models.icv_gnn_scorer import ICVGraphScorer, ICVGNNScorer
from src.training.custom_ppo_trainer import CustomPPOTrainer


class VehicleScoreDataset(Dataset):
    """车辆评分数据集"""

    def __init__(self, vehicle_states_list: List[torch.Tensor], scores_list: List[torch.Tensor]):
        """
        Args:
            vehicle_states_list: 车辆状态列表 [N, 9]
            scores_list: 评分列表 [N]
        """
        self.vehicle_states_list = vehicle_states_list
        self.scores_list = scores_list

    def __len__(self):
        return len(self.vehicle_states_list)

    def __getitem__(self, idx):
        return self.vehicle_states_list[idx], self.scores_list[idx]


class ICVScorerTrainer:
    """ICV评分器训练器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器

        Args:
            config: 配置字典
        """
        self.config = config

        # 基础配置
        self.device = config.get('device', 'cuda')
        self.seed = config.get('seed', 42)

        # 设置随机种子
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # 路径配置
        paths = config.get('paths', {})
        self.checkpoint_dir = Path(paths.get('checkpoint_dir', 'checkpoints/icv_scorer'))
        self.log_dir = Path(paths.get('log_dir', 'logs/icv_scorer'))
        self.tensorboard_dir = Path(paths.get('tensorboard_dir', 'runs/icv_scorer'))

        # 创建目录
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        # 初始化TensorBoard
        self.writer = SummaryWriter(str(self.tensorboard_dir))

        # 环境配置
        self.env_config = config.get('environment', {})

        # 神经网络配置
        self.neural_config = config.get('neural_icv_scoring', {})

        # Stage 1配置
        self.stage1_config = config.get('stage1_behavior_cloning', {})

        # Stage 2配置
        self.stage2_config = config.get('stage2_ppo', {})

        # 初始化模型
        self.model = None
        self.rule_scorer = None

        print(f"[ICVScorerTrainer] 初始化完成 (device={self.device})")

    def collect_data_with_rule_scorer(
        self,
        num_episodes: int,
        num_envs: int = 1
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        使用规则评分器收集训练数据

        Args:
            num_episodes: 收集的episode数
            num_envs: 并行环境数

        Returns:
            vehicle_states_list: 车辆状态列表
            scores_list: 规则评分列表
        """
        print(f"\n{'='*60}")
        print(f"Stage 1: 数据收集 (num_episodes={num_episodes}, num_envs={num_envs})")
        print(f"{'='*60}\n")

        # 初始化规则评分器
        self.rule_scorer = RuleBasedVehicleScorer(
            config=self.config,
            frenet_system=None
        )

        # 初始化环境
        env = CompetitionSumoEnv(
            config=self.env_config,
            use_gui=False,
            device=self.device
        )

        vehicle_states_list = []
        scores_list = []

        for episode in tqdm(range(num_episodes), desc="数据收集"):
            obs = env.reset()
            done = False
            step = 0

            while not done:
                # 获取车辆状态
                vehicle_states = env.get_vehicle_states()
                all_vehicle_ids = list(vehicle_states.keys())

                if len(all_vehicle_ids) == 0:
                    obs, reward, done, info = env.step({})
                    step += 1
                    continue

                # 使用规则评分器计算评分
                context = {
                    'traci_lib': env.traci,
                    'all_vehicle_ids': all_vehicle_ids
                }
                rule_scores = self.rule_scorer.compute_scores(vehicle_states, context)

                # 准备训练数据
                vehicle_tensor = self._prepare_vehicle_tensor(vehicle_states, all_vehicle_ids)
                scores_tensor = self._prepare_scores_tensor(rule_scores, all_vehicle_ids)

                vehicle_states_list.append(vehicle_tensor)
                scores_list.append(scores_tensor)

                # 随机动作（数据收集阶段）
                actions = {}
                for veh_id in all_vehicle_ids:
                    actions[veh_id] = np.random.uniform(-1, 1, size=2)

                obs, reward, done, info = env.step(actions)
                step += 1

                # 限制episode长度
                if step >= 500:  # 每个episode最多500步
                    break

            # 每10个episode打印一次进度
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes}: 收集了 {len(vehicle_states_list)} 个样本")

        env.close()

        print(f"\n数据收集完成: 总共 {len(vehicle_states_list)} 个样本\n")

        return vehicle_states_list, scores_list

    def train_stage1_behavior_cloning(
        self,
        vehicle_states_list: List[torch.Tensor],
        scores_list: List[torch.Tensor]
    ):
        """
        Stage 1: 行为克隆训练（监督学习）

        Args:
            vehicle_states_list: 车辆状态列表
            scores_list: 规则评分列表
        """
        print(f"\n{'='*60}")
        print("Stage 1: 行为克隆训练（监督学习）")
        print(f"{'='*60}\n")

        # 初始化模型
        self.model = ICVGraphScorer(
            node_dim=self.neural_config.get('node_dim', 9),
            hidden_dim=self.neural_config.get('hidden_dim', 64),
            num_layers=self.neural_config.get('num_layers', 3),
            dropout=self.neural_config.get('dropout', 0.1),
            interaction_radius=self.neural_config.get('interaction_radius', 0.15)
        ).to(self.device)

        # 创建数据集和数据加载器
        dataset = VehicleScoreDataset(vehicle_states_list, scores_list)

        # 划分训练集和验证集（80/20）
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        batch_size = self.stage1_config.get('batch_size', 32)
        num_workers = self.config.get('advanced', {}).get('num_workers', 4)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        # 优化器
        learning_rate = self.stage1_config.get('learning_rate', 0.001)
        weight_decay = self.stage1_config.get('weight_decay', 1e-5)
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度器
        scheduler_patience = self.stage1_config.get('lr_scheduler_patience', 2)
        scheduler_factor = self.stage1_config.get('lr_scheduler_factor', 0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True
        )

        # 损失函数
        criterion = nn.MSELoss()

        # 训练参数
        epochs = self.stage1_config.get('epochs', 10)

        # 混合精度训练
        use_amp = self.config.get('training', {}).get('mixed_precision', True)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # 早停
        early_stopping_patience = self.stage1_config.get('early_stopping_patience', 5)
        early_stopping_min_delta = self.stage1_config.get('early_stopping_min_delta', 1e-4)
        best_val_loss = float('inf')
        patience_counter = 0

        # 训练循环
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)

            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_samples = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")
            for batch_idx, (vehicle_states, scores) in enumerate(pbar):
                vehicle_states = vehicle_states.to(self.device)
                scores = scores.to(self.device)

                optimizer.zero_grad()

                # 前向传播
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(vehicle_states)
                        predicted_scores = outputs['scores'].squeeze(-1)  # [B, N]

                        # 计算损失（只对有效车辆计算）
                        valid_mask = (scores >= 0).float()
                        if valid_mask.sum() > 0:
                            loss = criterion(predicted_scores * valid_mask, scores * valid_mask)
                        else:
                            loss = criterion(predicted_scores, scores)

                    # 反向传播
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(vehicle_states)
                    predicted_scores = outputs['scores'].squeeze(-1)  # [B, N]

                    # 计算损失
                    valid_mask = (scores >= 0).float()
                    if valid_mask.sum() > 0:
                        loss = criterion(predicted_scores * valid_mask, scores * valid_mask)
                    else:
                        loss = criterion(predicted_scores, scores)

                    # 反向传播
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item() * vehicle_states.size(0)
                train_samples += vehicle_states.size(0)

                # 更新进度条
                pbar.set_postfix({'loss': loss.item()})

            avg_train_loss = train_loss / train_samples

            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_samples = 0

            with torch.no_grad():
                for vehicle_states, scores in val_loader:
                    vehicle_states = vehicle_states.to(self.device)
                    scores = scores.to(self.device)

                    outputs = self.model(vehicle_states)
                    predicted_scores = outputs['scores'].squeeze(-1)

                    # 计算损失
                    valid_mask = (scores >= 0).float()
                    if valid_mask.sum() > 0:
                        loss = criterion(predicted_scores * valid_mask, scores * valid_mask)
                    else:
                        loss = criterion(predicted_scores, scores)

                    val_loss += loss.item() * vehicle_states.size(0)
                    val_samples += vehicle_states.size(0)

            avg_val_loss = val_loss / val_samples

            # 学习率调度
            scheduler.step(avg_val_loss)

            # TensorBoard记录
            current_lr = optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Stage1/Train_Loss', avg_train_loss, epoch)
            self.writer.add_scalar('Stage1/Val_Loss', avg_val_loss, epoch)
            self.writer.add_scalar('Stage1/Learning_Rate', current_lr, epoch)

            print(f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}")

            # 早停检查
            if avg_val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0

                # 保存最佳模型
                self._save_checkpoint('stage1_best.pth', epoch, optimizer)
                print(f"✓ 保存最佳模型 (val_loss={best_val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n早停触发 (patience={early_stopping_patience})")
                    break

        print("\nStage 1 训练完成!")
        self._save_checkpoint('stage1_final.pth', epochs, optimizer)

    def train_stage2_ppo(self):
        """
        Stage 2: PPO微调（强化学习）

        直接优化OCR奖励
        """
        print(f"\n{'='*60}")
        print("Stage 2: PPO微调（强化学习）")
        print(f"{'='*60}\n")

        # 加载Stage 1的模型
        checkpoint_path = self.checkpoint_dir / 'stage1_best.pth'
        if checkpoint_path.exists():
            print(f"加载Stage 1模型: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("警告：未找到Stage 1模型，使用随机初始化")

        # 创建PPO训练器
        # 这里需要根据实际情况调整
        print("Stage 2: PPO训练（待实现）")
        print("提示：可以使用现有的CustomPPOTrainer类")

        # TODO: 实现PPO微调
        # 1. 创建环境
        # 2. 创建PPO训练器
        # 3. 训练循环
        # 4. 保存模型

    def _prepare_vehicle_tensor(
        self,
        vehicle_states: Dict[str, Dict],
        all_vehicle_ids: List[str]
    ) -> torch.Tensor:
        """将车辆状态字典转换为张量"""
        max_vehicles = len(all_vehicle_ids)
        vehicle_tensor = np.zeros((max_vehicles, 9), dtype=np.float32)

        for i, veh_id in enumerate(all_vehicle_ids):
            if veh_id not in vehicle_states:
                continue

            state = vehicle_states[veh_id]

            vehicle_tensor[i, 0] = state.get('s', 0.0)
            vehicle_tensor[i, 1] = state.get('d', 0.0)
            vehicle_tensor[i, 2] = state.get('vs', 0.0)
            vehicle_tensor[i, 3] = state.get('vd', 0.0)
            vehicle_tensor[i, 4] = state.get('speed', 0.0)
            vehicle_tensor[i, 5] = state.get('acceleration', 0.0)
            vehicle_tensor[i, 6] = state.get('lane_index', 0)
            vehicle_tensor[i, 7] = state.get('angle', 0.0)
            vehicle_tensor[i, 8] = 1.0 if state.get('in_bottleneck', False) else 0.0

        return torch.from_numpy(vehicle_tensor)

    def _prepare_scores_tensor(
        self,
        scores: Dict[str, float],
        all_vehicle_ids: List[str]
    ) -> torch.Tensor:
        """将评分字典转换为张量"""
        max_vehicles = len(all_vehicle_ids)
        scores_tensor = np.zeros((max_vehicles,), dtype=np.float32)

        for i, veh_id in enumerate(all_vehicle_ids):
            if veh_id in scores:
                scores_tensor[i] = scores[veh_id]
            else:
                scores_tensor[i] = -1.0  # 无效标记

        return torch.from_numpy(scores_tensor)

    def _save_checkpoint(self, filename: str, epoch: int, optimizer: optim.Optimizer):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config,
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        print(f"保存检查点: {save_path}")

    def train(self):
        """执行完整的训练流程"""
        print("\n" + "="*60)
        print("ICV评分器训练开始")
        print("="*60 + "\n")

        start_time = time.time()

        # Stage 1: 行为克隆
        num_episodes = self.stage1_config.get('num_episodes', 20)
        num_envs = self.stage1_config.get('num_envs', 8)

        vehicle_states_list, scores_list = self.collect_data_with_rule_scorer(
            num_episodes=num_episodes,
            num_envs=num_envs
        )

        self.train_stage1_behavior_cloning(vehicle_states_list, scores_list)

        # Stage 2: PPO微调（可选）
        if self.config.get('stage2_ppo', {}).get('total_timesteps', 0) > 0:
            self.train_stage2_ppo()

        # 保存最终模型
        if self.model is not None:
            self._save_checkpoint('final.pth', -1, None)

        elapsed_time = time.time() - start_time
        print(f"\n训练完成! 总耗时: {elapsed_time/3600:.2f} 小时")

        self.writer.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练ICV评分器')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/icv_scorer_train.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--stage',
        type=str,
        choices=['stage1', 'stage2', 'all'],
        default='all',
        help='训练阶段'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='恢复训练的检查点路径'
    )

    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建训练器
    trainer = ICVScorerTrainer(config)

    # 训练
    trainer.train()


if __name__ == '__main__':
    main()
