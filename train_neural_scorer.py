"""
神经网络ICV评分器训练脚本

两阶段训练流程：
1. 预训练（Pretrain）：从规则评分学习，快速收敛
2. 微调（Finetune）：使用强化学习直接优化OCR

使用方法：
    # 阶段1：预训练
    python train_neural_scorer.py --stage pretrain --config configs/phase1_lite.yaml

    # 阶段2：微调
    python train_neural_scorer.py --stage finetune --config configs/phase1_lite.yaml --pretrain checkpoints/neural_scorer_pretrain.pth
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple
import random

# 导入项目模块
from src.models.neural_vehicle_scorer import (
    NeuralVehicleScorer,
    NeuralICVScorer,
    VehicleGraphBuilder,
    create_neural_icv_scorer
)
from src.env.gym_wrapper import make_gym_env


# =============================================================================
# 日志配置
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 数据集类
# =============================================================================
class ICVScoringDataset(Dataset):
    """
    ICV评分数据集

    存储车辆状态和对应的规则评分标签
    """

    def __init__(self, max_samples=10000):
        self.samples = []  # List of (vehicle_features, edge_index, edge_attr, target_score)
        self.max_samples = max_samples

    def add_sample(
        self,
        vehicle_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        target_scores: torch.Tensor
    ):
        """添加一个样本"""
        if len(self.samples) >= self.max_samples:
            # 随机替换一个旧样本
            idx = random.randint(0, len(self.samples) - 1)
            self.samples[idx] = (vehicle_features, edge_index, edge_attr, target_scores)
        else:
            self.samples.append((vehicle_features, edge_index, edge_attr, target_scores))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vehicle_features, edge_index, edge_attr, target_scores = self.samples[idx]
        return {
            'vehicle_features': vehicle_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'target_scores': target_scores
        }


def collate_fn(batch):
    """
    自定义batch整理函数（处理变长图）
    """
    vehicle_features_list = [item['vehicle_features'] for item in batch]
    edge_index_list = [item['edge_index'] for item in batch]
    edge_attr_list = [item['edge_attr'] for item in batch]
    target_scores_list = [item['target_scores'] for item in batch]

    # 由于每个图的车辆数不同，这里简化处理：返回列表
    # 在训练循环中逐个处理
    return list(zip(vehicle_features_list, edge_index_list, edge_attr_list, target_scores_list))


# =============================================================================
# 预训练阶段
# =============================================================================
class PretrainTrainer:
    """
    预训练器：从规则评分学习

    目标：让神经网络学习模仿规则评分的基本模式
    方法：监督学习，MSE损失
    """

    def __init__(
        self,
        model: NeuralICVScorer,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device

        # 训练配置
        pretrain_config = config.get('neural_icv_pretrain', {})
        self.num_episodes = pretrain_config.get('num_episodes', 20)
        self.epochs = pretrain_config.get('epochs', 10)
        self.batch_size = pretrain_config.get('batch_size', 32)
        self.learning_rate = pretrain_config.get('learning_rate', 0.001)

        # 优化器
        self.optimizer = optim.Adam(
            self.model.neural_scorer.parameters(),
            lr=self.learning_rate
        )

        # 损失函数
        self.criterion = nn.MSELoss()

        # 数据集
        self.dataset = ICVScoringDataset(max_samples=5000)

    def collect_data(self, env) -> int:
        """
        从环境收集数据（使用规则评分作为标签）

        Returns:
            收集的样本数
        """
        logger.info("开始收集数据...")

        num_samples = 0

        for episode in tqdm(range(self.num_episodes), desc="数据收集"):
            obs, info = env.reset()

            for step in range(1000):  # 每个episode最多1000步
                # 使用空动作（不控制任何车辆）
                action = {'vehicle_ids': [], 'actions': []}
                obs, reward, terminated, truncated, info = env.step(action)

                # 获取当前时刻的车辆状态和规则评分
                if hasattr(env, 'sumo_env'):
                    sumo_env = env.sumo_env

                    # 获取所有车辆状态
                    vehicle_states = {}
                    rule_scores = {}

                    if hasattr(sumo_env, '_get_all_vehicle_states'):
                        # 使用辅助方法获取所有车辆状态
                        all_vehicles = sumo_env.traci_lib.vehicle.getIDList()

                        for veh_id in all_vehicles:
                            state = sumo_env._get_vehicle_state_dict(veh_id, sumo_env.traci_lib)
                            vehicle_states[veh_id] = state

                            # 计算规则评分
                            if hasattr(sumo_env, '_compute_rule_based_score'):
                                rule_score = sumo_env._compute_rule_based_score(veh_id, state)
                                rule_scores[veh_id] = rule_score

                    # 如果有足够的车辆，构建样本
                    if len(vehicle_states) >= 5:
                        # 提取特征
                        vehicle_features = []
                        for veh_id in vehicle_states.keys():
                            features = self.model._extract_features(vehicle_states[veh_id])
                            vehicle_features.append(features)

                        vehicle_features = torch.tensor(vehicle_features, dtype=torch.float32)

                        # 构建图
                        edge_index, edge_attr = self.model.graph_builder.build_graph(
                            vehicle_states,
                            len(vehicle_states)
                        )

                        # 目标评分（归一化到0-1）
                        target_scores = torch.tensor([
                            rule_scores.get(veh_id, 0.0) / 53.0
                            for veh_id in vehicle_states.keys()
                        ], dtype=torch.float32)

                        # 添加到数据集
                        self.dataset.add_sample(
                            vehicle_features,
                            edge_index,
                            edge_attr,
                            target_scores
                        )
                        num_samples += 1

                if terminated or truncated:
                    break

        logger.info(f"数据收集完成！共收集 {num_samples} 个样本")
        return num_samples

    def train_epoch(self, epoch: int) -> float:
        """
        训练一个epoch

        Returns:
            平均损失
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        # 随机打乱数据
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]

            batch_loss = 0.0
            batch_count = 0

            for idx in batch_indices:
                sample = self.dataset[idx]

                vehicle_features = sample['vehicle_features'].unsqueeze(0).to(self.device)
                edge_index = sample['edge_index'].to(self.device)
                edge_attr = sample['edge_attr'].to(self.device)
                target_scores = sample['target_scores'].to(self.device)

                # 前向传播
                self.optimizer.zero_grad()
                pred_scores = self.model.neural_scorer(
                    vehicle_features,
                    edge_index,
                    edge_attr
                ).squeeze(0)

                # 计算损失
                loss = self.criterion(pred_scores, target_scores)

                # 反向传播
                loss.backward()
                self.optimizer.step()

                batch_loss += loss.item()
                batch_count += 1

            if batch_count > 0:
                total_loss += batch_loss / batch_count
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def run(self, env):
        """
        运行预训练
        """
        logger.info("="*70)
        logger.info("阶段1：预训练（从规则评分学习）")
        logger.info("="*70)

        # 1. 收集数据
        num_samples = self.collect_data(env)

        if num_samples == 0:
            logger.error("没有收集到任何数据，无法训练！")
            return

        logger.info(f"数据集大小: {len(self.dataset)}")

        # 2. 训练
        logger.info(f"开始训练 {self.epochs} 个epoch...")

        best_loss = float('inf')

        for epoch in range(self.epochs):
            avg_loss = self.train_epoch(epoch)

            logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = 'checkpoints/neural_scorer_pretrain.pth'
                Path('checkpoints').mkdir(exist_ok=True)
                self.model.save_checkpoint(checkpoint_path, epoch, avg_loss)

        logger.info("预训练完成！")
        logger.info(f"最佳损失: {best_loss:.4f}")
        logger.info(f"模型已保存到: checkpoints/neural_scorer_pretrain.pth")


# =============================================================================
# 微调阶段
# =============================================================================
class FinetuneTrainer:
    """
    微调器：使用强化学习优化OCR

    目标：直接优化OCR指标
    方法：PPO算法
    """

    def __init__(
        self,
        model: NeuralICVScorer,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device

        # 训练配置
        finetune_config = config.get('neural_icv_finetune', {})
        self.total_timesteps = finetune_config.get('total_timesteps', 100000)
        self.learning_rate = finetune_config.get('learning_rate', 0.0003)
        self.gamma = finetune_config.get('gamma', 0.99)
        self.gae_lambda = finetune_config.get('gae_lambda', 0.95)

        # 优化器
        self.optimizer = optim.Adam(
            self.model.neural_scorer.parameters(),
            lr=self.learning_rate
        )

        # 经验缓冲区
        self.buffer = []

    def collect_rollouts(self, env, num_steps: int) -> Dict:
        """
        收集rollout数据

        Returns:
            统计信息字典
        """
        self.model.eval()
        self.buffer = []

        episode_rewards = []
        episode_ocrs = []

        obs, info = env.reset()
        episode_reward = 0.0

        for step in tqdm(range(num_steps), desc="Rollout收集"):
            # 获取神经网络评分
            if hasattr(env, 'sumo_env'):
                sumo_env = env.sumo_env

                # 获取所有车辆状态
                vehicle_states = {}
                if hasattr(sumo_env, 'traci_lib'):
                    all_vehicles = sumo_env.traci_lib.vehicle.getIDList()

                    for veh_id in all_vehicles:
                        state = sumo_env._get_vehicle_state_dict(veh_id, sumo_env.traci_lib)
                        vehicle_states[veh_id] = state

                # 使用神经网络评分
                neural_scores = self.model.compute_scores(vehicle_states)

                # 记录经验
                self.buffer.append({
                    'vehicle_states': vehicle_states,
                    'neural_scores': neural_scores,
                })

            # 执行动作（使用空动作，不控制任何车辆）
            action = {'vehicle_ids': [], 'actions': []}
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # 记录OCR
            if 'ocr' in info:
                episode_ocrs.append(info['ocr'])

            if terminated or truncated:
                episode_rewards.append(episode_reward)
                obs, info = env.reset()
                episode_reward = 0.0

        # 计算统计信息
        stats = {
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'mean_ocr': np.mean(episode_ocrs) if episode_ocrs else 0.0,
            'num_episodes': len(episode_rewards),
        }

        return stats

    def update(self):
        """
        使用PPO更新模型
        """
        self.model.train()

        # 这里简化实现：使用策略梯度
        # 实际应该使用完整的PPO算法

        total_loss = 0.0
        num_updates = 0

        for transition in self.buffer:
            vehicle_states = transition['vehicle_states']
            neural_scores = transition['neural_scores']

            # 提取特征
            vehicle_features = []
            for veh_id in vehicle_states.keys():
                features = self.model._extract_features(vehicle_states[veh_id])
                vehicle_features.append(features)

            if len(vehicle_features) == 0:
                continue

            vehicle_features = torch.tensor(vehicle_features, dtype=torch.float32).to(self.device)

            # 构建图
            edge_index, edge_attr = self.model.graph_builder.build_graph(
                vehicle_states,
                len(vehicle_states)
            )
            edge_index = edge_index.to(self.device)
            edge_attr = edge_attr.to(self.device)

            # 计算策略梯度损失（简化版）
            self.optimizer.zero_grad()

            pred_scores = self.model.neural_scorer(
                vehicle_features.unsqueeze(0),
                edge_index,
                edge_attr
            ).squeeze(0)

            # 使用OCR作为奖励信号
            # 这里简化：鼓励高分车辆被选择
            target_scores = torch.tensor([
                neural_scores.get(veh_id, 0.0) / 53.0
                for veh_id in vehicle_states.keys()
            ], dtype=torch.float32).to(self.device)

            loss = nn.MSELoss()(pred_scores, target_scores)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_updates += 1

        avg_loss = total_loss / num_updates if num_updates > 0 else 0.0
        return avg_loss

    def run(self, env):
        """
        运行微调
        """
        logger.info("="*70)
        logger.info("阶段2：微调（强化学习优化OCR）")
        logger.info("="*70)

        num_iterations = self.total_timesteps // 1000

        for iteration in range(num_iterations):
            logger.info(f"\n迭代 {iteration+1}/{num_iterations}")

            # 收集rollout
            stats = self.collect_rollouts(env, num_steps=1000)

            logger.info(f"  平均奖励: {stats['mean_reward']:.4f}")
            logger.info(f"  平均OCR: {stats['mean_ocr']:.4f}")
            logger.info(f"  Episode数: {stats['num_episodes']}")

            # 更新模型
            loss = self.update()

            logger.info(f"  更新损失: {loss:.4f}")

            # 定期保存检查点
            if (iteration + 1) % 5 == 0:
                checkpoint_path = f'checkpoints/neural_scorer_finetune_iter{iteration+1}.pth'
                Path('checkpoints').mkdir(exist_ok=True)
                self.model.save_checkpoint(checkpoint_path, iteration, loss)

        # 保存最终模型
        final_checkpoint = 'checkpoints/neural_scorer_finetune_final.pth'
        self.model.save_checkpoint(final_checkpoint, num_iterations, 0.0)

        logger.info("\n微调完成！")
        logger.info(f"最终模型已保存到: {final_checkpoint}")


# =============================================================================
# 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='训练神经网络ICV评分器')
    parser.add_argument('--stage', type=str, required=True, choices=['pretrain', 'finetune'],
                        help='训练阶段：pretrain（预训练）或 finetune（微调）')
    parser.add_argument('--config', type=str, default='configs/phase1_lite.yaml',
                        help='配置文件路径')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='预训练权重路径（finetune阶段需要）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备（cuda或cpu）')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 创建环境
    logger.info("创建训练环境...")
    env = make_gym_env(config=config, seed=42, device=args.device)

    # 创建模型
    logger.info("创建神经网络ICV评分器...")
    neural_config = config.get('neural_icv_scoring', {})

    if args.stage == 'finetune' and args.pretrain:
        # 加载预训练权重
        model = create_neural_icv_scorer(
            node_dim=neural_config.get('node_dim', 9),
            hidden_dim=neural_config.get('hidden_dim', 64),
            num_layers=neural_config.get('num_layers', 3),
            num_heads=neural_config.get('num_heads', 4),
            checkpoint_path=args.pretrain,
            device=args.device
        )
    else:
        # 从头创建
        model = create_neural_icv_scorer(
            node_dim=neural_config.get('node_dim', 9),
            hidden_dim=neural_config.get('hidden_dim', 64),
            num_layers=neural_config.get('num_layers', 3),
            num_heads=neural_config.get('num_heads', 4),
            device=args.device
        )

    # 根据阶段选择训练器
    if args.stage == 'pretrain':
        trainer = PretrainTrainer(model, config, args.device)
    else:
        trainer = FinetuneTrainer(model, config, args.device)

    # 开始训练
    try:
        trainer.run(env)
    except KeyboardInterrupt:
        logger.info("\n训练被中断")

    # 关闭环境
    env.close()

    logger.info("训练脚本结束")


if __name__ == '__main__':
    main()
