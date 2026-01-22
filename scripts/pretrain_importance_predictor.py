#!/usr/bin/env python3
"""
预训练ImportancePredictor

使用规则选择器收集的数据，通过监督学习训练ImportancePredictor。

输出：
- checkpoints/v5_complete/importance_predictor_pretrained.pth

使用方法：
    python scripts/pretrain_importance_predictor.py --config configs/v5_complete.yaml --data_dir data/icv_selection
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.joint_icv_policy import ImportancePredictor
from src.models.joint_icv_policy import RiskSensitiveGNN


class ICVSelectionDataset(Dataset):
    """ICV选择数据集"""

    def __init__(self, observations_file, selections_file):
        """
        Args:
            observations_file: .npy文件，[N, max_vehicles, 9]
            selections_file: .npy文件，[N, max_vehicles]
        """
        self.observations = np.load(observations_file)  # [N, max_vehicles, 9]
        self.selections = np.load(selections_file)      # [N, max_vehicles]

        assert len(self.observations) == len(self.selections), "数据长度不匹配"

        print(f"[Dataset] 加载数据:")
        print(f"  样本数: {len(self.observations)}")
        print(f"  观测形状: {self.observations.shape}")
        print(f"  选择标签形状: {self.selections.shape}")
        print(f"  正样本比例: {self.selections.mean():.4f}")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.observations[idx]).float()  # [max_vehicles, 9]
        sel = torch.from_numpy(self.selections[idx]).float()    # [max_vehicles]
        return obs, sel


class PretrainedImportancePredictor(nn.Module):
    """预训练的重要性预测器（包含GNN编码器）"""

    def __init__(self, node_dim=9, hidden_dim=128):
        super().__init__()

        # GNN编码器
        self.gnn_encoder = RiskSensitiveGNN(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
            dropout=0.1
        )

        # 重要性预测头
        self.importance_predictor = ImportancePredictor(hidden_dim=hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [B, N, 9] 车辆特征

        Returns:
            importance: [B, N, 1] 重要性分数
            embeddings: [B, N, H] GNN嵌入
        """
        # GNN编码
        outputs = self.gnn_encoder(x)
        embeddings = outputs['embeddings']  # [B, N, H]

        # 重要性预测
        importance = self.importance_predictor(embeddings)  # [B, N, 1]

        return importance, embeddings


def train_one_epoch(model, dataloader, optimizer, device, pos_weight=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for observations, selections in dataloader:
        observations = observations.to(device)  # [B, N, 9]
        selections = selections.to(device)      # [B, N]

        # 前向传播
        importance, _ = model(observations)  # [B, N, 1]

        # 计算损失
        importance_squeezed = importance.squeeze(-1)  # [B, N]
        loss = criterion(importance_squeezed, selections)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 统计
        total_loss += loss.item()

        # 计算准确率（Top-K选择准确率）
        with torch.no_grad():
            batch_size, num_vehicles = selections.shape
            k = int(num_vehicles * 0.10)
            k = max(5, k)

            # 预测的Top-K
            pred_scores = importance_squeezed  # [B, N]
            pred_top_k = torch.topk(pred_scores, k, dim=1).indices  # [B, K]

            # 真实的Top-K
            true_top_k = torch.topk(selections, k, dim=1).indices  # [B, K]

            # 计算重叠率
            for i in range(batch_size):
                pred_set = set(pred_top_k[i].cpu().numpy())
                true_set = set(true_top_k[i].cpu().numpy())
                overlap = len(pred_set & true_set)
                total_correct += overlap
                total_samples += k

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return avg_loss, accuracy


def evaluate(model, dataloader, device, pos_weight=None):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    with torch.no_grad():
        for observations, selections in dataloader:
            observations = observations.to(device)
            selections = selections.to(device)

            importance, _ = model(observations)
            importance_squeezed = importance.squeeze(-1)
            loss = criterion(importance_squeezed, selections)

            total_loss += loss.item()

            # 计算准确率
            batch_size, num_vehicles = selections.shape
            k = int(num_vehicles * 0.10)
            k = max(5, k)

            pred_scores = importance_squeezed
            pred_top_k = torch.topk(pred_scores, k, dim=1).indices
            true_top_k = torch.topk(selections, k, dim=1).indices

            for i in range(batch_size):
                pred_set = set(pred_top_k[i].cpu().numpy())
                true_set = set(true_top_k[i].cpu().numpy())
                overlap = len(pred_set & true_set)
                total_correct += overlap
                total_samples += k

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="预训练ImportancePredictor")
    parser.add_argument('--config', type=str, default='configs/v5_complete.yaml',
                        help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default='data/icv_selection',
                        help='数据目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认使用配置文件中的checkpoint_dir）')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='验证集比例')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("ImportancePredictor预训练")
    print("=" * 80)
    print(f"\n配置文件: {args.config}")
    print(f"数据目录: {args.data_dir}")
    print(f"训练轮数: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"学习率: {args.lr}")

    # 加载元数据
    metadata_file = Path(args.data_dir) / 'metadata.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    max_vehicles = metadata['max_vehicles']
    node_dim = metadata['node_dim']
    hidden_dim = config['policy']['hidden_dim']

    print(f"\n数据元信息:")
    print(f"  样本数: {metadata['num_samples']}")
    print(f"  最大车辆数: {max_vehicles}")
    print(f"  特征维度: {node_dim}")
    print(f"  隐藏层维度: {hidden_dim}")

    # 创建数据集
    print("\n加载数据集...")
    dataset = ICVSelectionDataset(
        Path(args.data_dir) / 'observations.npy',
        Path(args.data_dir) / 'selections.npy'
    )

    # 划分训练集和验证集
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"  训练集: {train_size} 样本")
    print(f"  验证集: {val_size} 样本")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # 创建模型
    print("\n创建模型...")
    model = PretrainedImportancePredictor(
        node_dim=node_dim,
        hidden_dim=hidden_dim
    ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {num_params:,}")

    # 计算正样本权重（处理类别不平衡）
    pos_weight = None
    if dataset.selections.mean() < 0.2:  # 如果正样本比例<20%
        neg_ratio = (1.0 - dataset.selections.mean()) / dataset.selections.mean()
        pos_weight = torch.tensor([neg_ratio]).to(args.device)
        print(f"  正样本权重: {pos_weight.item():.2f}")

    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5
    )

    # 训练
    print("\n开始训练...")
    print("=" * 80)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, args.device, pos_weight
        )

        # 验证
        val_loss, val_acc = evaluate(
            model, val_loader, args.device, pos_weight
        )

        # 更新学习率
        scheduler.step()

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # 确定输出目录
            if args.output_dir is None:
                output_dir = Path(config['global']['checkpoint_dir'])
            else:
                output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }

            checkpoint_path = output_dir / 'importance_predictor_pretrained.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✅ 最佳模型已保存: {checkpoint_path}")

    print("\n" + "=" * 80)
    print("🎉 预训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
