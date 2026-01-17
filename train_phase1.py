"""
Phase 1: 世界模型训练（感知层 + 预测层）

训练目标：
- 学习车辆状态编码器（GNN）
- 学习交通流预测器（World Model）
- 学习Frenet坐标系转换

使用方法：
    python train_phase1.py
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.world_model_train_v4 import WorldModelTrainer
from src.utils.helpers import get_device, set_seed


def load_config(config_path: str = 'configs/competition.yaml') -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    print("=" * 80)
    print("[PHASE 1] World Model Training")
    print("=" * 80)

    # 加载配置
    config = load_config()

    # 设备配置
    device = get_device(prefer_cuda=True, cuda_id=0)

    # 设置随机种子
    seed = config.get('seed', 42)
    set_seed(seed)

    # Phase 1配置
    phase1_config = config.get('training', {}).get('phase1', {})

    print(f"\n[CONFIG] Phase 1 Configuration:")
    print(f"  - Device: {device}")
    print(f"  - Seed: {seed}")
    print(f"  - Total epochs: {phase1_config.get('num_epochs', 100)}")
    print(f"  - Batch size: {phase1_config.get('batch_size', 32)}")
    print(f"  - Learning rate: {phase1_config.get('learning_rate', 1e-3)}")

    # 创建训练器
    trainer = WorldModelTrainer(
        config=config,
        device=device,
    )

    # 开始训练
    print("\n" + "=" * 80)
    print("[TRAIN] Starting Phase 1 Training...")
    print("=" * 80)

    trainer.train()

    # 保存最终模型
    checkpoint_dir = Path(config.get('paths', {}).get('checkpoint_dir', 'checkpoints/competition'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    final_checkpoint = checkpoint_dir / 'phase1' / 'world_model_final.pth'

    trainer.save_checkpoint(str(final_checkpoint))

    print("\n" + "=" * 80)
    print("[DONE] Phase 1 Training Completed!")
    print(f"[SAVE] Final model saved to: {final_checkpoint}")
    print("=" * 80)


if __name__ == '__main__':
    main()
