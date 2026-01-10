#!/usr/bin/env python3
"""
TJ_transport_v4 - WSL优化版本
主训练脚本

使用方法:
    python train.py                        # 完整训练流程（所有阶段）
    python train.py --config xxx.yaml      # 指定配置文件
    python train.py --phase world_model    # 指定训练阶段
    python train.py --eval-only            # 仅评估
    python train.py --eval-only --generate-xlsx  # 评估并生成XLSX
"""

import argparse
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import load_config, setup_logger, set_seed
from src.training import Trainer
from src.models import TrafficController


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="TJ_transport_v4 - WSL优化版本训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 配置文件
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (默认: config/base.yaml)",
    )

    # 训练阶段
    parser.add_argument(
        "-p",
        "--phase",
        type=str,
        default="all",
        choices=["all", "world_model", "ppo", "finetune", "constrained"],
        help="训练阶段 (默认: all)",
    )

    # 仅评估
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="仅评估模式",
    )

    # 生成XLSX
    parser.add_argument(
        "--generate-xlsx",
        action="store_true",
        help="生成XLSX结果文件",
    )

    # 评估episodes数量
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="评估episodes数量",
    )

    # 跳过数据收集
    parser.add_argument(
        "--skip-data-collection",
        action="store_true",
        help="跳过SUMO数据收集，使用已生成的数据",
    )

    # 其他选项
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子 (覆盖配置文件)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="设备 (覆盖配置文件)",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 打印欢迎信息
    print("=" * 70)
    print("TJ_transport_v4 - WSL优化版本")
    print("=" * 70)

    # 加载配置
    config_path = args.config
    config = load_config(config_path)

    # 应用命令行覆盖
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device is not None:
        overrides["device"] = args.device

    if overrides:
        from src.utils.config import merge_configs
        override_config = Config.from_dict(overrides)
        config = merge_configs(config, override_config)

    # 设置日志
    logger = setup_logger(
        log_file=config.log_dir / "main.log",
    )

    logger.info(f"配置文件: {config_path or 'config/base.yaml'}")
    logger.info(f"设备: {config.device}")
    logger.info(f"种子: {config.seed}")

    # 设置随机种子
    set_seed(config.seed)

    # 创建训练器
    trainer = Trainer(config)

    # 仅评估模式
    if args.eval_only:
        logger.info("评估模式")

        # 加载最终模型
        final_model_path = config.checkpoint_dir / "final_model.pth"
        if final_model_path.exists():
            import torch
            checkpoint = torch.load(final_model_path, map_location=config.device)
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("已加载最终模型")
        else:
            logger.warning("最终模型不存在，将使用当前模型")

        # 评估
        metrics = trainer.evaluate(
            num_episodes=args.num_episodes,
            generate_xlsx=args.generate_xlsx,
        )

        # 打印结果摘要
        from src.evaluation.metrics import print_metrics_report
        print_metrics_report(metrics, "评估结果摘要")

        return

    # 根据阶段训练
    if args.phase == "all":
        trainer.train_all(skip_data_collection=args.skip_data_collection)

        # 训练完成后自动评估
        logger.info("\n开始最终评估...")
        metrics = trainer.evaluate(
            num_episodes=args.num_episodes,
            generate_xlsx=args.generate_xlsx,
        )

        from src.evaluation.metrics import print_metrics_report
        print_metrics_report(metrics, "最终评估结果")

    elif args.phase == "world_model":
        trainer.train_phase1(skip_data_collection=args.skip_data_collection)

    elif args.phase == "ppo":
        trainer.train_phase2()

    elif args.phase == "finetune":
        trainer.train_phase3()

    elif args.phase == "constrained":
        trainer.train_phase4()

    logger.info("完成!")


if __name__ == "__main__":
    main()
