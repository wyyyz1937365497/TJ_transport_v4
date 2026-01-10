"""训练器封装 - 完整实现，集成所有训练阶段"""

from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
import torch

# 兼容不同版本的PyTorch Lightning
try:
    from pytorch_lightning.loggers.wandb import WandBLogger
except ImportError:
    try:
        from pytorch_lightning.loggers import WandBLogger
    except ImportError:
        WandBLogger = None

from ..models import TrafficController
from ..env import TrafficDataModule, collect_data
from ..utils import Config, setup_logger, get_logger
from ..evaluation import Evaluator, generate_xlsx_report
from .lightning_module import TrafficLightningModule
from .ppo import PPOTrainer
from .finetune import EndToEndFineTuner
from .constrained import ConstrainedOptimizer

logger = get_logger()


class Trainer:
    """
    训练器 - 完整实现

    支持多阶段训练：
    - Phase 1: 世界模型预训练
    - Phase 2: PPO训练
    - Phase 3: 端到端微调
    - Phase 4: 约束优化
    """

    def __init__(self, config: Config):
        self.config = config

        # 设备
        self.device = self._get_device()

        # 创建模型
        self.model = self._create_model()

        # 设置日志
        self.logger = setup_logger(
            log_file=config.log_dir / "training.log",
        )

    def _get_device(self) -> str:
        """获取设备"""
        device_str = self.config.device
        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA不可用，使用CPU")
            device_str = "cpu"
        return device_str

    def _create_model(self) -> TrafficController:
        """创建模型"""
        model = TrafficController(self.config.model)

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"模型已创建")
        logger.info(f"  总参数: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")

        return model

    def train_phase1(
        self,
        skip_data_collection: bool = False,
    ) -> TrafficController:
        """
        阶段1：世界模型预训练

        Args:
            skip_data_collection: 是否跳过数据收集
        """
        logger.info("=" * 70)
        logger.info("阶段1：世界模型预训练")
        logger.info("=" * 70)

        phase1_cfg = self.config.training.get("phase1", {})

        # 1. 收集数据
        if not skip_data_collection:
            logger.info("收集训练数据...")
            collect_data(
                config=self.config.environment,
                num_episodes=phase1_cfg.get("num_episodes", 5),
                max_steps=self.config.environment.get("max_steps", 3600),
                output_dir=self.config.data_dir,
            )

        # 2. 创建数据模块
        data_module = TrafficDataModule(
            config=self.config.to_dict(),
            data_dir=self.config.data_dir,
        )
        data_module.setup()

        if data_module.train_dataloader() is None:
            logger.error("数据加载失败，跳过训练")
            return self.model

        # 3. 创建Lightning模块
        lightning_module = TrafficLightningModule(
            model=self.model,
            config=self.config.to_dict(),
            phase="world_model",
        )

        # 4. 配置日志
        pl_logger = self._setup_logger("phase1")

        # 5. 创建Trainer
        trainer = pl.Trainer(
            max_epochs=phase1_cfg.get("epochs", 10),
            accelerator=self.device,
            devices=1,
            precision=self.config.precision,
            logger=pl_logger,
            callbacks=[
                RichProgressBar(),
            ],
            log_every_n_steps=self.config.logging.get("log_every_n_steps", 10),
            gradient_clip_val=phase1_cfg.get("gradient_clip", 1.0),
            gradient_clip_algorithm="norm",
        )

        # 6. 训练
        logger.info("开始训练...")
        trainer.fit(lightning_module, datamodule=data_module)

        # 7. 保存模型
        checkpoint_path = self.config.checkpoint_dir / "world_model_phase1.ckpt"
        trainer.save_checkpoint(str(checkpoint_path))
        logger.info(f"模型已保存: {checkpoint_path}")

        return self.model

    def train_phase2(self) -> TrafficController:
        """
        阶段2：PPO训练

        使用PPO算法训练控制器，同时冻结GNN和世界模型
        """
        logger.info("=" * 70)
        logger.info("阶段2：PPO训练（带安全屏障的RL）")
        logger.info("=" * 70)

        # 加载阶段1权重
        phase1_ckpt = self.config.checkpoint_dir / "world_model_phase1.ckpt"
        if phase1_ckpt.exists():
            self._load_checkpoint(phase1_ckpt)
            logger.info("已加载阶段1权重")

        # 创建PPO训练器
        ppo_trainer = PPOTrainer(
            model=self.model,
            config=self.config,
        )

        # 训练
        self.model = ppo_trainer.train()

        return self.model

    def train_phase3(self) -> TrafficController:
        """
        阶段3：端到端微调

        联合优化所有组件（GNN + 世界模型 + 控制器）
        """
        logger.info("=" * 70)
        logger.info("阶段3：端到端微调（所有组件联合优化）")
        logger.info("=" * 70)

        # 加载阶段2权重
        phase2_ckpt = self.config.checkpoint_dir / "ppo_phase2.ckpt"
        if phase2_ckpt.exists():
            self._load_checkpoint(phase2_ckpt)
            logger.info("已加载阶段2权重")

        # 创建端到端微调器
        finetuner = EndToEndFineTuner(
            model=self.model,
            config=self.config,
        )

        # 训练
        self.model = finetuner.train()

        return self.model

    def train_phase4(self) -> TrafficController:
        """
        阶段4：约束优化

        使用拉格朗日乘子法平衡性能与成本
        """
        logger.info("=" * 70)
        logger.info("阶段4：约束优化训练（拉格朗日乘子法）")
        logger.info("=" * 70)

        # 加载阶段3权重
        phase3_ckpt = self.config.checkpoint_dir / "finetune_phase3.ckpt"
        if phase3_ckpt.exists():
            self._load_checkpoint(phase3_ckpt)
            logger.info("已加载阶段3权重")

        # 创建约束优化器
        constrained_optimizer = ConstrainedOptimizer(
            model=self.model,
            config=self.config,
        )

        # 训练
        self.model = constrained_optimizer.train()

        return self.model

    def train_all(
        self,
        skip_data_collection: bool = False,
    ) -> TrafficController:
        """
        完整训练流程（所有阶段）

        Args:
            skip_data_collection: 是否跳过数据收集
        """
        logger.info("=" * 70)
        logger.info("开始完整训练流程")
        logger.info("=" * 70)

        # 阶段1
        if self.config.training.get("phase1", {}).get("enabled", True):
            self.model = self.train_phase1(skip_data_collection)

        # 阶段2
        if self.config.training.get("phase2", {}).get("enabled", True):
            self.model = self.train_phase2()

        # 阶段3
        if self.config.training.get("phase3", {}).get("enabled", True):
            self.model = self.train_phase3()

        # 阶段4
        if self.config.training.get("phase4", {}).get("enabled", True):
            self.model = self.train_phase4()

        logger.info("=" * 70)
        logger.info("训练完成!")
        logger.info("=" * 70)

        # 保存最终模型
        self._save_final_model()

        return self.model

    def evaluate(
        self,
        num_episodes: Optional[int] = None,
        generate_xlsx: bool = False,
    ) -> Dict[str, Any]:
        """
        评估模型

        Args:
            num_episodes: 评估episodes数量
            generate_xlsx: 是否生成XLSX文件

        Returns:
            评估指标
        """
        logger.info("=" * 70)
        logger.info("开始模型评估")
        logger.info("=" * 70)

        if num_episodes is None:
            num_episodes = self.config.evaluation.get("num_episodes", 5)

        # 创建评估器
        evaluator = Evaluator(
            model=self.model,
            config=self.config,
        )

        # 评估
        metrics = evaluator.evaluate(verbose=True)

        # 生成XLSX
        if generate_xlsx:
            logger.info("生成XLSX结果文件...")
            from ..evaluation.xlsx_generator import XLSXResultGenerator

            generator = XLSXResultGenerator(self.model, self.config)
            output_path = generator.generate(
                num_episodes=num_episodes,
                verbose=True,
            )
            logger.info(f"XLSX文件已生成: {output_path}")

        return metrics

    def _setup_logger(self, phase: str) -> Optional[WandBLogger]:
        """设置日志记录器"""
        if not self.config.wandb.get("enabled", False):
            return None

        return WandBLogger(
            project=self.config.wandb.get("project", "tj-transport"),
            entity=self.config.wandb.get("entity"),
            name=self.config.wandb.get("run_name", f"{phase}_experiment"),
            save_dir=self.config.log_dir,
        )

    def _load_checkpoint(self, checkpoint_path: Path):
        """加载检查点"""
        if not checkpoint_path.exists():
            logger.warning(f"检查点不存在: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 尝试加载模型状态
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            # PyTorch Lightning格式
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            logger.warning("无法识别检查点格式")

        logger.info(f"已加载检查点: {checkpoint_path}")

    def _save_final_model(self):
        """保存最终模型"""
        checkpoint_path = self.config.checkpoint_dir / "final_model.pth"

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
        }, checkpoint_path)

        logger.info(f"最终模型已保存: {checkpoint_path}")
