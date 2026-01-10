"""PyTorch Lightning训练模块"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT

from ..models import TrafficController
from ..utils.logging import get_logger

logger = get_logger()


class TrafficLightningModule(LightningModule):
    """交通控制器的PyTorch Lightning模块"""

    def __init__(
        self,
        model: TrafficController,
        config: Dict[str, Any],
        phase: str = "world_model",
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.phase = phase

        # 保存超参数
        self.save_hyperparameters({
            "phase": phase,
            "config": config,
        })

        # 训练阶段配置
        phase_cfg = config.get("training", {}).get(f"phase{self._phase_number()}", {})

        self.learning_rate = phase_cfg.get("learning_rate", 1e-4)
        self.weight_decay = phase_cfg.get("weight_decay", 1e-5)
        self.gradient_clip = phase_cfg.get("gradient_clip", 1.0)

        # 混合精度
        self.use_amp = phase_cfg.get("use_amp", True)

    def _phase_number(self) -> int:
        """获取阶段编号"""
        phase_map = {
            "world_model": 1,
            "ppo": 2,
            "finetune": 3,
            "constrained": 4,
        }
        return phase_map.get(self.phase, 1)

    def configure_optimizers(self):
        """配置优化器"""
        if self.phase == "world_model":
            # 只训练世界模型
            params = self.model.world_model.parameters()
        elif self.phase == "ppo":
            # 只训练控制器
            params = self.model.controller.parameters()
        elif self.phase == "finetune":
            # 训练所有组件
            params = self.model.parameters()
        else:  # constrained
            # 只训练控制器
            params = self.model.controller.parameters()

        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # 学习率调度器
        if self.phase == "world_model":
            warmup_epochs = self.config.get("training", {}).get("phase1", {}).get("warmup_epochs", 2)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=0.1,
                        total_iters=warmup_epochs,
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=self.trainer.max_epochs - warmup_epochs,
                        eta_min=self.learning_rate * 0.01,
                    ),
                ],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.learning_rate * 0.01,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """前向传播"""
        return self.model(batch)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        """训练步骤"""
        # 传输到设备
        current_states = batch["current"].to(self.device)
        future_states = batch["future"].to(self.device)
        graph_data = batch["graph_data"].to(self.device)

        B, T, _ = current_states.shape

        # 通过GNN
        gnn_output = self.model.risk_gnn(
            node_features=graph_data.x,
            edge_index=graph_data.edge_index,
            edge_features=graph_data.edge_attr,
            batch=graph_data.batch,
        )

        gnn_embedding = gnn_output["node_embedding"]

        # 通过世界模型
        predictions = self.model.world_model(gnn_embedding)
        next_state_pred = predictions["next_state"]

        # 准备目标
        future_state = future_states[:, 0, :]  # [B, 3]

        # 投影到目标维度
        if next_state_pred.dim() == 2 and next_state_pred.size(1) == 256:
            if not hasattr(self, "state_projection"):
                self.state_projection = nn.Linear(256, 3).to(self.device)
            pred_state = self.state_projection(next_state_pred)
        else:
            pred_state = next_state_pred

        # 计算损失
        loss = F.mse_loss(pred_state, future_state)

        # 日志
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Optional[STEP_OUTPUT]:
        """验证步骤"""
        current_states = batch["current"].to(self.device)
        future_states = batch["future"].to(self.device)
        graph_data = batch["graph_data"].to(self.device)

        with torch.no_grad():
            gnn_output = self.model.risk_gnn(
                node_features=graph_data.x,
                edge_index=graph_data.edge_index,
                edge_features=graph_data.edge_attr,
                batch=graph_data.batch,
            )

            gnn_embedding = gnn_output["node_embedding"]
            predictions = self.model.world_model(gnn_embedding)
            next_state_pred = predictions["next_state"]

            future_state = future_states[:, 0, :]

            if hasattr(self, "state_projection"):
                pred_state = self.state_projection(next_state_pred)
            else:
                pred_state = next_state_pred

            loss = F.mse_loss(pred_state, future_state)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        """训练epoch结束"""
        train_loss = self.trainer.callback_metrics.get("train_loss_epoch")
        val_loss = self.trainer.callback_metrics.get("val_loss")

        if train_loss is not None:
            logger.info(f"Epoch {self.current_epoch}: Train Loss = {train_loss:.6f}")

        if val_loss is not None:
            logger.info(f"Epoch {self.current_epoch}: Val Loss = {val_loss:.6f}")

    def configure_callbacks(self):
        """配置回调"""
        callbacks = []

        # 模型检查点
        from pytorch_lightning.callbacks import ModelCheckpoint

        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=3,
            mode="min",
            filename="model-{epoch:02d}-{val_loss:.6f}",
        )
        callbacks.append(checkpoint)

        # 早停
        from pytorch_lightning.callbacks import EarlyStopping

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
        )
        callbacks.append(early_stop)

        # 学习率监控
        from pytorch_lightning.callbacks import LearningRateMonitor

        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

        return callbacks
