#!/usr/bin/env python3
"""
Imitation Learning Training Script (Behavioral Cloning)

This script trains a policy network through behavioral cloning,
using expert demonstrations collected from rule-based policies.

Key Features:
1. MSE loss for behavioral cloning
2. Learning rate scheduling (ReduceLROnPlateau)
3. Early stopping based on validation loss
4. Checkpoint management (save best and last)
5. Comprehensive logging and metrics

Usage:
    # Basic training
    python scripts/train_imitation_learning.py \
        --config configs/imitation_learning.yaml \
        --demo_data data/demonstrations/expert/demonstrations.pkl

    # Resume from checkpoint
    python scripts/train_imitation_learning.py \
        --config configs/imitation_learning.yaml \
        --demo_data data/demonstrations/expert/demonstrations.pkl \
        --resume checkpoints/imitation_learning/checkpoint_epoch_30.pth

Output:
    - checkpoints/imitation_learning/: Model checkpoints
    - logs/imitation_learning/: Training logs and TensorBoard events
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.imitation_dataset import create_dataloaders
from src.models.simplified_icv_policy import SimplifiedICVPolicy


class ImitationLearningTrainer:
    """
    Behavioral Cloning Trainer for Imitation Learning

    This trainer implements supervised learning from demonstrations:
    - Loss: MSE between predicted and expert actions
    - Optimizer: Adam with learning rate scheduling
    - Early Stopping: Based on validation loss
    """

    def __init__(
        self,
        policy: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = 'cuda',
        log_dir: str = 'logs/imitation_learning',
        checkpoint_dir: str = 'checkpoints/imitation_learning'
    ):
        """
        Initialize trainer

        Args:
            policy: Policy network to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on ('cuda' or 'cpu')
            log_dir: Directory for logs
            checkpoint_dir: Directory for checkpoints
        """
        self.policy = policy.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Training hyperparameters
        training_config = config.get('training', {})
        self.num_epochs = training_config.get('num_epochs', 50)
        self.lr = training_config.get('lr', 1e-3)
        self.weight_decay = training_config.get('weight_decay', 1e-5)
        self.grad_clip = training_config.get('grad_clip', 1.0)

        # Loss function (MSE for behavioral cloning)
        self.mse_loss = nn.MSELoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Learning rate scheduler
        scheduler_config = training_config.get('lr_scheduler', {})
        if scheduler_config.get('type') == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                verbose=True
            )
        else:
            self.scheduler = None

        # Early stopping
        eval_config = config.get('evaluation', {})
        early_stopping_config = eval_config.get('early_stopping', {})
        self.early_stopping_enabled = early_stopping_config.get('enabled', True)
        self.early_stopping_patience = early_stopping_config.get('patience', 10)
        self.early_stopping_min_delta = early_stopping_config.get('min_delta', 0.001)

        # Checkpointing
        checkpoint_config = config.get('checkpoints', {})
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = checkpoint_config.get('save_interval', 10)

        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir / self.log_dir.name)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0

        # Metrics history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        print(f"\n{'='*70}")
        print("Imitation Learning Trainer Initialized")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Learning rate: {self.lr}")
        print(f"Optimizer: Adam (weight_decay={self.weight_decay})")
        print(f"Scheduler: {type(self.scheduler).__name__ if self.scheduler else 'None'}")
        print(f"Early stopping: {'Enabled' if self.early_stopping_enabled else 'Disabled'}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"Log dir: {self.log_dir}")
        print(f"{'='*70}\n")

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch

        Args:
            epoch: Current epoch number

        Returns:
            avg_loss: Average training loss for the epoch
        """
        self.policy.train()
        total_loss = 0.0
        num_batches = 0

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")

        for batch in pbar:
            # Move batch to device
            obs = batch['obs'].to(self.device)        # [B, 321]
            actions = batch['actions'].to(self.device)  # [B, 64]
            mask = batch['mask'].to(self.device)       # [B, 32]

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.policy(obs, deterministic=True)
            pred_actions = outputs['actions']  # [B, 64]

            # Compute MSE loss (only for controlled vehicles)
            # Reshape mask to match actions: [B, 32] -> [B, 32, 1] -> [B, 64]
            mask_expanded = mask.unsqueeze(-1).repeat(1, 1, 2).view(mask.size(0), -1)

            # Apply mask to loss (uncontrolled vehicles don't contribute)
            loss_per_element = (pred_actions - actions) ** 2
            masked_loss = (loss_per_element * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)

            loss = masked_loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{total_loss/num_batches:.6f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self, epoch: int) -> float:
        """
        Validate the model

        Args:
            epoch: Current epoch number

        Returns:
            avg_loss: Average validation loss
        """
        self.policy.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]  ")

            for batch in pbar:
                # Move batch to device
                obs = batch['obs'].to(self.device)
                actions = batch['actions'].to(self.device)
                mask = batch['mask'].to(self.device)

                # Forward pass
                outputs = self.policy(obs, deterministic=True)
                pred_actions = outputs['actions']

                # Compute MSE loss (masked)
                mask_expanded = mask.unsqueeze(-1).repeat(1, 1, 2).view(mask.size(0), -1)
                loss_per_element = (pred_actions - actions) ** 2
                masked_loss = (loss_per_element * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)

                loss = masked_loss

                # Update metrics
                total_loss += loss.item()
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss={self.best_val_loss:.6f}) to {best_path}")

        # Save last checkpoint
        last_path = self.checkpoint_dir / 'last.pth'
        torch.save(checkpoint, last_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])

        print(f"✓ Loaded checkpoint from {checkpoint_path}")
        print(f"  Resuming from epoch {self.current_epoch}")
        print(f"  Best val_loss: {self.best_val_loss:.6f} (epoch {self.best_epoch})")

    def train(self):
        """
        Main training loop
        """
        print(f"\n{'='*70}")
        print("Starting Training")
        print(f"{'='*70}\n")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(epoch)
            self.val_losses.append(val_loss)

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Learning_rate', current_lr, epoch)

            # Print summary
            print(f"\nEpoch {epoch+1}/{self.num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  LR:         {current_lr:.2e}")

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Check if best model
            is_best = val_loss < self.best_val_loss - self.early_stopping_min_delta

            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)

            # Early stopping
            if self.early_stopping_enabled and self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n{'='*70}")
                print("Early Stopping Triggered!")
                print(f"  No improvement for {self.early_stopping_patience} epochs")
                print(f"  Best val_loss: {self.best_val_loss:.6f} (epoch {self.best_epoch})")
                print(f"{'='*70}\n")
                break

        # Training complete
        total_time = time.time() - start_time

        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"{'='*70}")
        print(f"Total time: {total_time/60:.1f} min")
        print(f"Total epochs: {self.current_epoch + 1}")
        print(f"Best val_loss: {self.best_val_loss:.6f} (epoch {self.best_epoch})")
        print(f"Final train loss: {self.train_losses[-1]:.6f}")
        print(f"Final val loss:   {self.val_losses[-1]:.6f}")
        print(f"{'='*70}\n")

        # Save final training summary
        summary = {
            'total_epochs': self.current_epoch + 1,
            'best_epoch': self.best_epoch,
            'best_val_loss': float(self.best_val_loss),
            'final_train_loss': float(self.train_losses[-1]),
            'final_val_loss': float(self.val_losses[-1]),
            'total_time_minutes': total_time / 60,
            'train_losses': [float(l) for l in self.train_losses],
            'val_losses': [float(l) for l in self.val_losses],
            'learning_rates': [float(lr) for lr in self.learning_rates]
        }

        summary_path = self.checkpoint_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Training summary saved to {summary_path}")

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train policy through imitation learning')

    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--demo_data', type=str, required=True,
                        help='Path to demonstrations pickle file')
    parser.add_argument('--output_dir', type=str, default='checkpoints/imitation_learning',
                        help='Output directory for checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/imitation_learning',
                        help='Directory for logs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on (cuda or cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("\n" + "="*70)
    print("Imitation Learning Training")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Demo data: {args.demo_data}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*70)

    # Create dataloaders
    data_config = config.get('data', {})
    train_loader, val_loader, metadata = create_dataloaders(
        demonstrations_path=args.demo_data,
        batch_size=data_config.get('batch_size', 32),
        val_split=data_config.get('val_split', 0.2),
        num_workers=data_config.get('num_workers', 4),
        filter_low_ocr=True,
        min_ocr_threshold=0.5469,
        seed=config.get('global', {}).get('seed', 42)
    )

    print("\nDataset Statistics:")
    print(f"  Total transitions: {metadata['total_transitions']}")
    print(f"  Train: {metadata['train_transitions']}")
    print(f"  Val:   {metadata['val_transitions']}")

    # Create policy network
    policy_config = config.get('policy', {})
    policy = SimplifiedICVPolicy(
        obs_dim=policy_config.get('obs_dim', 321),
        node_dim=policy_config.get('node_dim', 9),
        hidden_dim=policy_config.get('hidden_dim', 128),
        num_layers=policy_config.get('num_layers', 3),
        num_vehicles=policy_config.get('num_vehicles', 32),
        device=args.device,
        use_safety_shield=False  # Not needed for training
    )

    print(f"\nPolicy Network:")
    print(f"  obs_dim: {policy_config.get('obs_dim', 321)}")
    print(f"  hidden_dim: {policy_config.get('hidden_dim', 128)}")
    print(f"  num_layers: {policy_config.get('num_layers', 3)}")
    print(f"  Parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = ImitationLearningTrainer(
        policy=policy,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device,
        log_dir=args.log_dir,
        checkpoint_dir=args.output_dir
    )

    # Resume from checkpoint if specified
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
