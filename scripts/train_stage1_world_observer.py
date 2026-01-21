#!/usr/bin/env python3
"""
Stage 1: 世界观察者训练（World Observer Training）

功能：
1. 使用IDM策略收集人工驾驶轨迹
2. 训练WorldModel预测未来状态和风险
3. 监督学习，MSE + BCE loss

使用方法：
    python scripts/train_stage1_world_observer.py --config configs/v5_complete.yaml

输出：
    - checkpoints/v5_complete/stage1_world_model.pth
    - logs/v5_complete/stage1/
"""

import os
import sys
import argparse
import yaml
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.joint_icv_policy import create_joint_icv_policy
from src.models.world_model import WorldModel, create_world_model
from src.env.competition_env import CompetitionSumoEnv


class TrajectoryDataset(Dataset):
    """
    轨迹数据集

    存储IDM策略生成的轨迹，用于训练WorldModel
    """

    def __init__(self, max_episodes=1000, max_steps_per_episode=500):
        self.trajectories = []
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode

    def add_trajectory(self, trajectory):
        """
        添加一条轨迹

        Args:
            trajectory: Dict {
                'observations': [T, obs_dim],
                'next_observations': [T, obs_dim],
                'risk_labels': [T, N]
            }
        """
        self.trajectories.append(trajectory)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        """
        获取一条轨迹

        Returns:
            Dict: {
                'obs': [T, obs_dim],
                'next_obs': [T, obs_dim],
                'risk_labels': [T, N]
            }
        """
        return self.trajectories[idx]


class IDMController:
    """
    IDM控制器（智能驾驶模型）

    用于生成人工驾驶轨迹作为WorldModel的训练数据
    """

    def __init__(self, num_vehicles=32, device='cuda'):
        self.num_vehicles = num_vehicles
        self.device = device

        # IDM参数
        self.desired_velocity = 20.0  # 期望速度 (m/s)
        self.min_gap = 2.0  # 最小车距 (m)
        self.time_headway = 1.5  # 车头时距 (s)
        self.accel_max = 1.5  # 最大加速度 (m/s²)
        self.decel_comfort = 2.0  # 舒适减速度 (m/s²)

    def compute_idm_action(self, vehicle_states):
        """
        计算IDM动作

        Args:
            vehicle_states: [N, 9] 车辆状态

        Returns:
            actions: [N, 2] (accel, lane_change)
        """
        num_vehicles = vehicle_states.shape[0]
        actions = torch.zeros(num_vehicles, 2, device=self.device)

        # 提取信息
        x = vehicle_states[:, 0]  # 纵向位置
        v = vehicle_states[:, 2]  # 纵向速度
        lanes = vehicle_states[:, 6].long()  # 车道索引

        # 为每辆车计算IDM加速度
        for i in range(num_vehicles):
            # 找到同车道的前车
            lane_i = lanes[i].item()
            same_lane = (lanes == lane_i) & (x > x[i])

            if same_lane.any():
                # 有前车
                lead_idx = torch.where(same_lane)[0][x[same_lane].argmin()].item()

                # 前车信息
                gap = x[lead_idx] - x[i] - 5.0  # 车距（减去车长）
                delta_v = v[i] - v[lead_idx]  # 相对速度

                # IDM公式
                accel = self.accel_max * (
                    1.0 - (v[i] / self.desired_velocity)**4 -
                    (self.compute_desired_gap(v[i], delta_v) / (gap + 1e-6))**2
                )

                # 限制加速度范围
                accel = torch.clamp(accel, -self.decel_comfort, self.accel_max)
            else:
                # 无前车，自由流
                accel = self.accel_max * (1.0 - (v[i] / self.desired_velocity)**4)
                accel = torch.clamp(accel, 0, self.accel_max)

            actions[i, 0] = accel
            # IDM不考虑换道，设为0
            actions[i, 1] = 0.0

        return actions

    def compute_desired_gap(self, v, delta_v):
        """
        计算期望车距

        Args:
            v: 当前速度
            delta_v: 相对速度（自车-前车）

        Returns:
            desired_gap: 期望车距
        """
        return self.min_gap + v * self.time_headway + \
               (v * delta_v) / (2 * torch.sqrt(torch.tensor(self.accel_max * self.decel_comfort)))


def collect_idm_trajectories(env, num_episodes=20, max_steps=500, device='cuda'):
    """
    收集IDM轨迹

    Args:
        env: CompetitionSumoEnv
        num_episodes: 收集的episode数
        max_steps: 每个episode的最大步数
        device: 设备

    Returns:
        dataset: TrajectoryDataset
    """
    print(f"\n收集IDM轨迹: {num_episodes} episodes, {max_steps} steps/episode")

    idm_controller = IDMController(num_vehicles=32, device=device)
    dataset = TrajectoryDataset()

    for episode in tqdm(range(num_episodes), desc="收集IDM轨迹"):
        obs, _ = env.reset()
        episode_data = {
            'observations': [],
            'next_observations': [],
            'risk_labels': []
        }

        for step in range(max_steps):
            # 转换为tensor
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            # 解析车辆状态
            obs_dim = obs.shape[0]
            vehicle_dim = (obs_dim - 32 - 1)
            num_vehicles = vehicle_dim // 9
            vehicle_states_flat = obs[:vehicle_dim]
            vehicle_states = torch.tensor(vehicle_states_flat, dtype=torch.float32).view(num_vehicles, 9).to(device)

            # IDM动作
            actions = idm_controller.compute_idm_action(vehicle_states)

            # 执行动作
            next_obs, reward, done, truncated, info = env.step(actions.cpu().numpy())

            # 存储数据
            episode_data['observations'].append(obs.copy())
            episode_data['next_observations'].append(next_obs.copy())

            # 计算风险标签（基于TTC）
            vehicle_states_np = vehicle_states.cpu().numpy()
            risk_labels = compute_risk_labels(vehicle_states_np)
            episode_data['risk_labels'].append(risk_labels)

            obs = next_obs

            if done or truncated:
                break

        # 转换为numpy数组
        episode_data['observations'] = np.array(episode_data['observations'])
        episode_data['next_observations'] = np.array(episode_data['next_observations'])
        episode_data['risk_labels'] = np.array(episode_data['risk_labels'])

        dataset.add_trajectory(episode_data)

    print(f"收集完成: {len(dataset)} 条轨迹")

    return dataset


def compute_risk_labels(vehicle_states):
    """
    计算风险标签（基于TTC）

    Args:
        vehicle_states: [N, 9] 车辆状态

    Returns:
        risk_labels: [N] 风险标签（0=安全，1=风险）
    """
    num_vehicles = vehicle_states.shape[0]
    risk_labels = np.zeros(num_vehicles)

    x = vehicle_states[:, 0]
    v = vehicle_states[:, 2]
    lanes = vehicle_states[:, 6].astype(int)

    for i in range(num_vehicles):
        # 找到同车道前车
        same_lane = (lanes == lanes[i]) & (x > x[i])

        if same_lane.any():
            lead_idx = np.where(same_lane)[0][x[same_lane].argmin()]

            # 计算TTC
            gap = x[lead_idx] - x[i] - 5.0
            delta_v = v[i] - v[lead_idx]

            if delta_v > 0:
                ttc = gap / delta_v
                # TTC < 2秒为风险
                risk_labels[i] = 1 if ttc < 2.0 else 0

    return risk_labels


def train_world_model(model, dataloader, config, device='cuda'):
    """
    训练WorldModel

    Args:
        model: WorldModel
        dataloader: DataLoader
        config: 配置字典
        device: 设备

    Returns:
        best_loss: 最佳损失
    """
    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['lr_schedule']['T_max'],
        eta_min=config['training']['lr_schedule']['eta_min']
    )

    # 混合精度训练
    scaler = GradScaler()

    # 训练循环
    num_epochs = config['training']['num_epochs']
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_flow_loss = 0.0
        epoch_risk_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            # 解析batch
            obs = batch['obs'].to(device)  # [B, T, obs_dim]
            next_obs = batch['next_obs'].to(device)  # [B, T, obs_dim]
            risk_labels = batch['risk_labels'].to(device)  # [B, T, N]

            B, T, obs_dim = obs.shape

            # 解析观测
            vehicle_dim = (obs_dim - 32 - 1)
            num_vehicles = vehicle_dim // 9

            # 初始化hidden状态
            hidden = model.initialize_hidden(B * T, device)

            # 前向传播（逐步）
            pred_states_list = []
            pred_risks_list = []

            obs_flat = obs.view(-1, obs_dim)  # [B*T, obs_dim]
            next_obs_flat = next_obs.view(-1, obs_dim)
            risk_flat = risk_labels.view(-1, num_vehicles)

            for t in range(T):
                obs_t = obs_flat[:, t] if obs.dim() == 3 else obs  # [B, obs_dim]

                # 解析车辆状态
                vehicle_features_t = obs_t[:, :vehicle_dim]  # [B, N*9]
                vehicle_states_t = vehicle_features_t.view(B, num_vehicles, 9)  # [B, N, 9]

                # 创建嵌入（简化：直接使用车辆状态特征）
                embeddings_t = vehicle_states_t[:, :, :2].float()  # [B, N, 2] 简化

                # 扩展到64维
                if embeddings_t.size(-1) < 64:
                    embeddings_t = torch.cat([
                        embeddings_t,
                        torch.zeros(B, num_vehicles, 64 - embeddings_t.size(-1), device=device)
                    ], dim=-1)

                # WorldModel前向传播
                world_outputs = model(embeddings_t, hidden)

                pred_states_list.append(world_outputs['pred_next_states'])
                pred_risks_list.append(world_outputs['risk_prob'])

                hidden = world_outputs['hidden']

            # 堆叠预测
            pred_states = torch.stack(pred_states_list, dim=1)  # [B, T, N, 9]
            pred_risks = torch.stack(pred_risks_list, dim=1)  # [B, T, N]

            # 目标状态
            target_states_list = []
            for t in range(T):
                next_obs_t = next_obs_flat[:, t] if next_obs.dim() == 3 else next_obs
                vehicle_features_t = next_obs_t[:, :vehicle_dim]
                vehicle_states_t = vehicle_features_t.view(B, num_vehicles, 9)
                target_states_list.append(vehicle_states_t)

            target_states = torch.stack(target_states_list, dim=1)  # [B, T, N, 9]

            # 计算损失
            # 简化：只计算有效时间步
            valid_mask = torch.ones(B, T, 1, device=device)

            # 流损失（MSE）
            flow_loss = F.mse_loss(
                pred_states * valid_mask,
                target_states * valid_mask
            )

            # 风险损失（BCE）
            risk_loss = F.binary_cross_entropy(
                pred_risks.view(-1),
                risk_flat.view(-1).float()
            )

            # 总损失
            loss = flow_loss + 0.5 * risk_loss

            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()

            # 统计
            epoch_loss += loss.item()
            epoch_flow_loss += flow_loss.item()
            epoch_risk_loss += risk_loss.item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'flow': f"{flow_loss.item():.4f}",
                'risk': f"{risk_loss.item():.4f}"
            })

        # 更新学习率
        scheduler.step()

        # 平均损失
        avg_loss = epoch_loss / len(dataloader)
        avg_flow_loss = epoch_flow_loss / len(dataloader)
        avg_risk_loss = epoch_risk_loss / len(dataloader)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Loss: {avg_loss:.4f} | Flow: {avg_flow_loss:.4f} | Risk: {avg_risk_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 早停检查
        if avg_loss < best_loss - config['training']['early_stopping']['min_delta']:
            best_loss = avg_loss
            patience_counter = 0

            # 保存最佳模型
            checkpoint_path = Path(config['global']['checkpoint_dir']) / 'stage1_best.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✅ 最佳模型已保存: {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{config['training']['early_stopping']['patience']}")

        if patience_counter >= config['training']['early_stopping']['patience']:
            print(f"\n早停触发！")
            break

    return best_loss


def main():
    parser = argparse.ArgumentParser(description="Stage 1: World Observer Training")
    parser.add_argument('--config', type=str, default='configs/v5_complete.yaml',
                        help='配置文件路径')
    parser.add_argument('--num_episodes', type=int, default=20,
                        help='收集的episode数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("Stage 1: 世界观察者训练")
    print("=" * 80)
    print(f"\n配置文件: {args.config}")
    print(f"设备: {args.device}")
    print(f"收集episode数: {args.num_episodes}")

    # 创建环境
    print("\n创建环境...")
    env = CompetitionSumoEnv(
        net_file=config['environment']['net_file'],
        route_file=config['environment']['route_file'],
        max_vehicles=config['environment']['icv_config']['max_vehicles'],
        use_icv=False  # 使用IDM而非ICV
    )

    # 收集IDM轨迹
    print("\n" + "=" * 80)
    dataset = collect_idm_trajectories(
        env,
        num_episodes=args.num_episodes,
        max_steps=config['training']['episode_length'],
        device=args.device
    )

    # 创建数据加载器
    print("\n创建数据加载器...")
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['global']['num_workers']
    )

    # 创建WorldModel
    print("\n创建WorldModel...")
    world_model = create_world_model(
        hidden_dim=config['policy']['hidden_dim'],
        latent_dim=config['stage1_world_observer']['world_model']['latent_dim'],
        num_layers=config['stage1_world_observer']['world_model']['num_layers'],
        num_vehicles=config['environment']['icv_config']['max_vehicles'],
        device=args.device
    )

    print(f"WorldModel参数量: {sum(p.numel() for p in world_model.parameters()):,}")

    # 训练WorldModel
    print("\n" + "=" * 80)
    print("开始训练WorldModel")
    print("=" * 80)

    start_time = time.time()
    best_loss = train_world_model(world_model, dataloader, config, args.device)
    elapsed_time = time.time() - start_time

    print(f"\n训练完成！")
    print(f"  最佳损失: {best_loss:.4f}")
    print(f"  训练时长: {elapsed_time/3600:.2f} 小时")

    # 保存最终模型
    final_checkpoint = Path(config['global']['checkpoint_dir']) / 'stage1_final.pth'
    torch.save(world_model.state_dict(), final_checkpoint)
    print(f"  最终模型已保存: {final_checkpoint}")

    print("\n" + "=" * 80)
    print("🎉 Stage 1 训练完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
