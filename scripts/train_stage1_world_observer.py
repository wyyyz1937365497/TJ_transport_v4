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
import torch.nn.functional as F
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


class VehicleEmbedding(nn.Module):
    """
    车辆状态嵌入层

    将9维车辆状态映射到64维嵌入空间
    """

    def __init__(self, input_dim=9, hidden_dim=64):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, hidden_dim)
        )

    def forward(self, vehicle_states):
        """
        Args:
            vehicle_states: [B, N, 9] 车辆状态

        Returns:
            embeddings: [B, N, hidden_dim] 车辆嵌入
        """
        return self.embedding(vehicle_states)


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


def normalize_vehicle_states(raw_states):
    """
    归一化车辆状态特征

    特征范围:
    - [0] s: 纵向位置 [0, 1000+] → [0, 1]
    - [1] d: 横向偏移 [-10, 10] → [-1, 1]
    - [2] vs: 纵向速度 [0, 30] → [0, 1]
    - [3] vd: 横向速度 [-5, 5] → [-1, 1]
    - [4] speed: 总速度 [0, 30] → [0, 1]
    - [5] acceleration: 加速度 [-3, 3] → [-1, 1]
    - [6] lane_index: 车道索引 [0, 3] → [0, 1]
    - [7] angle: 角度 [0, 360] → [0, 1]
    - [8] in_bottleneck: 是否在瓶颈 [0, 1] → [0, 1]

    Args:
        raw_states: [N, 9] 原始车辆状态

    Returns:
        normalized_states: [N, 9] 归一化后的车辆状态
    """
    normalized = raw_states.copy()

    # 位置特征归一化
    normalized[:, 0] /= 1000.0      # s: [0, 1000+] → [0, ~1]
    normalized[:, 1] /= 10.0        # d: [-10, 10] → [-1, 1]

    # 速度特征归一化
    normalized[:, 2] /= 30.0        # vs: [0, 30] → [0, 1]
    normalized[:, 3] /= 5.0         # vd: [-5, 5] → [-1, 1]
    normalized[:, 4] /= 30.0        # speed: [0, 30] → [0, 1]

    # 加速度归一化
    normalized[:, 5] /= 3.0         # acceleration: [-3, 3] → [-1, 1]

    # 离散特征归一化
    normalized[:, 6] /= 3.0         # lane_index: [0, 3] → [0, 1]
    normalized[:, 7] /= 360.0       # angle: [0, 360] → [0, 1]

    # in_bottleneck已经是[0, 1]，无需归一化
    # normalized[:, 8] 保持不变

    return normalized


def denormalize_vehicle_states(normalized_states):
    """
    反归一化车辆状态（用于可视化或评估）

    Args:
        normalized_states: [N, 9] 归一化后的车辆状态

    Returns:
        raw_states: [N, 9] 原始尺度的车辆状态
    """
    raw = normalized_states.copy()

    # 反归一化
    raw[:, 0] *= 1000.0       # s
    raw[:, 1] *= 10.0         # d
    raw[:, 2] *= 30.0         # vs
    raw[:, 3] *= 5.0          # vd
    raw[:, 4] *= 30.0         # speed
    raw[:, 5] *= 3.0          # acceleration
    raw[:, 6] *= 3.0          # lane_index
    raw[:, 7] *= 360.0        # angle

    return raw


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
        计算IDM动作（向量化加速版）

        Args:
            vehicle_states: [N, 9] 车辆状态

        Returns:
            actions: [N, 2] (accel, lane_change)
        """
        num_vehicles = vehicle_states.shape[0]
        if num_vehicles == 0:
            return torch.zeros(0, 2, device=self.device)

        actions = torch.zeros(num_vehicles, 2, device=self.device)

        # 提取信息
        x = vehicle_states[:, 0]  # 纵向位置 [N]
        v = vehicle_states[:, 2]  # 纵向速度 [N]
        lanes = vehicle_states[:, 6].long()  # 车道索引 [N]

        # 1. 构建前车矩阵
        # 扩展维度以进行广播: [N, 1] vs [1, N]
        lane_matrix = (lanes.unsqueeze(1) == lanes.unsqueeze(0))  # 同车道 [N, N]
        dist_matrix = x.unsqueeze(0) - x.unsqueeze(1)  # 相对距离 x_j - x_i [N, N] (正值表示j在i前面)
        
        # 过滤无效前车（非同车道 或 距离<=0）
        # 将无效距离设为无穷大
        valid_leader = lane_matrix & (dist_matrix > 0)
        dist_matrix = torch.where(valid_leader, dist_matrix, torch.tensor(float('inf'), device=self.device))
        
        # 找到最近的前车
        gap_raw, lead_idx = torch.min(dist_matrix, dim=1)  # [N]
        has_leader = gap_raw != float('inf')
        
        # 2. 计算IDM加速度
        # 计算自由流项
        accel_free = self.accel_max * (1.0 - (v / self.desired_velocity)**4)
        
        # 计算交互项（仅对有前车的车辆）
        accel_interaction = torch.zeros_like(v)
        
        if has_leader.any():
            v_lead = v[lead_idx[has_leader]]
            gap = gap_raw[has_leader] - 5.0  # 减去车长
            gap = torch.clamp(gap, min=1e-6)  # 避免除零
            
            delta_v = v[has_leader] - v_lead
            
            # 期望车距
            desired_gap = self.min_gap + v[has_leader] * self.time_headway + \
                        (v[has_leader] * delta_v) / (2 * np.sqrt(self.accel_max * self.decel_comfort))
            
            accel_interaction[has_leader] = -self.accel_max * (desired_gap / gap)**2

        # 总加速度
        accel = accel_free + accel_interaction
        
        # 限制范围
        accel = torch.clamp(accel, -self.decel_comfort, self.accel_max)
        
        actions[:, 0] = accel
        # IDM不考虑换道，设为0
        actions[:, 1] = 0.0

        return actions

    # compute_desired_gap 已被内联到 vectorization 版本中，不再需要


def collect_idm_trajectories(env, num_episodes=20, max_steps=500, device='cuda',
                             cache_dir=None, use_cache=True, force_refresh=False):
    """
    收集IDM轨迹（支持缓存）

    Args:
        env: CompetitionSumoEnv
        num_episodes: 收集的episode数
        max_steps: 每个episode的最大步数
        device: 设备
        cache_dir: 缓存目录路径
        use_cache: 是否使用缓存
        force_refresh: 是否强制刷新缓存

    Returns:
        dataset: TrajectoryDataset
    """
    print(f"\n收集IDM轨迹: {num_episodes} episodes, {max_steps} steps/episode")

    # 生成缓存key（基于参数的哈希值）
    import hashlib
    import pickle
    from pathlib import Path

    if cache_dir is None:
        cache_dir = Path("cache/stage1_trajectories")
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    # 创建唯一的缓存key
    cache_params = {
        'num_episodes': num_episodes,
        'max_steps': max_steps,
        'max_vehicles': 600,
    }
    cache_key = hashlib.md5(str(cache_params).encode()).hexdigest()[:12]
    cache_file = cache_dir / f"trajectories_{cache_key}.pkl"

    # 尝试从缓存加载
    if use_cache and not force_refresh and cache_file.exists():
        print(f"[缓存] 发现缓存文件: {cache_file}")
        print(f"[缓存] 正在加载...")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)

            # 验证缓存数据
            if cached_data['num_episodes'] == num_episodes:
                dataset = TrajectoryDataset()
                for traj in cached_data['trajectories']:
                    dataset.add_trajectory(traj)

                print(f"[缓存] 成功加载 {len(dataset)} 条轨迹（耗时: <1秒）")
                print(f"[缓存] 节省了约 {num_episodes * 6} 分钟的仿真时间")
                return dataset
            else:
                print(f"[缓存] 缓存数据不匹配，将重新收集")
        except Exception as e:
            print(f"[缓存] 加载失败: {e}，将重新收集")

    # 缓存未命中或强制刷新，进行数据收集
    print(f"[数据收集] 开始IDM轨迹收集...")
    idm_controller = IDMController(num_vehicles=600, device=device)
    dataset = TrajectoryDataset()

    # 固定最大车辆数（用于padding）
    MAX_VEHICLES = 600

    for episode in tqdm(range(num_episodes), desc="收集IDM轨迹"):
        obs = env.reset()
        episode_data = {
            'obs': [],
            'next_obs': [],
            'risk_labels': [],
            'valid_masks': []  # 1表示有效车辆，0表示padding
        }

        for step in range(max_steps):
            # 从dict中提取车辆状态
            vehicle_ids = obs['vehicle_ids']
            vehicle_states_dict = obs['vehicle_states']

            # 转换为list of list (优化的列表推导式)
            vehicle_states_list = [
                [
                    vehicle_states_dict[veh_id]['s'],
                    vehicle_states_dict[veh_id]['d'],
                    vehicle_states_dict[veh_id]['vs'],
                    vehicle_states_dict[veh_id]['vd'],
                    vehicle_states_dict[veh_id]['speed'],
                    vehicle_states_dict[veh_id]['acceleration'],
                    vehicle_states_dict[veh_id]['lane_index'],
                    vehicle_states_dict[veh_id]['angle'],
                    1.0 if vehicle_states_dict[veh_id]['in_bottleneck'] else 0.0
                ]
                for veh_id in vehicle_ids
            ]

            # Padding到固定大小
            if len(vehicle_states_list) > 0:
                vehicle_states = np.array(vehicle_states_list)  # [N, 9]
                num_vehicles = len(vehicle_states_list)
            else:
                vehicle_states = np.zeros((0, 9))
                num_vehicles = 0

            # Pad到MAX_VEHICLES
            if num_vehicles < MAX_VEHICLES:
                pad_count = MAX_VEHICLES - num_vehicles
                padding = np.zeros((pad_count, 9))
                vehicle_states = np.vstack([vehicle_states, padding])
            elif num_vehicles > MAX_VEHICLES:
                # 随机采样MAX_VEHICLES个车辆
                indices = np.random.choice(num_vehicles, MAX_VEHICLES, replace=False)
                vehicle_states = vehicle_states[indices]
                num_vehicles = MAX_VEHICLES

            # 🔥 关键修复：应用特征归一化
            vehicle_states = normalize_vehicle_states(vehicle_states)

            obs_tensor = torch.as_tensor(vehicle_states, dtype=torch.float32).unsqueeze(0).to(device)

            # IDM动作
            actions = idm_controller.compute_idm_action(obs_tensor.squeeze(0))

            # 执行动作 - 转换为dict格式 {vehicle_id: [accel, lane_change]}
            actions_dict = {}
            for i, veh_id in enumerate(vehicle_ids[:min(len(vehicle_ids), MAX_VEHICLES)]):
                actions_dict[veh_id] = actions[i].cpu().numpy()

            next_obs, reward, done, info = env.step(actions_dict)

            # 解析next_obs
            next_vehicle_ids = next_obs['vehicle_ids']
            next_vehicle_states_dict = next_obs['vehicle_states']

            next_states_list = [
                [
                    next_vehicle_states_dict[veh_id]['s'],
                    next_vehicle_states_dict[veh_id]['d'],
                    next_vehicle_states_dict[veh_id]['vs'],
                    next_vehicle_states_dict[veh_id]['vd'],
                    next_vehicle_states_dict[veh_id]['speed'],
                    next_vehicle_states_dict[veh_id]['acceleration'],
                    next_vehicle_states_dict[veh_id]['lane_index'],
                    next_vehicle_states_dict[veh_id]['angle'],
                    1.0 if next_vehicle_states_dict[veh_id]['in_bottleneck'] else 0.0
                ]
                for veh_id in next_vehicle_ids
            ]

            if len(next_states_list) > 0:
                next_states = np.array(next_states_list)
                next_num = len(next_states_list)
            else:
                next_states = np.zeros((0, 9))
                next_num = 0

            if next_num < MAX_VEHICLES:
                pad_count = MAX_VEHICLES - next_num
                next_states = np.vstack([next_states, np.zeros((pad_count, 9))])
            elif next_num > MAX_VEHICLES:
                indices = np.random.choice(next_num, MAX_VEHICLES, replace=False)
                next_states = next_states[indices]
                next_num = MAX_VEHICLES

            # 🔥 关键修复：应用特征归一化到next_states
            next_states = normalize_vehicle_states(next_states)

            # 有效mask（基于next_obs的真实车辆数）
            valid_mask = np.zeros((MAX_VEHICLES, 1), dtype=np.float32)
            valid_mask[:next_num] = 1.0

            # 存储数据（已经是固定大小）
            episode_data['obs'].append(vehicle_states.copy())
            episode_data['next_obs'].append(next_states.copy())
            episode_data['valid_masks'].append(valid_mask)

            # 计算风险标签（基于TTC）- 只对有效车辆计算
            if next_num > 0:
                valid_states = next_states[:next_num]
                risk_labels = compute_risk_labels(valid_states)
                # Padding到MAX_VEHICLES
                if next_num < MAX_VEHICLES:
                    risk_padding = np.zeros(MAX_VEHICLES - next_num)
                    risk_labels = np.concatenate([risk_labels, risk_padding])
            else:
                risk_labels = np.zeros(MAX_VEHICLES)
            episode_data['risk_labels'].append(risk_labels)

            obs = next_obs

            if done:
                break

        # 转换为numpy数组（现在是固定大小，使用float32以节省空间）
        episode_data['obs'] = np.array(episode_data['obs'], dtype=np.float32)  # [T, MAX_VEHICLES, 9]
        episode_data['next_obs'] = np.array(episode_data['next_obs'], dtype=np.float32)
        episode_data['risk_labels'] = np.array(episode_data['risk_labels'], dtype=np.float32)  # [T, MAX_VEHICLES]
        episode_data['valid_masks'] = np.array(episode_data['valid_masks'], dtype=np.float32)  # [T, MAX_VEHICLES, 1]

        dataset.add_trajectory(episode_data)

    print(f"收集完成: {len(dataset)} 条轨迹")

    # 保存到缓存
    if use_cache or force_refresh:
        print(f"[缓存] 正在保存到缓存...")
        try:
            cache_data = {
                'num_episodes': num_episodes,
                'max_steps': max_steps,
                'max_vehicles': MAX_VEHICLES,
                'trajectories': dataset.trajectories,
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_steps': sum(len(traj['obs']) for traj in dataset.trajectories),
                }
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

            # 计算缓存文件大小
            cache_size_mb = cache_file.stat().st_size / (1024 * 1024)
            print(f"[缓存] 缓存已保存: {cache_file}")
            print(f"[缓存] 缓存大小: {cache_size_mb:.2f} MB")
            print(f"[缓存] 下次运行将自动加载，节省约 {num_episodes * 6} 分钟")
        except Exception as e:
            print(f"[缓存] 保存失败: {e}")

    return dataset


def compute_risk_labels(vehicle_states):
    """
    计算风险标签（基于TTC，向量化加速版）

    Args:
        vehicle_states: [N, 9] 车辆状态

    Returns:
        risk_labels: [N] 风险标签（0=安全，1=风险）
    """
    num_vehicles = vehicle_states.shape[0]
    if num_vehicles == 0:
        return np.zeros(0)

    # 提取信息
    x = vehicle_states[:, 0]
    v = vehicle_states[:, 2]
    lanes = vehicle_states[:, 6].astype(int)

    # 1. 构建矩阵计算所有车辆对的距离
    # [N, 1] vs [1, N] -> [N, N]
    lane_matrix = (lanes[:, None] == lanes[None, :])
    dist_matrix = x[None, :] - x[:, None]  # x_j - x_i
    
    # 过滤: 同车道且在前方的车辆
    # 将无效值设为无穷大
    valid_mask = lane_matrix & (dist_matrix > 0)
    dist_matrix_filtered = np.where(valid_mask, dist_matrix, np.inf)
    
    # 2. 找到每辆车的紧前车
    min_dist_idx = np.argmin(dist_matrix_filtered, axis=1)
    min_dist = np.min(dist_matrix_filtered, axis=1)
    
    # 哪些车有前车
    has_leader = min_dist != np.inf
    
    risk_labels = np.zeros(num_vehicles)
    
    # 3. 对有前车的计算TTC
    if np.any(has_leader):
        # 提取前车索引
        leader_indices = min_dist_idx[has_leader]
        
        # 计算Gap和Delta V
        gap = min_dist[has_leader] - 5.0
        # 限制gap非负
        gap = np.maximum(gap, 0.001)

        v_current = v[has_leader]
        v_leader = v[leader_indices]
        delta_v = v_current - v_leader
        # 🔥 关键修复：避免除零
        delta_v = np.maximum(delta_v, 0.001)

        # 计算TTC: gap / delta_v
        # 只关心 delta_v > 0 (正在接近) 的情况
        ttc = gap / delta_v
        # 处理可能的inf或nan
        ttc = np.nan_to_num(ttc, posinf=999, neginf=0)

        risk_mask = (delta_v > 0.001) & (ttc < 2.0)
        
        # 填充结果
        # 需要将risk_mask映射回所有车辆的索引
        risk_indices = np.where(has_leader)[0][risk_mask]
        risk_labels[risk_indices] = 1.0

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
        vehicle_embedding: VehicleEmbedding层（用于Stage 2/3）
    """
    # 提取训练配置
    stage1_config = config.get('stage1_world_observer', {})
    training_config = stage1_config.get('training', {})

    # 创建车辆嵌入层
    vehicle_embedding = VehicleEmbedding(
        input_dim=9,
        hidden_dim=config['policy']['hidden_dim']
    ).to(device)

    # 将嵌入层参数加入优化器
    optimizer = optim.Adam(
        list(model.parameters()) + list(vehicle_embedding.parameters()),
        lr=training_config.get('lr', 3.0e-4),
        weight_decay=training_config.get('weight_decay', 1.0e-5)
    )

    # 学习率调度器
    lr_schedule_config = training_config.get('lr_schedule', {})
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=lr_schedule_config.get('T_max', 20),
        eta_min=lr_schedule_config.get('eta_min', 1.0e-5)
    )

    # 混合精度训练
    try:
        # Pytorch 2.0+
        scaler = torch.amp.GradScaler('cuda')
    except:
        # 旧版本
        scaler = GradScaler()

    # 训练循环
    num_epochs = training_config['num_epochs']
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
            obs = batch['obs'].to(device)  # [B, T, MAX_VEHICLES, 9]
            next_obs = batch['next_obs'].to(device)  # [B, T, MAX_VEHICLES, 9]
            risk_labels = batch['risk_labels'].to(device)  # [B, T, MAX_VEHICLES]
            valid_masks = batch['valid_masks'].to(device)  # [B, T, MAX_VEHICLES, 1]

            B, T, MAX_VEHICLES, state_dim = obs.shape
            
            # 开启混合精度上下文
            with torch.amp.autocast('cuda') if hasattr(torch.amp, 'autocast') else autocast():
                # 初始化hidden状态
                hidden = model.initialize_hidden(B, device)

                # 前向传播（逐步）
                pred_states_list = []
                pred_risks_list = []

                for t in range(T):
                    obs_t = obs[:, t, :, :]  # [B, MAX_VEHICLES, 9]

                    # 使用嵌入层创建64维嵌入（完整的9维特征）
                    embeddings_t = vehicle_embedding(obs_t)  # [B, MAX_VEHICLES, 64]

                    # WorldModel前向传播（不flatten，保持batch维度）
                    world_outputs = model(embeddings_t, hidden)

                    pred_next = world_outputs['pred_next_states']  # [B, MAX_VEHICLES, 9]
                    risk_prob = world_outputs['risk_prob']  # [B, MAX_VEHICLES]

                    pred_states_list.append(pred_next)
                    pred_risks_list.append(risk_prob)

                    hidden = world_outputs['hidden']

                # 堆叠预测
                pred_states = torch.stack(pred_states_list, dim=1)  # [B, T, MAX_VEHICLES, 9]
                pred_risks = torch.stack(pred_risks_list, dim=1)  # [B, T, MAX_VEHICLES]

                # 目标状态
                target_states = next_obs  # [B, T, MAX_VEHICLES, 9]

                # 计算损失（仅对有效车辆）
                valid_mask = valid_masks  # [B, T, MAX, 1]

                # 流损失（MSE）带mask，按维度平均，避免因特征尺度过大导致loss爆炸
                flow_diff = (pred_states - target_states) ** 2  # [B, T, MAX, 9]
                flow_loss = (flow_diff * valid_mask).sum() / (
                    valid_mask.sum().clamp_min(1.0) * flow_diff.size(-1)
                )

            # 风险损失：在FP32 + 禁用autocast下计算，避免BCE与autocast冲突
            autocast_off = (
                torch.amp.autocast(device_type='cuda', enabled=False)
                if hasattr(torch.amp, 'autocast') else
                autocast(enabled=False)
            )
            with autocast_off:
                # Mask化BCE：仅对有效车辆计算
                risk_loss_raw = F.binary_cross_entropy(
                    pred_risks.view(-1).float(),
                    risk_labels.view(-1).float(),
                    reduction='none'
                )
                risk_mask = valid_mask.view(-1)
                risk_loss = (risk_loss_raw * risk_mask).sum() / risk_mask.sum().clamp_min(1.0)

            # 总损失
            loss = flow_loss + 0.5 * risk_loss

            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(vehicle_embedding.parameters()),
                training_config.get('grad_clip', 1.0)
            )
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
        early_stopping_config = training_config.get('early_stopping', {})
        if avg_loss < best_loss - early_stopping_config.get('min_delta', 1.0e-4):
            best_loss = avg_loss
            patience_counter = 0

            # 保存最佳模型（WorldModel + VehicleEmbedding）
            checkpoint_path = Path(config['global']['checkpoint_dir']) / 'stage1_best.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'world_model': model.state_dict(),
                'vehicle_embedding': vehicle_embedding.state_dict()
            }, checkpoint_path)
            print(f"  ✅ 最佳模型已保存: {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{early_stopping_config.get('patience', 5)}")

        if patience_counter >= early_stopping_config.get('patience', 5):
            print(f"\n早停触发！")
            break

    return best_loss, vehicle_embedding


def main():
    parser = argparse.ArgumentParser(description="Stage 1: World Observer Training")
    parser.add_argument('--config', type=str, default='configs/v5_complete.yaml',
                        help='配置文件路径')
    parser.add_argument('--num_episodes', type=int, default=20,
                        help='收集的episode数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')

    # 缓存相关参数
    parser.add_argument('--use_cache', type=lambda x: x.lower() == 'true', default=True,
                        help='是否使用缓存 (True/False, 默认: True)')
    parser.add_argument('--force_refresh', action='store_true',
                        help='强制刷新缓存，重新收集数据')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='自定义缓存目录路径')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 提取Stage 1训练配置
    stage1_config = config.get('stage1_world_observer', {})
    training_config = stage1_config.get('training', {})

    print("=" * 80)
    print("Stage 1: 世界观察者训练")
    print("=" * 80)
    print(f"\n配置文件: {args.config}")
    print(f"设备: {args.device}")
    print(f"收集episode数: {args.num_episodes}")

    # 创建环境
    print("\n创建环境...")
    env = CompetitionSumoEnv(
        config=config,
        use_gui=False,
        device=args.device
    )

    # 收集IDM轨迹
    print("\n" + "=" * 80)
    dataset = collect_idm_trajectories(
        env,
        num_episodes=args.num_episodes,
        max_steps=training_config['episode_length'],
        device=args.device,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        force_refresh=args.force_refresh
    )

    # 创建数据加载器
    print("\n创建数据加载器...")
    dataloader = DataLoader(
        dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=0  # 避免multiprocessing问题
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
    best_loss, vehicle_embedding = train_world_model(world_model, dataloader, config, args.device)
    elapsed_time = time.time() - start_time

    print(f"\n训练完成！")
    print(f"  最佳损失: {best_loss:.4f}")
    print(f"  训练时长: {elapsed_time/3600:.2f} 小时")

    # 保存最终模型（WorldModel + VehicleEmbedding）
    final_checkpoint = Path(config['global']['checkpoint_dir']) / 'stage1_final.pth'
    torch.save({
        'world_model': world_model.state_dict(),
        'vehicle_embedding': vehicle_embedding.state_dict()
    }, final_checkpoint)
    print(f"  最终模型已保存: {final_checkpoint}")

    print("\n" + "=" * 80)
    print("🎉 Stage 1 训练完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
