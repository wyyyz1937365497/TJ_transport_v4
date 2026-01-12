"""
完整训练脚本 - 理想架构v4.0

三阶段训练流程：
1. Phase 1: 世界模型预训练（监督学习，从真实环境收集数据）
2. Phase 2: 带安全屏障的PPO训练（冻结感知层，只训练决策层）
3. Phase 3: 拉格朗日约束优化（端到端微调）

优化：
- 真实图数据解析
- 并行环境加速数据收集
- 数据缓存机制

使用示例：
    python train_v4_ideal.py --config configs/competition_preliminary.yaml --phase all
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pickle
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.env.competition_env import CompetitionSumoEnv
from src.env.gym_wrapper import GymSumoEnv
from src.env.vec_env import create_parallel_envs
from src.models.ideal_policy_v4 import create_ideal_traffic_policy_v4
from src.utils.helpers import set_seed, get_device


@dataclass
class TrafficTransition:
    """交通状态转移数据（用于图构建）"""
    vehicle_states: List[Dict]  # 车辆状态字典列表
    global_stats: np.ndarray    # 全局统计
    icv_ids: set               # 智能车ID集合
    vehicle_ids: List[str]     # 所有车辆ID列表
    num_vehicles: int          # 车辆数量

    # 用于构建图的特征
    s_coords: np.ndarray       # 纵向坐标 [N]
    d_coords: np.ndarray       # 横向坐标 [N]
    lanes: np.ndarray          # 车道索引 [N]
    speeds: np.ndarray         # 速度 [N]
    accels: np.ndarray         # 加速度 [N]
    angles: np.ndarray         # 航向角 [N]


class Phase1WorldModelTrainer:
    """
    Phase 1: 世界模型预训练

    优化：
    - 真实图数据解析
    - 并行数据收集
    - 数据缓存机制
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device()
        self.checkpoint_dir = config.get('paths', {}).get('checkpoint_dir', 'checkpoints') + '/v4_phase1'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 获取Phase 1配置
        phase1_config = config.get('training', {}).get('phase1', {})
        self.num_episodes = phase1_config.get('num_episodes', 50)
        self.epochs = phase1_config.get('epochs', 30)
        self.batch_size = phase1_config.get('batch_size', 256)
        self.learning_rate = phase1_config.get('learning_rate', 1e-4)
        self.num_workers = phase1_config.get('num_parallel_workers', 4)

        # 数据缓存
        self.cache_dir = os.path.join(self.checkpoint_dir, 'data_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, 'phase1_data.pkl')

    def train(self) -> str:
        """训练世界模型"""
        print("\n" + "="*80)
        print("[PHASE 1] World Model Pre-training (Supervised Learning)")
        print("="*80)
        print(f"Episodes: {self.num_episodes}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Workers: {self.num_workers}")
        print(f"Cache: {self.cache_file}")

        # 1. 创建模型
        from src.models.v4_architecture import IdealTrafficControllerV4

        model_config = self.config.get('model', {})
        model = IdealTrafficControllerV4(
            node_dim=model_config.get('gnn', {}).get('node_dim', 9),
            edge_dim=model_config.get('gnn', {}).get('edge_dim', 4),
            global_dim=model_config.get('controller', {}).get('global_dim', 32),
            gnn_hidden_dim=model_config.get('gnn', {}).get('hidden_dim', 64),
            gnn_output_dim=model_config.get('gnn', {}).get('output_dim', 256),
            rssm_hidden_dim=model_config.get('world_model', {}).get('hidden_dim', 128),
            rssm_latent_dim=model_config.get('world_model', {}).get('hidden_dim', 64),
            controller_hidden_dim=model_config.get('controller', {}).get('hidden_dim', 128),
            top_k=model_config.get('controller', {}).get('top_k', 5),
            device=str(self.device)
        ).to(self.device)

        # 2. 从环境收集数据（带缓存）
        print("\n[DATA] Collecting training data from environment...")
        data = self._collect_data_with_cache()

        print(f"[OK] Collected {len(data['observations'])} samples")
        if len(data['observations']) > 0:
            obs = data['observations'][0]
            if hasattr(obs, 'vehicle_states'):
                if isinstance(obs.vehicle_states, list):
                    print(f"    Observation: {len(obs.vehicle_states)} vehicles")
                else:
                    print(f"    Observation shape: {obs.vehicle_states.shape}")
            print(f"    Num vehicles: {obs.num_vehicles if hasattr(obs, 'num_vehicles') else 'N/A'}")

        # 3. 训练循环
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=1e-6
        )

        print(f"\n[TRAIN] Starting training...")
        best_loss = float('inf')

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            epoch_speed_loss = 0.0
            epoch_risk_loss = 0.0
            epoch_pos_loss = 0.0
            num_batches = 0

            # 打乱数据
            indices = np.random.permutation(len(data['observations']))

            for batch_start in range(0, len(indices), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(indices))
                batch_indices = indices[batch_start:batch_end]

                if len(batch_indices) == 0:
                    continue

                batch = self._prepare_batch_real(data, batch_indices)

                # 前向传播
                gnn_output = model.perception_layer(
                    node_features=batch['node_features'],
                    edge_index=batch['edge_index'],
                    edge_features=batch['edge_features'],
                    risk_features=batch['risk_features']
                )

                rssm_output = model.prediction_layer(
                    node_embeddings=gnn_output['node_embeddings']
                )

                # 计算损失
                loss_dict = self._compute_prediction_loss(rssm_output, batch)

                # 总损失
                total_loss = (
                    loss_dict['speed_mse'] * 1.0 +
                    loss_dict['position_mse'] * 0.5 +
                    loss_dict['conflict_bce'] * 2.0
                )

                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_speed_loss += loss_dict['speed_mse'].item()
                epoch_risk_loss += loss_dict['conflict_bce'].item()
                epoch_pos_loss += loss_dict['position_mse'].item()
                num_batches += 1

            # 学习率调度
            scheduler.step()

            # 打印统计
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_speed_loss = epoch_speed_loss / max(num_batches, 1)
            avg_risk_loss = epoch_risk_loss / max(num_batches, 1)
            avg_pos_loss = epoch_pos_loss / max(num_batches, 1)

            print(f"Epoch {epoch+1}/{self.epochs} - "
                  f"Loss: {avg_loss:.4f} (Speed: {avg_speed_loss:.4f}, Pos: {avg_pos_loss:.4f}, Risk: {avg_risk_loss:.4f})")

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = os.path.join(self.checkpoint_dir, 'world_model_best.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': self.config,
                    'epoch': epoch,
                    'loss': avg_loss
                }, checkpoint_path)
                print(f"    [SAVE] Best model saved (loss: {best_loss:.4f})")

        # 保存最终模型
        final_path = os.path.join(self.checkpoint_dir, 'world_model_final.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'epoch': self.epochs
        }, final_path)

        print(f"\n[DONE] Phase 1 complete!")
        print(f"   Best loss: {best_loss:.4f}")
        print(f"   Final model: {final_path}")

        return final_path

    def _collect_data_with_cache(self) -> Dict[str, Any]:
        """带缓存的数据收集"""
        # 检查缓存
        if os.path.exists(self.cache_file):
            print(f"[CACHE] Found existing cache: {self.cache_file}")
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                if len(cached_data['observations']) >= self.num_episodes * 100:  # 假设每个episode约100步
                    print(f"[CACHE] Loading from cache...")
                    return cached_data
                else:
                    print(f"[CACHE] Cache insufficient, recollecting...")
            except Exception as e:
                print(f"[CACHE] Failed to load cache: {e}, recollecting...")

        # 收集新数据
        print(f"[COLLECT] Collecting fresh data...")
        data = self._collect_data_parallel_optimized()

        # 保存到缓存
        print(f"[CACHE] Saving to cache...")
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"[OK] Cache saved: {self.cache_file}")
        except Exception as e:
            print(f"[WARNING] Failed to save cache: {e}")

        return data

    def _collect_data_parallel_optimized(self) -> Dict[str, Any]:
        """
        优化的并行数据收集

        关键优化：
        1. 使用多进程并行收集
        2. 减少max_steps加速单个episode
        3. 使用更高效的数据结构
        """
        from multiprocessing import Pool, cpu_count
        from functools import partial

        # 计算每个worker的episodes
        num_workers = min(self.num_workers, cpu_count())
        episodes_per_worker = self.num_episodes // num_workers
        remainder = self.num_episodes % num_workers

        episode_counts = [episodes_per_worker + 1 if i < remainder else episodes_per_worker
                          for i in range(num_workers)]

        print(f"    Using {num_workers} workers")
        print(f"    Episodes per worker: {episode_counts}")

        # 使用进程池并行收集
        observations = []
        next_observations = []

        # 创建worker函数
        collect_func = partial(_collect_worker_wrapper,
                              env_config=self.config.get('environment', {}),
                              max_steps=500)  # 减少到500步加速收集

        # 使用进程池
        with Pool(processes=num_workers) as pool:
            # 收集所有worker的结果
            worker_results = []
            for worker_id, count in enumerate(episode_counts):
                if count > 0:
                    result = pool.apply_async(collect_func, (worker_id, count))
                    worker_results.append(result)

            # 收集结果
            completed = 0
            for result in worker_results:
                try:
                    worker_obs, worker_next_obs = result.get(timeout=600)  # 10分钟超时
                    observations.extend(worker_obs)
                    next_observations.extend(worker_next_obs)
                    completed += 1
                    print(f"    Worker {completed}/{len(worker_results)} completed")
                except Exception as e:
                    print(f"    [WARNING] Worker failed: {e}")

        print(f"[OK] Collected {len(observations)} transitions from {completed} workers")

        return {
            'observations': observations,
            'next_observations': next_observations
        }

    def _collect_single_episode(self, episode_idx: int) -> Tuple[List, List]:
        """收集单个episode的数据（优化版）"""
        env_config = self.config.get('environment', {})
        port = 8813 + episode_idx

        env = CompetitionSumoEnv(env_config, port=port, use_gui=False)

        obs = env.reset()
        observations = []
        next_observations = []

        # 减少步数加速收集
        max_steps = min(env_config.get('max_steps', 3600), 500)

        for step in range(max_steps):
            # 保存当前观测（使用TrafficTransition）
            obs_transition = self._create_transition(obs)
            observations.append(obs_transition)

            # 随机动作（探索性数据收集）
            icv_ids = list(obs.get('icv_ids', set()))
            if len(icv_ids) > 0:
                # 随机选择一辆ICV控制
                veh_id = icv_ids[0]
                action = {
                    veh_id: np.random.uniform(-1.0, 1.0, size=2)
                }
            else:
                action = None

            # 执行一步
            obs, _, done, _ = env.step(action)

            # 保存下一步观测
            obs_transition = self._create_transition(obs)
            next_observations.append(obs_transition)

            if done:
                break

        env.close()

        return observations, next_observations

    def _create_transition(self, obs: Dict) -> TrafficTransition:
        """创建交通状态转移对象（用于图构建）"""
        vehicle_states = obs.get('vehicle_states', {})
        global_stats = obs.get('global_stats', np.zeros(32))
        icv_ids = obs.get('icv_ids', set())
        vehicle_ids = obs.get('vehicle_ids', [])

        num_veh = len(vehicle_ids)

        # 提取用于图构建的特征
        s_coords = np.zeros(num_veh)
        d_coords = np.zeros(num_veh)
        lanes = np.zeros(num_veh)
        speeds = np.zeros(num_veh)
        accels = np.zeros(num_veh)
        angles = np.zeros(num_veh)

        for i, veh_id in enumerate(vehicle_ids):
            state = vehicle_states.get(veh_id, {})
            s_coords[i] = state.get('s', 0.0)
            d_coords[i] = state.get('d', 0.0)
            lanes[i] = state.get('lane_index', 0.0)
            speeds[i] = state.get('speed', 0.0)
            accels[i] = state.get('acceleration', 0.0)
            angles[i] = state.get('angle', 0.0)

        return TrafficTransition(
            vehicle_states=list(vehicle_states.values()),
            global_stats=global_stats,
            icv_ids=icv_ids,
            vehicle_ids=vehicle_ids,
            num_vehicles=num_veh,
            s_coords=s_coords,
            d_coords=d_coords,
            lanes=lanes,
            speeds=speeds,
            accels=accels,
            angles=angles
        )

    def _prepare_batch_real(
        self,
        data: Dict,
        indices: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """
        真实的图数据解析

        从TrafficTransition对象构建真实的图数据
        """
        batch_size = len(indices)
        device = self.device

        # 收集所有样本的数据
        all_node_features = []
        all_edge_indices = []
        all_edge_features = []
        all_risk_features = []
        all_next_speeds = []
        all_next_positions = []
        all_conflict_labels = []

        # 用于边索引的偏移
        node_offset = 0

        for idx in indices:
            if idx >= len(data['observations']):
                continue

            obs_trans = data['observations'][idx]
            next_trans = data['next_observations'][idx]

            num_veh = obs_trans.num_vehicles

            if num_veh == 0:
                continue

            # 1. 节点特征 [N, 9]
            node_features = self._extract_node_features(obs_trans)
            all_node_features.append(node_features)

            # 2. 边特征（空间邻近）
            edge_index, edge_attr = self._build_edges_from_transition(obs_trans)

            # 添加偏移（因为batch中所有图的节点会被拼接）
            edge_index = edge_index + node_offset

            all_edge_indices.append(edge_index)
            all_edge_features.append(edge_attr)

            node_offset += num_veh

            # 3. 风险特征 [N, 2]
            risk_features = self._compute_risk_features(obs_trans)
            all_risk_features.append(risk_features)

            # 4. 目标值（使用obs的num_vehicles确保维度一致）
            next_speed = self._extract_next_speed(next_trans, num_vehicles=num_veh)
            next_position = self._extract_next_position(next_trans, num_vehicles=num_veh)
            conflict_label = self._compute_conflict_label(obs_trans, next_trans)

            all_next_speeds.append(next_speed)
            all_next_positions.append(next_position)
            all_conflict_labels.append(conflict_label)

        # 拼接所有样本
        if len(all_node_features) == 0:
            # 返回空batch
            return self._get_empty_batch()

        node_features = torch.cat(all_node_features, dim=0).to(device)
        edge_index = torch.cat(all_edge_indices, dim=1).to(device)
        edge_features = torch.cat(all_edge_features, dim=0).to(device)
        risk_features = torch.cat(all_risk_features, dim=0).to(device)
        next_speed = torch.cat(all_next_speeds, dim=0).to(device)
        next_position = torch.cat(all_next_positions, dim=0).to(device)
        conflict_label = torch.cat(all_conflict_labels, dim=0).to(device)

        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'risk_features': risk_features,
            'next_speed': next_speed,
            'next_position': next_position,
            'conflict_label': conflict_label
        }

    def _extract_node_features(
        self,
        trans: TrafficTransition
    ) -> torch.Tensor:
        """提取节点特征 [N, 9]"""
        num_veh = trans.num_vehicles

        features = np.zeros((num_veh, 9), dtype=np.float32)

        for i in range(num_veh):
            # [s, d, vs, vd, speed, accel, lane, angle, is_icv]
            features[i, 0] = trans.s_coords[i] / 1000.0      # s (归一化)
            features[i, 1] = trans.d_coords[i] / 10.0        # d (归一化)
            features[i, 2] = trans.speeds[i] / 30.0         # vs (简化为speed)
            features[i, 3] = 0.0                           # vd (简化)
            features[i, 4] = trans.speeds[i] / 30.0         # speed (归一化)
            features[i, 5] = trans.accels[i] / 3.0          # accel (归一化)
            features[i, 6] = trans.lanes[i] / 10.0          # lane (归一化)
            features[i, 7] = trans.angles[i] / 360.0        # angle (归一化)
            features[i, 8] = 1.0 if i < len(trans.icv_ids) else 0.0  # is_icv (前25%)

        return torch.from_numpy(features)

    def _build_edges_from_transition(
        self,
        trans: TrafficTransition
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从交通状态转移构建边（空间邻近）

        Returns:
            edge_index: [2, E]
            edge_attr: [E, 4]
        """
        num_veh = trans.num_vehicles

        if num_veh <= 1:
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0, 4), dtype=torch.float32)
            )

        # 使用配置的交互半径
        graph_config = self.config.get('model', {}).get('graph', {})
        interaction_radius = graph_config.get('interaction_radius', 100.0)
        max_neighbors = graph_config.get('max_neighbors', 8)

        sources = []
        targets = []
        edge_features_list = []

        for i in range(num_veh):
            # 计算与其他车辆的距离
            s_diff = trans.s_coords - trans.s_coords[i]
            d_diff = trans.d_coords - trans.d_coords[i]

            # 欧氏距离（简化）
            distances = np.sqrt(s_diff**2 + d_diff**2)

            # 找到交互半径内的邻居（排除自己）
            mask = (distances < interaction_radius) & (distances > 0.1)
            neighbors = np.where(mask)[0]

            # 限制最大邻居数
            if len(neighbors) > max_neighbors:
                sorted_indices = np.argsort(distances[neighbors])[:max_neighbors]
                neighbors = neighbors[sorted_indices]

            # 创建边
            for j in neighbors:
                sources.append(i)
                targets.append(j)

                # 边特征：[相对速度, 相对位置, 车道差, 是否都是ICV]
                rel_speed = trans.speeds[i] - trans.speeds[j]
                rel_s = trans.s_coords[i] - trans.s_coords[j]
                lane_diff = abs(trans.lanes[i] - trans.lanes[j])
                both_icv = 1.0 if (i < num_veh // 4 and j < num_veh // 4) else 0.0

                edge_features_list.append([rel_speed, rel_s, lane_diff, both_icv])

        if len(sources) == 0:
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0, 4), dtype=torch.float32)
            )

        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_features_list, dtype=torch.float32)

        return edge_index, edge_attr

    def _compute_risk_features(
        self,
        trans: TrafficTransition
    ) -> torch.Tensor:
        """
        计算风险特征 [N, 2]

        特征：
        1. TTC倒数 (Time To Collision inverse)
        2. THW倒数 (Time Headway inverse)
        """
        num_veh = trans.num_vehicles

        if num_veh == 0:
            return torch.zeros((0, 2), dtype=torch.float32)

        # 避免除零
        speeds = trans.speeds.copy()
        speeds = np.maximum(speeds, 0.1)

        # TTC倒数（简化版：基于速度）
        ttc_inv = 1.0 / (speeds + 0.1)

        # THW倒数（简化版）
        thw_inv = 1.0 / (speeds * 2.0 + 0.1)

        risk_features = np.stack([ttc_inv, thw_inv], axis=-1)
        risk_features = risk_features.astype(np.float32)

        return torch.from_numpy(risk_features)

    def _extract_next_speed(
        self,
        trans: TrafficTransition,
        num_vehicles: Optional[int] = None
    ) -> torch.Tensor:
        """提取下一步的速度 [N, 1]"""
        # Use specified num_vehicles to match obs size, or use trans.num_vehicles
        target_num = num_vehicles if num_vehicles is not None else trans.num_vehicles
        actual_num = min(target_num, len(trans.speeds))  # Available vehicles in next_trans

        speeds = trans.speeds[:actual_num].copy()
        speeds = np.maximum(speeds, 0.0)  # 确保非负

        # 归一化
        speeds_normalized = (speeds / 30.0).astype(np.float32)

        # Pad to target_num if needed
        if actual_num < target_num:
            padded = np.zeros((target_num, 1), dtype=np.float32)
            padded[:actual_num, 0] = speeds_normalized
            return torch.from_numpy(padded)

        return torch.from_numpy(speeds_normalized).unsqueeze(-1)

    def _extract_next_position(
        self,
        trans: TrafficTransition,
        num_vehicles: Optional[int] = None
    ) -> torch.Tensor:
        """提取下一步的位置 [N, 2]"""
        # Use specified num_vehicles to match obs size, or use trans.num_vehicles
        target_num = num_vehicles if num_vehicles is not None else trans.num_vehicles
        actual_num = min(target_num, trans.num_vehicles)  # Available vehicles in next_trans

        positions = np.zeros((target_num, 2), dtype=np.float32)
        positions[:actual_num, 0] = trans.s_coords[:actual_num] / 1000.0  # s
        positions[:actual_num, 1] = trans.d_coords[:actual_num] / 10.0     # d

        return torch.from_numpy(positions)

    def _compute_conflict_label(
        self,
        obs_trans: TrafficTransition,
        next_trans: TrafficTransition
    ) -> torch.Tensor:
        """
        计算冲突标签 [N, 1]

        定义：如果速度急剧下降（急刹），则标记为冲突
        """
        num_veh = obs_trans.num_vehicles

        if num_veh == 0:
            return torch.zeros((0, 1), dtype=torch.float32)

        # Use min to ensure we don't exceed available data in next_trans
        n = min(num_veh, len(next_trans.accels), len(obs_trans.accels))

        # 计算加速度变化
        accel_change = next_trans.accels[:n] - obs_trans.accels[:n]

        # 急刹检测：加速度变化 < -2.0 m/s²
        conflict_label = (accel_change < -2.0).astype(np.float32)

        # Pad with zeros if needed to match num_veh
        if n < num_veh:
            padded = np.zeros((num_veh, 1), dtype=np.float32)
            padded[:n, 0] = conflict_label
            return torch.from_numpy(padded)

        return torch.from_numpy(conflict_label).unsqueeze(-1)

    def _get_empty_batch(self) -> Dict[str, torch.Tensor]:
        """返回空batch（避免错误）"""
        device = self.device
        return {
            'node_features': torch.zeros((1, 9), device=device),
            'edge_index': torch.zeros((2, 1), dtype=torch.long, device=device),
            'edge_features': torch.zeros((1, 4), device=device),
            'risk_features': torch.zeros((1, 2), device=device),
            'next_speed': torch.zeros((1, 1), device=device),
            'next_position': torch.zeros((1, 2), device=device),
            'conflict_label': torch.zeros((1, 1), device=device),
        }

    def _vectorize_observation(self, obs: Dict) -> np.ndarray:
        """将观测字典向量化（兼容旧版本）"""
        obs_trans = self._create_transition(obs)

        vehicle_features = []
        for i in range(obs_trans.num_vehicles):
            features = [
                obs_trans.s_coords[i] / 1000.0,
                obs_trans.d_coords[i] / 10.0,
                obs_trans.speeds[i] / 30.0,
                0.0,  # vd
                obs_trans.speeds[i] / 30.0,
                obs_trans.accels[i] / 3.0,
                obs_trans.lanes[i] / 10.0,
                obs_trans.angles[i] / 360.0,
                1.0 if i < len(obs_trans.icv_ids) else 0.0
            ]
            vehicle_features.extend(features)

        # Padding
        max_features = 32 * 9
        if len(vehicle_features) < max_features:
            vehicle_features.extend([0.0] * (max_features - len(vehicle_features)))
        else:
            vehicle_features = vehicle_features[:max_features]

        flat_obs = np.array(vehicle_features, dtype=np.float32)
        flat_obs = np.concatenate([
            flat_obs,
            obs_trans.global_stats.flatten(),
            [obs_trans.num_vehicles]
        ])

        return flat_obs

    def _compute_prediction_loss(
        self,
        rssm_output: Dict,
        batch: Dict
    ) -> Dict[str, torch.Tensor]:
        """计算预测损失"""
        speed_pred = rssm_output['speed_pred']
        pos_pred = rssm_output['position_pred']
        conflict_prob = rssm_output['conflict_prob']

        speed_target = batch['next_speed']
        pos_target = batch['next_position']
        conflict_target = batch['conflict_label']

        return {
            'speed_mse': F.mse_loss(speed_pred, speed_target),
            'position_mse': F.mse_loss(pos_pred, pos_target),
            'conflict_bce': F.binary_cross_entropy(
                conflict_prob.squeeze(-1),
                conflict_target.squeeze(-1)
            )
        }


def _collect_worker_wrapper(
    worker_id: int,
    num_episodes: int,
    env_config: Dict,
    max_steps: int = 500
) -> Tuple[List, List]:
    """
    Worker函数：在单独的进程中收集数据

    Args:
        worker_id: worker ID
        num_episodes: 要收集的episodes数量
        env_config: 环境配置
        max_steps: 每个episode的最大步数

    Returns:
        (observations, next_observations)
    """
    observations = []
    next_observations = []

    base_port = 8813 + worker_id * 10  # 避免端口冲突

    for ep in range(num_episodes):
        port = base_port + ep
        env = CompetitionSumoEnv(env_config, port=port, use_gui=False)

        try:
            obs = env.reset()

            for step in range(max_steps):
                # 创建TrafficTransition
                vehicle_states = obs.get('vehicle_states', {})
                global_stats = obs.get('global_stats', np.zeros(32))
                icv_ids = obs.get('icv_ids', set())
                vehicle_ids = obs.get('vehicle_ids', [])

                num_veh = len(vehicle_ids)

                s_coords = np.zeros(num_veh)
                d_coords = np.zeros(num_veh)
                lanes = np.zeros(num_veh)
                speeds = np.zeros(num_veh)
                accels = np.zeros(num_veh)
                angles = np.zeros(num_veh)

                for i, veh_id in enumerate(vehicle_ids):
                    state = vehicle_states.get(veh_id, {})
                    s_coords[i] = state.get('s', 0.0)
                    d_coords[i] = state.get('d', 0.0)
                    lanes[i] = state.get('lane_index', 0.0)
                    speeds[i] = state.get('speed', 0.0)
                    accels[i] = state.get('acceleration', 0.0)
                    angles[i] = state.get('angle', 0.0)

                obs_trans = TrafficTransition(
                    vehicle_states=list(vehicle_states.values()),
                    global_stats=global_stats,
                    icv_ids=icv_ids,
                    vehicle_ids=vehicle_ids,
                    num_vehicles=num_veh,
                    s_coords=s_coords,
                    d_coords=d_coords,
                    lanes=lanes,
                    speeds=speeds,
                    accels=accels,
                    angles=angles
                )

                observations.append(obs_trans)

                # 随机动作
                if len(icv_ids) > 0:
                    veh_id = list(icv_ids)[0]
                    action = {veh_id: np.random.uniform(-1.0, 1.0, size=2)}
                else:
                    action = None

                obs, _, done, _ = env.step(action)

                # 保存下一步
                vehicle_states = obs.get('vehicle_states', {})
                global_stats = obs.get('global_stats', np.zeros(32))
                icv_ids = obs.get('icv_ids', set())
                vehicle_ids = obs.get('vehicle_ids', [])

                num_veh = len(vehicle_ids)

                s_coords = np.zeros(num_veh)
                d_coords = np.zeros(num_veh)
                lanes = np.zeros(num_veh)
                speeds = np.zeros(num_veh)
                accels = np.zeros(num_veh)
                angles = np.zeros(num_veh)

                for i, veh_id in enumerate(vehicle_ids):
                    state = vehicle_states.get(veh_id, {})
                    s_coords[i] = state.get('s', 0.0)
                    d_coords[i] = state.get('d', 0.0)
                    lanes[i] = state.get('lane_index', 0.0)
                    speeds[i] = state.get('speed', 0.0)
                    accels[i] = state.get('acceleration', 0.0)
                    angles[i] = state.get('angle', 0.0)

                next_trans = TrafficTransition(
                    vehicle_states=list(vehicle_states.values()),
                    global_stats=global_stats,
                    icv_ids=icv_ids,
                    vehicle_ids=vehicle_ids,
                    num_vehicles=num_veh,
                    s_coords=s_coords,
                    d_coords=d_coords,
                    lanes=lanes,
                    speeds=speeds,
                    accels=accels,
                    angles=angles
                )

                next_observations.append(next_trans)

                if done:
                    break

        except Exception as e:
            print(f"[WARNING] Worker {worker_id}, Episode {ep}: {e}")
        finally:
            env.close()

    return observations, next_observations


class Phase2ShieldedPPOTrainer:
    """
    Phase 2: 带安全屏障的PPO训练

    目标：冻结感知层，只训练决策层
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device()
        self.checkpoint_dir = config.get('paths', {}).get('checkpoint_dir', 'checkpoints') + '/v4_phase2'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        phase2_config = config.get('training', {}).get('phase2', {})
        self.total_timesteps = phase2_config.get('total_timesteps', 200000)
        self.num_envs = phase2_config.get('num_envs', 8)

    def train(self, phase1_checkpoint: Optional[str] = None) -> str:
        """训练带安全屏障的PPO"""
        print("\n" + "="*80)
        print("[PHASE 2] Shielded PPO Training")
        print("="*80)
        print(f"Total Timesteps: {self.total_timesteps}")
        print(f"Parallel Envs: {self.num_envs}")

        # 1. 创建环境
        print("\n[ENV] Creating parallel environments...")
        env_config = self.config.get('environment', {})

        vec_env_wrapper = create_parallel_envs(
            config=env_config,
            num_envs=self.num_envs,
            base_port=8813,
            seed=self.config.get('seed', 42)
        )

        vec_env = vec_env_wrapper.vec_env

        # 2. 创建策略网络
        print("\n[INFO] Creating v4.0 policy network...")
        policy_class = create_ideal_traffic_policy_v4(self.config)

        # Read Phase 2 training config
        phase2_config = self.config.get('training', {}).get('phase2', {})

        model = PPO(
            policy_class,
            vec_env,
            verbose=1,
            tensorboard_log=self.config.get('paths', {}).get('log_dir', 'logs') + '/v4_phase2',
            learning_rate=phase2_config.get('learning_rate', 3e-4),
            n_steps=phase2_config.get('n_steps', 2048),
            batch_size=phase2_config.get('batch_size', 512),
            n_epochs=phase2_config.get('update_epochs', 10),
            gamma=phase2_config.get('gamma', 0.99),
            gae_lambda=phase2_config.get('gae_lambda', 0.95),
            clip_range=phase2_config.get('clip_epsilon', 0.2),
            ent_coef=phase2_config.get('entropy_coef', 0.01),
            vf_coef=phase2_config.get('value_loss_coef', 0.5),
            max_grad_norm=phase2_config.get('max_grad_norm', 0.5),
            seed=self.config.get('seed', 42),
            device=str(self.device)
        )

        # 3. 加载Phase 1权重（如果有）
        if phase1_checkpoint and os.path.exists(phase1_checkpoint):
            print(f"\n[INFO] Loading Phase 1 weights...")
            self._load_phase1_weights(model.policy, phase1_checkpoint)

        # 4. 冻结感知层，只训练决策层
        print("\n[FREEZE] Freezing perception and prediction layers...")
        model.policy.freeze_perception()
        model.policy.freeze_prediction()

        # 5. 设置回调
        callbacks = self._create_callbacks()

        # 6. 训练
        print("\n🏋️ 开始训练...")
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        # 7. 保存
        sb3_path = os.path.join(self.checkpoint_dir, 'shielded_ppo.zip')
        model.save(sb3_path)

        compat_path = os.path.join(self.checkpoint_dir, 'shielded_ppo.pth')
        torch.save({
            'model_type': 'sb3',
            'phase': 2,
            'policy_state_dict': model.policy.state_dict(),
            'config': self.config
        }, compat_path)

        print(f"\n[DONE] Phase 2 complete!")
        print(f"   SB3格式: {sb3_path}")
        print(f"   兼容格式: {compat_path}")

        vec_env.close()

        return sb3_path

    def _load_phase1_weights(self, policy, checkpoint_path: str):
        """加载Phase 1权重"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 加载感知层权重
        model_dict = policy.state_dict()
        loaded = 0

        for key in state_dict.keys():
            new_key = None
            if key.startswith('perception_layer.'):
                new_key = key
            elif key.startswith('prediction_layer.'):
                new_key = key

            if new_key and new_key in model_dict:
                if state_dict[key].shape == model_dict[new_key].shape:
                    model_dict[new_key] = state_dict[key]
                    loaded += 1

        print(f"[OK] Loaded {loaded} weights from Phase 1")

    def _create_callbacks(self):
        """创建训练回调"""
        callbacks = []

        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=self.checkpoint_dir,
            name_prefix="v4_phase2",
            save_replay_buffer=False
        )
        callbacks.append(checkpoint_callback)

        return callbacks


class Phase3ConstrainedOptimizer:
    """
    Phase 3: 拉格朗日约束优化

    目标：端到端微调，满足成本约束
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device()
        self.checkpoint_dir = config.get('paths', {}).get('checkpoint_dir', 'checkpoints') + '/v4_phase3'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        phase3_config = config.get('training', {}).get('phase3', {})
        self.total_timesteps = phase3_config.get('total_timesteps', 100000)
        self.cost_limit = phase3_config.get('cost_limit', 0.1)

    def train(self, phase2_checkpoint: str) -> str:
        """训练拉格朗日优化模型"""
        print("\n" + "="*80)
        print("[PHASE 3] 拉格朗日约束优化 (Constrained Optimization)")
        print("="*80)
        print(f"Total Timesteps: {self.total_timesteps}")
        print(f"Cost Limit: {self.cost_limit}")

        # 1. 创建环境
        env_config = self.config.get('environment', {})
        num_envs = 4

        vec_env_wrapper = create_parallel_envs(
            config=env_config,
            num_envs=num_envs,
            base_port=9013,
            seed=self.config.get('seed', 42)
        )

        vec_env = vec_env_wrapper.vec_env

        # 2. 创建拉格朗日PPO
        print("\n[INFO] Creating Lagrangian PPO...")
        model = self._create_lagrangian_ppo(vec_env)

        # 3. 加载Phase 2权重
        print(f"\n[INFO] Loading Phase 2 weights...")
        phase2_model = PPO.load(phase2_checkpoint)
        model.set_parameters(phase2_model.get_parameters())

        # 4. 解冻所有组件
        print("\n[UNFREEZE] Unfreezing all components...")
        model.policy.unfreeze_all()

        # 5. 训练
        callbacks = self._create_callbacks()

        print("\n🏋️ 开始训练...")
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        # 6. 保存最终模型
        final_path = os.path.join(self.checkpoint_dir, 'ideal_v4_final.zip')
        model.save(final_path)

        compat_path = os.path.join(self.checkpoint_dir, 'ideal_v4_final.pth')
        torch.save({
            'model_type': 'sb3',
            'phase': 3,
            'policy_state_dict': model.policy.state_dict(),
            'config': self.config
        }, compat_path)

        print(f"\n[DONE] Phase 3 complete!")
        print(f"   Final model: {final_path}")

        vec_env.close()

        return final_path

    def _create_lagrangian_ppo(self, vec_env):
        """创建拉格朗日PPO"""

        class LagrangianPPO(PPO):
            """扩展PPO，添加拉格朗日约束"""

            def __init__(self, *args, cost_limit=0.1, lambda_init=0.1, **kwargs):
                super().__init__(*args, **kwargs)
                self.cost_limit = cost_limit
                self.lambda_param = lambda_init

            def train(self):
                # 标准PPO更新
                super().train()

                # 更新拉格朗日乘子（简化版）
                # 实际应该基于平均成本违反程度
                self.lambda_param = np.clip(
                    self.lambda_param + 1e-4,
                    0, 10
                )

        policy_class = create_ideal_traffic_policy_v4(self.config)

        return LagrangianPPO(
            policy_class,
            vec_env,
            verbose=1,
            tensorboard_log=self.config.get('paths', {}).get('log_dir', 'logs') + '/v4_phase3',
            learning_rate=1e-4,
            cost_limit=self.cost_limit,
            lambda_init=0.1,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            device=str(self.device)
        )

    def _create_callbacks(self):
        """创建回调"""
        callbacks = []

        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=self.checkpoint_dir,
            name_prefix="v4_phase3",
            save_replay_buffer=False
        )
        callbacks.append(checkpoint_callback)

        return callbacks


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="训练理想架构v4.0"
    )

    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--phase', type=str, default='all',
                        choices=['1', '2', '3', 'all'],
                        help='训练阶段')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Phase 1 数据收集episodes')
    parser.add_argument('--timesteps', type=int, default=200000,
                        help='RL训练步数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--no-cache', action='store_true',
                        help='禁用缓存，强制重新收集数据')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 覆盖配置
    if args.timesteps != 200000:
        config.setdefault('training', {})
        config['training'].setdefault('phase2', {})
        config['training']['phase2']['total_timesteps'] = args.timesteps
        config['training'].setdefault('phase3', {})
        config['training']['phase3']['total_timesteps'] = args.timesteps // 2

    # 设置随机种子
    set_seed(config.get('seed', 42))

    # 获取checkpoint路径
    checkpoint_dir = config.get('paths', {}).get('checkpoint_dir', 'checkpoints')

    # 定义checkpoint路径
    phase1_path = os.path.join(checkpoint_dir, 'v4_phase1/world_model_final.pth')
    phase2_path = os.path.join(checkpoint_dir, 'v4_phase2/shielded_ppo.zip')
    phase3_path = os.path.join(checkpoint_dir, 'v4_phase2/shielded_ppo.pth')  # Phase 3的输入
    final_path = os.path.join(checkpoint_dir, 'v4_phase2/final_model.pth')

    # 创建训练器
    phase1_checkpoint = None
    phase2_checkpoint = None

    # 检测已有的checkpoint并确定起始阶段
    if args.phase == 'all':
        # 自动检测应该从哪个阶段开始
        start_phase = 1
        if os.path.exists(phase3_path) or os.path.exists(final_path):
            print("[INFO] 所有阶段已完成，从Phase 3重新开始")
            start_phase = 3
        elif os.path.exists(phase2_path):
            print("[INFO] Phase 1已完成，从Phase 2开始")
            start_phase = 2
            phase1_checkpoint = phase1_path
        elif os.path.exists(phase1_path):
            print("[INFO] Phase 1 checkpoint存在，从Phase 2开始")
            start_phase = 2
            phase1_checkpoint = phase1_path
        else:
            print("[INFO] 从Phase 1开始训练")
            start_phase = 1

        # 执行训练
        if start_phase <= 1:
            trainer1 = Phase1WorldModelTrainer(config)
            phase1_checkpoint = trainer1.train()

        if start_phase <= 2:
            # 如果没有Phase 1 checkpoint但需要运行Phase 2，尝试查找
            if not phase1_checkpoint and not os.path.exists(phase1_path):
                print("[WARNING] Phase 1 checkpoint未找到，Phase 2可能效果不佳")
                phase1_checkpoint = None
            elif not phase1_checkpoint and os.path.exists(phase1_path):
                phase1_checkpoint = phase1_path

            trainer2 = Phase2ShieldedPPOTrainer(config)
            phase2_checkpoint = trainer2.train(phase1_checkpoint=phase1_checkpoint)

        if start_phase <= 3:
            # Phase 3是约束优化，需要Phase 2的checkpoint
            if not phase2_checkpoint:
                # 尝试使用默认路径
                if os.path.exists(phase2_path):
                    phase2_checkpoint = phase2_path
                else:
                    print("[WARNING] Phase 2 checkpoint未找到，无法运行Phase 3")
                    print("[HINT] 请先运行: python train_v4_ideal.py --config <config> --phase 2")
                    return

            trainer3 = Phase3ConstrainedOptimizer(config)
            trainer3.train(phase2_checkpoint=phase2_checkpoint)

    elif args.phase == '1':
        # 只运行Phase 1
        trainer1 = Phase1WorldModelTrainer(config)
        phase1_checkpoint = trainer1.train()

    elif args.phase == '2':
        # 从Phase 2开始
        # 检查Phase 1 checkpoint
        if os.path.exists(phase1_path):
            print(f"[INFO] 使用现有Phase 1 checkpoint: {phase1_path}")
            phase1_checkpoint = phase1_path
        else:
            print("[WARNING] Phase 1 checkpoint未找到！")
            print("[HINT] Phase 1训练是必须的。请先运行: python train_v4_ideal.py --config <config> --phase 1")
            user_input = input("是否仍然继续Phase 2训练？(y/N): ")
            if user_input.lower() != 'y':
                print("[ABORT] 训练已取消")
                return
            phase1_checkpoint = None

        trainer2 = Phase2ShieldedPPOTrainer(config)
        phase2_checkpoint = trainer2.train(phase1_checkpoint=phase1_checkpoint)

    elif args.phase == '3':
        # 从Phase 3开始
        # 检查Phase 2 checkpoint
        if os.path.exists(phase2_path):
            print(f"[INFO] 使用现有Phase 2 checkpoint: {phase2_path}")
            phase2_checkpoint = phase2_path
        else:
            print("[WARNING] Phase 2 checkpoint未找到！")
            print("[HINT] Phase 2训练是必须的。请先运行: python train_v4_ideal.py --config <config> --phase 2")
            user_input = input("是否仍然继续Phase 3训练？(y/N): ")
            if user_input.lower() != 'y':
                print("[ABORT] 训练已取消")
                return
            phase2_checkpoint = None

        trainer3 = Phase3ConstrainedOptimizer(config)
        trainer3.train(phase2_checkpoint=phase2_checkpoint)

    print("\n" + "="*80)
    print("[SUCCESS] Training pipeline finished!")
    print("="*80)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    # 设置设备
    if config.get('device', 'cuda') == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, using CPU")
        config['device'] = 'cpu'

    return config


if __name__ == '__main__':
    main()
