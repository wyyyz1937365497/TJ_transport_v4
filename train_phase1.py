"""
Phase 1: 世界模型预训练（监督学习）

训练目标：
- 学习车辆状态编码器（RiskSensitiveGNN）
- 学习交通流预测器（MultiScaleRSSM）
- 学习动态权重门控（EnhancedDynamicWeightGating）
- 预训练特征提取器，为Phase 2的PPO训练提供良好初始化

使用方法：
    # 基础训练
    python train_phase1.py

    # 使用自定义配置
    python train_phase1.py --config configs/competition_preliminary.yaml

    # 指定设备
    python train_phase1.py --device cuda:0
"""

import os
import sys
import time
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import multiprocessing

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.v4_architecture import IdealTrafficControllerV4
from src.env.competition_env import CompetitionSumoEnv
from src.training.multi_gpu_utils import (
    MultiGPUManager,
    ProgressTracker,
    DataCollectionProgress,
    TrainingMetrics
)
from src.utils.helpers import set_seed, get_device


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class TrafficTransition:
    """交通状态转移数据（用于图构建）"""
    vehicle_states: List[Dict]
    global_stats: np.ndarray
    icv_ids: set
    vehicle_ids: List[str]
    num_vehicles: int
    s_coords: np.ndarray
    d_coords: np.ndarray
    lanes: np.ndarray
    speeds: np.ndarray
    accels: np.ndarray
    angles: np.ndarray


# =============================================================================
# Phase 1: 世界模型预训练
# =============================================================================

class Phase1WorldModelTrainer:
    """Phase 1: 世界模型预训练（增强版 + 多进程支持）"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 初始化多GPU管理器
        self.gpu_manager = MultiGPUManager(config)
        self.device = self.gpu_manager.device

        # 初始化训练指标记录器
        self.metrics = TrainingMetrics()

        # Phase 1配置
        phase1_config = config.get('training', {}).get('phase1', {})

        # 检查点路径
        self.model_path = phase1_config.get('phase1_model_path',
            'checkpoints/competition/phase1/world_model_final.pth')
        self.checkpoint_dir = os.path.dirname(self.model_path)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.num_episodes = phase1_config.get('num_episodes', 100)
        self.epochs = phase1_config.get('epochs', 50)
        self.batch_size = phase1_config.get('batch_size', 256)
        self.learning_rate = phase1_config.get('learning_rate', 1e-4)
        self.num_workers = phase1_config.get('num_parallel_workers', 4)

        # 数据缓存
        self.cache_dir = os.path.join(self.checkpoint_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, 'data.pkl')

        # 课程学习配置
        curriculum_config = config.get('training', {}).get('curriculum', {})
        self.use_curriculum = curriculum_config.get('enabled', False)
        if self.use_curriculum:
            from src.training.train_enhancements import CurriculumManager
            self.curriculum = CurriculumManager(config.get('curriculum_levels', []))
        else:
            self.curriculum = None

        # 打印配置
        print(f"\n[PHASE 1] Configuration:")
        print(f"  Episodes: {self.num_episodes}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Parallel Workers: {self.num_workers}")
        print(f"  Effective Batch Size: {self.gpu_manager.get_effective_batch_size(self.batch_size)}")
        print(f"  Curriculum Learning: {'ENABLED' if self.use_curriculum else 'DISABLED'}")
        if self.use_curriculum:
            print(f"     Levels: {len(self.curriculum.levels)}")

    def train(self) -> str:
        """训练世界模型"""
        print("\n" + "="*80)
        print("[PHASE 1] World Model Pre-training")
        print("="*80)

        # 1. 创建模型
        model_config = self.config.get('model', {})
        model = IdealTrafficControllerV4(
            node_dim=model_config.get('gnn', {}).get('node_dim', 9),
            edge_dim=model_config.get('gnn', {}).get('edge_dim', 4),
            global_dim=model_config.get('controller', {}).get('global_dim', 32),
            gnn_hidden_dim=model_config.get('gnn', {}).get('hidden_dim', 64),
            gnn_output_dim=model_config.get('gnn', {}).get('output_dim', 256),
            rssm_hidden_dim=model_config.get('world_model', {}).get('hidden_dim', 128),
            rssm_latent_dim=model_config.get('world_model', {}).get('latent_dim', 64),
            controller_hidden_dim=model_config.get('controller', {}).get('hidden_dim', 128),
            top_k=model_config.get('controller', {}).get('top_k', 5),
            device=str(self.device)
        ).to(self.device)

        # 多GPU包装
        model = self.gpu_manager.wrap_model(model)

        # 2. 收集数据
        print("\n[DATA] Collecting training data...")
        data = self._collect_data()
        print(f"[OK] Collected {len(data['observations'])} samples")

        # 3. 训练
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)

        print(f"\n[TRAIN] Starting training...")
        best_loss = float('inf')
        num_samples = len(data['observations'])
        num_batches_per_epoch = (num_samples + self.batch_size - 1) // self.batch_size

        # 创建进度跟踪器
        progress = ProgressTracker(
            total_epochs=self.epochs,
            num_batches_per_epoch=num_batches_per_epoch,
            phase_name="PHASE 1"
        )

        try:
            for epoch in range(self.epochs):
                progress.start_epoch(epoch)
                model.train()
                epoch_loss = 0.0
                num_batches = 0

                indices = np.random.permutation(num_samples)

                for batch_start in range(0, num_samples, self.batch_size):
                    batch_end = min(batch_start + self.batch_size, num_samples)
                    batch_indices = indices[batch_start:batch_end]

                    if len(batch_indices) == 0:
                        continue

                    batch = self._prepare_batch(data, batch_indices)

                    # 前向传播（处理多GPU）
                    if self.gpu_manager.multi_gpu:
                        gnn_output = model.module.perception_layer(
                            node_features=batch['node_features'],
                            edge_index=batch['edge_index'],
                            edge_features=batch['edge_features'],
                            risk_features=batch['risk_features']
                        )

                        rssm_output = model.module.prediction_layer(
                            node_embeddings=gnn_output['node_embeddings']
                        )
                    else:
                        gnn_output = model.perception_layer(
                            node_features=batch['node_features'],
                            edge_index=batch['edge_index'],
                            edge_features=batch['edge_features'],
                            risk_features=batch['risk_features']
                        )

                        rssm_output = model.prediction_layer(
                            node_embeddings=gnm_output['node_embeddings']
                        )

                    # 计算损失
                    loss_dict = self._compute_loss(rssm_output, batch)
                    total_loss = (loss_dict['speed_mse'] * 1.0 +
                                loss_dict['position_mse'] * 0.5 +
                                loss_dict['conflict_bce'] * 2.0)

                    # 反向传播
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += total_loss.item()
                    num_batches += 1

                    # 更新batch进度
                    progress.update_batch(
                        batch_idx=num_batches - 1,
                        loss=total_loss.item(),
                        extra_metrics={'lr': f'{optimizer.param_groups[0]["lr"]:.1e}'}
                    )

                scheduler.step()
                avg_loss = epoch_loss / max(num_batches, 1)

                # 检查是否最佳
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss
                    checkpoint_path = os.path.join(self.checkpoint_dir, 'best.pth')

                    # 保存模型（自动处理DataParallel包装）
                    model_state = self.gpu_manager.get_model_state_dict(model)

                    torch.save({
                        'model_state_dict': model_state,
                        'config': self.config,
                        'epoch': epoch,
                        'loss': avg_loss
                    }, checkpoint_path)

                    print(f"  [SAVE] Best model saved (val_loss: {avg_loss:.4f})")

                # 结束epoch
                progress.end_epoch(
                    avg_loss=avg_loss,
                    is_best=is_best,
                    extra_metrics={'lr': optimizer.param_groups[0]["lr"]}
                )

                # 记录指标
                self.metrics.update(
                    epoch=epoch,
                    train_loss=avg_loss,
                    learning_rate=optimizer.param_groups[0]["lr"]
                )

        finally:
            progress.close()

        # 保存最终模型
        final_path = self.model_path
        model_state = self.gpu_manager.get_model_state_dict(model)

        torch.save({
            'model_state_dict': model_state,
            'config': self.config,
            'epoch': self.epochs
        }, final_path)

        print(f"\n[DONE] Phase 1 complete!")
        print(f"   Model: {final_path}")

        # 打印训练摘要
        self.metrics.print_summary()

        return final_path

    def _validate_cached_data(self, data: Dict[str, Any]) -> bool:
        """验证缓存数据的完整性和格式"""
        required_keys = ['observations', 'next_observations']

        for key in required_keys:
            if key not in data:
                print(f"[CACHE] Missing key: {key}")
                return False

        obs_len = len(data['observations'])
        next_obs_len = len(data['next_observations'])

        if obs_len == 0:
            print(f"[CACHE] Empty observations")
            return False

        if obs_len != next_obs_len:
            print(f"[CACHE] Length mismatch: observations={obs_len}, next_observations={next_obs_len}")
            return False

        try:
            first_obs = data['observations'][0]
            if not hasattr(first_obs, 'num_vehicles'):
                print(f"[CACHE] Invalid observation format")
                return False
        except Exception as e:
            print(f"[CACHE] Failed to validate first observation: {e}")
            return False

        return True

    def _collect_data(self) -> Dict[str, Any]:
        """收集训练数据（支持课程学习和并行收集）"""
        # 检查缓存
        if os.path.exists(self.cache_file):
            print(f"[CACHE] Loading from cache...")
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                if self._validate_cached_data(cached_data):
                    print(f"[CACHE] Cache loaded successfully")
                    return cached_data
                else:
                    print(f"[CACHE] Invalid cache format, recollecting...")
                    os.remove(self.cache_file)
            except Exception as e:
                print(f"[CACHE] Failed to load cache: {e}")
                print(f"[CACHE] Recollecting data...")

        print("[COLLECT] Collecting data...")

        observations = []
        next_observations = []

        # 如果启用课程学习，从不同级别收集
        if self.use_curriculum and self.curriculum:
            episodes_per_level = self.num_episodes // len(self.curriculum.levels)

            for level in self.curriculum.levels:
                print(f"\n[LEVEL {level.level}] {level.name}")
                print(f"   Episodes: {episodes_per_level}")
                print(f"   Config: vehicles={level.max_vehicles}, flow={level.inflow_rate}")

                env_config = self.config.get('environment', {}).copy()
                env_config['max_vehicles'] = level.max_vehicles
                env_config['inflow_rate'] = level.inflow_rate

                level_obs, level_next_obs = self._collect_level_data(env_config, episodes_per_level)
                observations.extend(level_obs)
                next_observations.extend(level_next_obs)
        else:
            env_config = self.config.get('environment', {})
            observations, next_observations = self._collect_level_data(env_config, self.num_episodes)

        data = {'observations': observations, 'next_observations': next_observations}

        # 保存缓存
        print(f"[CACHE] Saving...")
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"[WARNING] Failed to save cache: {e}")

        return data

    def _collect_level_data(self, env_config: Dict, num_episodes: int) -> Tuple[List, List]:
        """收集单个难度级别的数据（并行收集，使用spawn启动方法）"""
        from multiprocessing import get_context, cpu_count
        from functools import partial

        # 使用spawn方法启动进程，支持CUDA
        mp_context = get_context('spawn')

        num_workers = min(self.num_workers, cpu_count())
        episodes_per_worker = num_episodes // num_workers
        remainder = num_episodes % num_workers

        episode_counts = [episodes_per_worker + 1 if i < remainder else episodes_per_worker
                          for i in range(num_workers)]

        print(f"   Workers: {num_workers}, Episodes/worker: {episode_counts}")

        observations = []
        next_observations = []

        collect_func = partial(_collect_worker, env_config=env_config, max_steps=500)

        # 创建数据收集进度跟踪器
        collect_progress = DataCollectionProgress(num_workers)

        try:
            with mp_context.Pool(processes=num_workers) as pool:
                worker_results = []
                for worker_id, count in enumerate(episode_counts):
                    if count > 0:
                        result = pool.apply_async(collect_func, (worker_id, count))
                        worker_results.append((worker_id, result))

                for worker_id, result in worker_results:
                    try:
                        worker_obs, worker_next_obs = result.get(timeout=600)
                        observations.extend(worker_obs)
                        next_observations.extend(worker_next_obs)

                        # 更新进度
                        collect_progress.update_worker(worker_id, len(worker_obs))

                    except Exception as e:
                        print(f"\n   [WARNING] Worker {worker_id} failed: {e}")
        finally:
            collect_progress.close()

        print(f"   [OK] Collected {len(observations)} transitions")

        return observations, next_observations

    def _prepare_batch(self, data: Dict, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """准备训练batch"""
        device = self.device

        all_node_features = []
        all_edge_indices = []
        all_edge_features = []
        all_risk_features = []
        all_next_speeds = []
        all_next_positions = []
        all_conflict_labels = []

        node_offset = 0

        for idx in indices:
            if idx >= len(data['observations']):
                continue

            obs = data['observations'][idx]
            next_obs = data['next_observations'][idx]

            num_veh = obs.num_vehicles

            if num_veh == 0:
                continue

            # 节点特征
            node_features = _extract_node_features(obs)
            all_node_features.append(node_features)

            # 边特征
            edge_index, edge_attr = _build_edges(obs, self.config)
            edge_index = edge_index + node_offset
            all_edge_indices.append(edge_index)
            all_edge_features.append(edge_attr)
            node_offset += num_veh

            # 风险特征
            risk_features = _compute_risk_features(obs)
            all_risk_features.append(risk_features)

            # 目标值
            all_next_speeds.append(_extract_next_speed(next_obs, num_veh))
            all_next_positions.append(_extract_next_position(next_obs, num_veh))
            all_conflict_labels.append(_compute_conflict_label(obs, next_obs))

        if len(all_node_features) == 0:
            return self._get_empty_batch(device)

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

    def _compute_loss(self, rssm_output: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """计算损失"""
        speed_pred = rssm_output['speed_pred']
        pos_pred = rssm_output['position_pred']
        conflict_prob = rssm_output['conflict_prob']

        speed_target = batch['next_speed']
        pos_target = batch['next_position']
        conflict_target = batch['conflict_label']

        return {
            'speed_mse': F.mse_loss(speed_pred, speed_target),
            'position_mse': F.mse_loss(pos_pred, pos_target),
            'conflict_bce': F.binary_cross_entropy(conflict_prob.squeeze(-1), conflict_target.squeeze(-1))
        }

    def _get_empty_batch(self, device) -> Dict[str, torch.Tensor]:
        """返回空batch"""
        return {
            'node_features': torch.zeros((1, 9), device=device),
            'edge_index': torch.zeros((2, 1), dtype=torch.long, device=device),
            'edge_features': torch.zeros((1, 4), device=device),
            'risk_features': torch.zeros((1, 2), device=device),
            'next_speed': torch.zeros((1, 1), device=device),
            'next_position': torch.zeros((1, 2), device=device),
            'conflict_label': torch.zeros((1, 1), device=device),
        }


# =============================================================================
# 辅助函数
# =============================================================================

def _collect_worker(worker_id: int, num_episodes: int, env_config: Dict, max_steps: int = 500) -> Tuple[List, List]:
    """Worker函数：在单独进程中收集数据（使用spawn启动方法支持CUDA）"""
    observations = []
    next_observations = []

    for ep in range(num_episodes):
        env = CompetitionSumoEnv(env_config, use_gui=False)

        try:
            obs = env.reset()

            for step in range(max_steps):
                obs_trans = _create_transition(obs)
                observations.append(obs_trans)

                # 随机动作
                icv_ids = list(obs.get('icv_ids', set()))
                if len(icv_ids) > 0:
                    veh_id = icv_ids[0]
                    action = {veh_id: np.random.uniform(-1.0, 1.0, size=2)}
                else:
                    action = None

                obs, _, done, _ = env.step(action)

                next_trans = _create_transition(obs)
                next_observations.append(next_trans)

                if done:
                    break

        except Exception as e:
            print(f"[WARNING] Worker {worker_id}, Episode {ep}: {e}")
        finally:
            env.close()

    return observations, next_observations


def _create_transition(obs: Dict) -> TrafficTransition:
    """创建交通状态转移对象"""
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


def _extract_node_features(trans: TrafficTransition) -> torch.Tensor:
    """提取节点特征 [N, 9]"""
    num_veh = trans.num_vehicles

    features = np.zeros((num_veh, 9), dtype=np.float32)

    for i in range(num_veh):
        features[i, 0] = trans.s_coords[i] / 1000.0
        features[i, 1] = trans.d_coords[i] / 10.0
        features[i, 2] = trans.speeds[i] / 30.0
        features[i, 3] = 0.0
        features[i, 4] = trans.speeds[i] / 30.0
        features[i, 5] = trans.accels[i] / 3.0
        features[i, 6] = trans.lanes[i] / 10.0
        features[i, 7] = trans.angles[i] / 360.0
        features[i, 8] = 1.0 if i < len(trans.icv_ids) else 0.0

    return torch.from_numpy(features)


def _build_edges(trans: TrafficTransition, config: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """构建边（空间邻近）"""
    num_veh = trans.num_vehicles

    if num_veh <= 1:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 4), dtype=torch.float32)
        )

    graph_config = config.get('model', {}).get('graph', {})
    interaction_radius = graph_config.get('interaction_radius', 100.0)
    max_neighbors = graph_config.get('max_neighbors', 8)

    sources = []
    targets = []
    edge_features_list = []

    for i in range(num_veh):
        s_diff = trans.s_coords - trans.s_coords[i]
        d_diff = trans.d_coords - trans.d_coords[i]

        distances = np.sqrt(s_diff**2 + d_diff**2)

        mask = (distances < interaction_radius) & (distances > 0.1)
        neighbors = np.where(mask)[0]

        if len(neighbors) > max_neighbors:
            sorted_indices = np.argsort(distances[neighbors])[:max_neighbors]
            neighbors = neighbors[sorted_indices]

        for j in neighbors:
            sources.append(i)
            targets.append(j)

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


def _compute_risk_features(trans: TrafficTransition) -> torch.Tensor:
    """计算风险特征 [N, 2]"""
    num_veh = trans.num_vehicles

    if num_veh == 0:
        return torch.zeros((0, 2), dtype=torch.float32)

    speeds = np.maximum(trans.speeds, 0.1)

    ttc_inv = 1.0 / (speeds + 0.1)
    thw_inv = 1.0 / (speeds * 2.0 + 0.1)

    risk_features = np.stack([ttc_inv, thw_inv], axis=-1).astype(np.float32)

    return torch.from_numpy(risk_features)


def _extract_next_speed(trans: TrafficTransition, num_vehicles: int) -> torch.Tensor:
    """提取下一步的速度 [N, 1]"""
    target_num = num_vehicles
    actual_num = min(target_num, trans.num_vehicles)

    speeds = trans.speeds[:actual_num].copy()
    speeds = np.maximum(speeds, 0.0)
    speeds_normalized = (speeds / 30.0).astype(np.float32)

    if actual_num < target_num:
        padded = np.zeros((target_num, 1), dtype=np.float32)
        padded[:actual_num, 0] = speeds_normalized
        return torch.from_numpy(padded)

    return torch.from_numpy(speeds_normalized).unsqueeze(-1)


def _extract_next_position(trans: TrafficTransition, num_vehicles: int) -> torch.Tensor:
    """提取下一步的位置 [N, 2]"""
    target_num = num_vehicles
    actual_num = min(target_num, trans.num_vehicles)

    positions = np.zeros((target_num, 2), dtype=np.float32)
    positions[:actual_num, 0] = trans.s_coords[:actual_num] / 1000.0
    positions[:actual_num, 1] = trans.d_coords[:actual_num] / 10.0

    return torch.from_numpy(positions)


def _compute_conflict_label(obs_trans: TrafficTransition, next_trans: TrafficTransition) -> torch.Tensor:
    """计算冲突标签 [N, 1]"""
    num_veh = obs_trans.num_vehicles

    if num_veh == 0:
        return torch.zeros((0, 1), dtype=np.float32)

    n = min(num_veh, len(next_trans.accels), len(obs_trans.accels))

    accel_change = next_trans.accels[:n] - obs_trans.accels[:n]
    conflict_label = (accel_change < -2.0).astype(np.float32)

    if n < num_veh:
        padded = np.zeros((num_veh, 1), dtype=np.float32)
        padded[:n, 0] = conflict_label
        return torch.from_numpy(padded)

    return torch.from_numpy(conflict_label).unsqueeze(-1)


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    # 设置多进程启动方法为spawn，支持CUDA
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError as e:
        if "start_method has already been set" not in str(e):
            raise

    parser = argparse.ArgumentParser(
        description="Phase 1: 世界模型预训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础训练
  python train_phase1.py

  # 使用自定义配置
  python train_phase1.py --config configs/competition_preliminary.yaml

  # 指定设备
  python train_phase1.py --device cuda:0
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/competition_preliminary.yaml',
        help='配置文件路径 (默认: configs/competition_preliminary.yaml)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='计算设备 (默认: cuda:0)'
    )

    args = parser.parse_args()

    # 加载配置
    print(f"\n[CONFIG] Loading configuration from: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using device: {device}")

    # 设置随机种子
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"[SEED] Random seed: {seed}")

    # 创建训练器
    trainer = Phase1WorldModelTrainer(config)

    # 开始训练
    print("\n" + "="*80)
    print("[TRAIN] Starting Phase 1 Training...")
    print("="*80)

    start_time = time.time()
    final_model_path = trainer.train()
    training_time = time.time() - start_time

    print("\n" + "="*80)
    print("[SUCCESS] Phase 1 Training Completed!")
    print(f"  Training time: {training_time/60:.1f} minutes")
    print(f"  Final model: {final_model_path}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
