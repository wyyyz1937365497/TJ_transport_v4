"""
Phase 1 世界模型训练器 - 正确版本
"""

import os
import pickle
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, Tuple, List
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from dataclasses import dataclass

# 导入正确的模型
try:
    from src.models.v4_architecture import IdealTrafficControllerV4
except ImportError:
    # Fallback
    from src.models.ideal_policy_v4 import IdealTrafficPolicyV4 as IdealTrafficControllerV4
    
from src.env.competition_env import CompetitionSumoEnv


@dataclass
class TrafficTransition:
    """交通状态转移数据"""
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


def _collect_worker(worker_id: int, num_episodes: int, env_config: Dict, max_steps: int = 500) -> Tuple[List, List]:
    """Worker函数"""
    # ⭐ 关键：必须在任何PyTorch导入之前设置
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # 强制使用CPU
    import torch
    torch.cuda.is_available = lambda: False

    observations = []
    next_observations = []

    for ep in range(num_episodes):
        env = CompetitionSumoEnv(env_config, use_gui=False)
        try:
            obs = env.reset()

            for step in range(max_steps):
                # ⭐ 只存储成对数据
                obs_trans = _create_transition(obs)

                icv_ids = list(obs.get('icv_ids', set()))
                if len(icv_ids) > 0:
                    action = {icv_ids[0]: np.random.uniform(-1.0, 1.0, size=2)}
                else:
                    action = None

                next_obs_dict, reward, done, info = env.step(action)
                next_trans = _create_transition(next_obs_dict)

                # ⭐ 只在不done时存储，确保成对
                if not done:
                    observations.append(obs_trans)
                    next_observations.append(next_trans)

                if done:
                    break

                obs = next_obs_dict

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
    """提取节点特征"""
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
    """构建边"""
    num_veh = trans.num_vehicles
    if num_veh <= 1:
        return (torch.empty((2, 0), dtype=torch.long),
                torch.empty((0, 4), dtype=torch.float32))
    
    graph_config = config.get('model', {}).get('graph', {})
    interaction_radius = graph_config.get('interaction_radius', 100.0)
    max_neighbors = graph_config.get('max_neighbors', 8)
    
    sources, targets, edge_features_list = [], [], []
    
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
            edge_feat = np.array([
                s_diff[j] / 100.0,
                d_diff[j] / 10.0,
                distances[j] / 100.0,
                1.0 if trans.vehicle_ids[j] in trans.icv_ids else 0.0
            ], dtype=np.float32)
            edge_features_list.append(edge_feat)
    
    if len(sources) == 0:
        return (torch.empty((2, 0), dtype=torch.long),
                torch.empty((0, 4), dtype=torch.float32))
    
    edge_index = torch.stack([
        torch.tensor(sources, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long)
    ])
    edge_features = torch.stack([torch.from_numpy(f) for f in edge_features_list])
    
    return edge_index, edge_features


def _compute_risk_features(trans: TrafficTransition) -> torch.Tensor:
    """计算风险特征"""
    num_veh = trans.num_vehicles
    if num_veh == 0:
        return torch.zeros((0, 2), dtype=torch.float32)
    
    risk1 = np.var(trans.speeds) / 100.0
    risk2 = np.var(trans.accels) / 10.0
    
    risk_features = np.zeros((num_veh, 2), dtype=np.float32)
    risk_features[:, 0] = risk1
    risk_features[:, 1] = risk2
    
    return torch.from_numpy(risk_features)


def _extract_next_speed(next_obs: TrafficTransition, num_veh: int) -> torch.Tensor:
    """提取下一个时刻的速度"""
    if num_veh == 0:
        return torch.zeros((0, 1), dtype=torch.float32)
    speeds = next_obs.speeds[:num_veh] / 30.0
    return torch.from_numpy(speeds).unsqueeze(-1).float()


def _extract_next_position(next_obs: TrafficTransition, num_veh: int) -> torch.Tensor:
    """提取下一个时刻的位置"""
    if num_veh == 0:
        return torch.zeros((0, 2), dtype=torch.float32)
    s = next_obs.s_coords[:num_veh] / 1000.0
    d = next_obs.d_coords[:num_veh] / 10.0
    positions = np.stack([s, d], axis=1)
    return torch.from_numpy(positions).float()


def _compute_conflict_label(obs: TrafficTransition, next_obs: TrafficTransition) -> torch.Tensor:
    """计算冲突标签"""
    # ⭐ 使用min确保车辆数一致
    num_veh = min(obs.num_vehicles, next_obs.num_vehicles)

    if num_veh == 0:
        return torch.zeros((0, 1), dtype=torch.float32)

    labels = np.zeros(num_veh, dtype=np.float32)
    for i in range(num_veh):
        min_dist = float('inf')
        for j in range(num_veh):
            if i == j:
                continue
            s_diff = obs.s_coords[i] - obs.s_coords[j]
            d_diff = obs.d_coords[i] - obs.d_coords[j]
            dist = np.sqrt(s_diff**2 + d_diff**2)
            if dist < min_dist:
                min_dist = dist
        if min_dist < 5.0:
            labels[i] = 1.0

    return torch.from_numpy(labels).unsqueeze(-1)


class WorldModelTrainer:
    """世界模型训练器"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        
        phase1_config = config.get('training', {}).get('phase1', {})
        self.num_epochs = phase1_config.get('epochs', 50)
        self.batch_size = phase1_config.get('batch_size', 256)
        self.learning_rate = float(phase1_config.get('learning_rate', 1e-4))
        self.num_episodes = phase1_config.get('num_episodes', 100)
        self.num_workers = phase1_config.get('num_parallel_workers', 4)
        
        checkpoint_dir = Path(config.get('paths', {}).get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir = checkpoint_dir / 'preliminary' / 'phase1'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = self.checkpoint_dir / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / 'data.pkl'
        
        self.env_config = config.get('environment', {}).copy()
        self.env_config.update({
            'max_vehicles': 10,
            'inflow_rate': 800,
            'icv_ratio': 0.3,
            'disturbance_level': 0.0,
        })
        
        print(f"[OK] WorldModelTrainer initialized")
        print(f"  - Device: {device}")
        print(f"  - Epochs: {self.num_epochs}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Learning rate: {self.learning_rate}")
        print(f"  - Parallel workers: {self.num_workers}")
    
    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 80)
        print("[PHASE 1] World Model Training")
        print("=" * 80)
        
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
        
        # 2. 收集数据
        print("\n[DATA] Collecting training data...")
        data = self._collect_data()
        print(f"[OK] Collected {len(data['observations'])} samples")

        # ⭐ 验证数据完整性
        if len(data['observations']) != len(data['next_observations']):
            print(f"[ERROR] CRITICAL: Data length mismatch!")
            print(f"  observations: {len(data['observations'])}")
            print(f"  next_observations: {len(data['next_observations'])}")
            raise ValueError("Observations and next_observations must have the same length")

        print(f"[VALIDATE] Data integrity check passed")

        # 3. 训练
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)
        
        print(f"\n[TRAIN] Starting training...")
        best_loss = float('inf')
        num_samples = len(data['observations'])
        
        for epoch in range(self.num_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            indices = np.random.permutation(num_samples)
            
            pbar = tqdm(range(0, num_samples, self.batch_size), desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for batch_start in pbar:
                batch_end = min(batch_start + self.batch_size, num_samples)
                batch_indices = indices[batch_start:batch_end]
                
                if len(batch_indices) == 0:
                    continue
                
                batch = self._prepare_batch(data, batch_indices)
                
                if batch['node_features'].size(0) == 0:
                    continue
                
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
                
                pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
            
            scheduler.step()
            avg_loss = epoch_loss / max(num_batches, 1)
            
            print(f"\n[EPOCH {epoch+1}/{self.num_epochs}]")
            print(f"  Train Loss: {avg_loss:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = self.checkpoint_dir / 'world_model_best.pth'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': self.config,
                    'epoch': epoch,
                    'loss': avg_loss
                }, checkpoint_path)
                print(f"  [SAVE] Best model saved")
        
        # 保存最终模型
        final_path = self.checkpoint_dir / 'world_model_final.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'epoch': self.num_epochs
        }, final_path)
        
        print(f"\n[DONE] Phase 1 complete!")
        print(f"   Model: {final_path}")
        print(f"   Best loss: {best_loss:.4f}")
        
        return str(final_path)
    
    def _collect_data(self) -> Dict[str, Any]:
        """收集训练数据"""
        if self.cache_file.exists():
            print(f"[CACHE] Loading from cache...")
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                if len(data.get('observations', [])) > 0:
                    print(f"[CACHE] Cache loaded successfully")
                    return data
                else:
                    print(f"[CACHE] Invalid cache, recollecting...")
                    self.cache_file.unlink()
            except Exception as e:
                print(f"[CACHE] Failed to load cache: {e}")

        print("[COLLECT] Collecting data...")

        num_workers = min(self.num_workers, cpu_count())
        episodes_per_worker = self.num_episodes // num_workers
        remainder = self.num_episodes % num_workers

        episode_counts = [
            episodes_per_worker + 1 if i < remainder else episodes_per_worker
            for i in range(num_workers)
        ]

        print(f"   Workers: {num_workers}, Episodes/worker: {episode_counts}")

        observations = []
        next_observations = []

        collect_func = partial(_collect_worker, env_config=self.env_config, max_steps=500)

        try:
            # ⭐ 使用spawn方法避免CUDA问题
            import multiprocessing as mp
            mp.set_start_method('spawn', force=True)

            with Pool(processes=num_workers) as pool:
                worker_results = []
                for worker_id, count in enumerate(episode_counts):
                    if count > 0:
                        result = pool.apply_async(collect_func, (worker_id, count))
                        worker_results.append(result)

                for result in worker_results:
                    worker_obs, worker_next_obs = result.get(timeout=600)
                    observations.extend(worker_obs)
                    next_observations.extend(worker_next_obs)

        except RuntimeError as e:
            if 'set_start_method' in str(e):
                # 已经设置过，使用默认方法
                with Pool(processes=num_workers) as pool:
                    worker_results = []
                    for worker_id, count in enumerate(episode_counts):
                        if count > 0:
                            result = pool.apply_async(collect_func, (worker_id, count))
                            worker_results.append(result)

                    for result in worker_results:
                        worker_obs, worker_next_obs = result.get(timeout=600)
                        observations.extend(worker_obs)
                        next_observations.extend(worker_next_obs)
            else:
                raise e

        except Exception as e:
            print(f"[ERROR] Collection failed: {e}")

        print(f"   [OK] Collected {len(observations)} transitions")

        # ⭐ 验证数据长度一致性
        if len(observations) != len(next_observations):
            print(f"[ERROR] Data length mismatch: obs={len(observations)}, next_obs={len(next_observations)}")
            min_len = min(len(observations), len(next_observations))
            observations = observations[:min_len]
            next_observations = next_observations[:min_len]
            print(f"[FIX] Truncated to {min_len} samples")

        data = {'observations': observations, 'next_observations': next_observations}

        # 保存缓存
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"[WARNING] Failed to save cache: {e}")

        return data
    
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
            # ⭐ 双重检查索引有效性
            if idx >= len(data['observations']) or idx >= len(data['next_observations']):
                continue

            obs = data['observations'][idx]
            next_obs = data['next_observations'][idx]

            # ⭐ 关键修复：使用min确保车辆数一致
            num_veh = min(obs.num_vehicles, next_obs.num_vehicles)

            if num_veh == 0:
                continue

            node_features = _extract_node_features(obs)
            all_node_features.append(node_features[:num_veh])

            edge_index, edge_attr = _build_edges(obs, self.config)
            # 只保留有效车辆的边
            valid_mask = (edge_index[0] < num_veh) & (edge_index[1] < num_veh)
            edge_index = edge_index[:, valid_mask] + node_offset
            edge_attr = edge_attr[valid_mask]
            all_edge_indices.append(edge_index)
            all_edge_features.append(edge_attr)
            node_offset += num_veh

            risk_features = _compute_risk_features(obs)
            all_risk_features.append(risk_features[:num_veh])
            
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
