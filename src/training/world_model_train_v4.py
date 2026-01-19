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


class WorldModelDataset(torch.utils.data.Dataset):
    """
    世界模型训练数据集

    收集真实交通数据用于监督学习预训练
    """

    def __init__(
        self,
        env_config: Dict[str, Any],
        num_episodes: int = 100,
        steps_per_episode: int = 1000,
        seed: int = 42
    ):
        """
        Args:
            env_config: 环境配置
            num_episodes: 收集的episode数量
            steps_per_episode: 每个episode的步数
            seed: 随机种子
        """
        self.env_config = env_config
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode
        self.seed = seed

        # 数据存储
        self.observations = []
        self.next_observations = []

        print(f"[DATASET] Collecting {num_episodes} episodes...")
        self._collect_data()
        print(f"[OK] Dataset collected: {len(self.observations)} samples")

    def _collect_data(self):
        """使用环境收集数据"""
        env = CompetitionSumoEnv(config=self.env_config)

        for episode in tqdm(range(self.num_episodes), desc="Collecting data"):
            # 设置随机种子（如果环境支持）
            if hasattr(env, 'seed'):
                env.seed(self.seed + episode)

            obs = env.reset()

            for step in range(self.steps_per_episode):
                # 生成随机动作字典 {vehicle_id: [acceleration, lane_change]}
                actions = {}
                if 'vehicles' in obs and 'id' in obs['vehicles']:
                    vehicle_ids = obs['vehicles']['id']
                    for i, veh_id in enumerate(vehicle_ids):
                        # 随机加速度 [-3, 2] m/s²
                        accel = np.random.uniform(-3.0, 2.0)
                        # 随机换道概率 [0, 1]
                        lane_change = np.random.uniform(0, 1)
                        actions[str(veh_id)] = np.array([accel, lane_change])

                # 如果没有车辆，跳过这个step
                if not actions:
                    continue

                next_obs, reward, done, info = env.step(actions)

                # 存储数据
                self.observations.append(obs)
                self.next_observations.append(next_obs)

                if done:
                    break

                obs = next_obs

        env.close()

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.observations[idx]).float(),
            torch.from_numpy(self.next_observations[idx]).float()
        )


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
