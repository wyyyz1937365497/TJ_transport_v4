"""数据模块 - PyTorch Lightning DataModule"""

import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch as PyGBatch

from ..models.graph import FastGraphBuilder
from ..utils.logging import get_logger

logger = get_logger()


# 全局变量：用于在multiprocessing中传递config
_collector_config = None


def _set_collector_config(config: Dict[str, Any]):
    """设置全局配置（用于multiprocessing）"""
    global _collector_config
    _collector_config = config


def _collect_episode_worker(args: Tuple[int, int, int, Optional[int]]) -> Tuple[Dict, Dict]:
    """
    单个episode收集函数（模块级别，可被pickle）

    Args:
        args: (episode_id, max_steps, timeout, port)

    Returns:
        (trajectories, stats)
    """
    from .sumo_env import SumoEnvironment

    global _collector_config
    config = _collector_config
    env_config = config.get("environment", config)  # 兼容两种格式

    episode_id, max_steps, timeout, port = args

    # 为每个worker分配不同端口
    if port is None:
        from ..utils.sumo_port_manager import find_free_port
        # 使用进程ID来分配端口范围，避免冲突
        base_port = 8813 + (os.getpid() % 100) + episode_id
        port = find_free_port(base_port)

    env = SumoEnvironment(env_config, use_gui=False, port=port)
    trajectories = {}
    stats = {
        "episode_id": episode_id,
        "total_steps": 0,
        "total_vehicles": 0,
        "success": False,
    }

    try:
        start_time = time.time()
        observation = env.reset()

        for step in range(max_steps):
            if time.time() - start_time > timeout:
                break

            # 记录数据
            current_time = step * config.get("environment", {}).get("step_length", 0.1)

            for veh_id, state in observation.vehicle_states.items():
                if veh_id not in trajectories:
                    trajectories[veh_id] = {
                        "id": veh_id,
                        "timestamps": [],
                        "positions": [],
                        "speeds": [],
                        "accelerations": [],
                        "lane_ids": [],
                        "lanes": [],
                    }

                trajectories[veh_id]["timestamps"].append(current_time)
                trajectories[veh_id]["positions"].append(state.position)
                trajectories[veh_id]["speeds"].append(state.speed)
                trajectories[veh_id]["accelerations"].append(state.acceleration)
                trajectories[veh_id]["lane_ids"].append(state.lane_id)
                trajectories[veh_id]["lanes"].append(state.lane_index)

            # 推进
            result = env.step()
            observation = result.observation
            done = result.done

            if done or len(observation.vehicle_states) == 0:
                break

        env.close()

        stats["total_steps"] = step + 1
        stats["total_vehicles"] = len(trajectories)
        stats["success"] = True

    except Exception as e:
        logger.error(f"Episode {episode_id} 失败: {e}")
        stats["error"] = str(e)

    return trajectories, stats


class TrajectoryDataset(Dataset):
    """轨迹数据集"""

    def __init__(
        self,
        trajectories: Dict[str, Dict],
        future_steps: int = 5,
        sequence_length: int = 10,
    ):
        self.trajectories = trajectories
        self.future_steps = future_steps
        self.sequence_length = sequence_length

        # 准备样本
        self.samples = self._prepare_samples()

    def _prepare_samples(self) -> List[Dict]:
        """准备训练样本"""
        samples = []

        for veh_id, traj in self.trajectories.items():
            timestamps = np.array(traj["timestamps"])
            positions = np.array(traj["positions"])
            speeds = np.array(traj["speeds"])
            accels = np.array(traj["accelerations"])
            lanes = np.array(traj["lanes"])

            # 构建序列
            for i in range(len(timestamps) - self.sequence_length - self.future_steps):
                # 当前序列
                current_seq = np.stack([
                    positions[i:i+self.sequence_length],
                    speeds[i:i+self.sequence_length],
                    accels[i:i+self.sequence_length],
                ], axis=-1)  # [T, 3]

                # 未来序列
                future_seq = np.stack([
                    positions[i+self.sequence_length:i+self.sequence_length+self.future_steps],
                    speeds[i+self.sequence_length:i+self.sequence_length+self.future_steps],
                    accels[i+self.sequence_length:i+self.sequence_length+self.future_steps],
                ], axis=-1)  # [T_future, 3]

                # 车道索引
                current_lanes = lanes[i:i+self.sequence_length]

                samples.append({
                    "current": current_seq,
                    "future": future_seq,
                    "current_lane_indices": current_lanes,
                    "vehicle_id": veh_id,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


class GraphCollator:
    """图数据整理器"""

    def __init__(self, device: torch.device):
        self.device = device
        # 在CPU上构建图
        self.graph_builder = FastGraphBuilder(device=torch.device("cpu"))

    def __call__(self, batch_list: List[Dict]) -> Dict[str, Any]:
        """整理一个batch"""
        batch_size = len(batch_list)

        # 提取数据
        all_current = np.stack([item["current"] for item in batch_list])  # [B, T, 3]
        all_future = np.stack([item["future"] for item in batch_list])  # [B, T_future, 3]
        all_lanes = np.stack([item["current_lane_indices"] for item in batch_list])  # [B, T]

        B, T, _ = all_current.shape

        # 提取最后一步状态
        last_position = all_current[:, -1, 0:1]  # [B, 1]
        last_position = np.concatenate([
            last_position,
            np.zeros((B, 1)),  # y = 0
        ], axis=1)  # [B, 2]

        last_speed = all_current[:, -1, 1:2]  # [B, 1]
        last_accel = all_current[:, -1, 2:3]  # [B, 1]
        last_lanes = all_lanes[:, -1]  # [B]

        # 构造2D速度和加速度
        velocities = np.concatenate([
            last_speed,
            np.zeros((B, 1)),
        ], axis=1)  # [B, 2]

        accelerations = np.concatenate([
            last_accel,
            np.zeros((B, 1)),
        ], axis=1)  # [B, 2]

        # 转换为CPU tensor
        positions = torch.from_numpy(last_position).float()
        velocities = torch.from_numpy(velocities).float()
        accelerations = torch.from_numpy(accelerations).float()
        lane_indices = torch.from_numpy(last_lanes).long()

        # 构建图
        graph_data = self.graph_builder.build_graph(
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            lane_indices=lane_indices,
        )

        # PyG batch
        graph_batch = PyGBatch.from_data_list([graph_data])

        return {
            "current": torch.from_numpy(all_current).float(),
            "future": torch.from_numpy(all_future).float(),
            "graph_data": graph_batch,
            "batch_size": batch_size,
        }


class TrafficDataModule:
    """交通数据模块"""

    def __init__(
        self,
        config: Dict[str, Any],
        data_dir: Path,
    ):
        self.config = config
        self.data_dir = Path(data_dir)

        # 训练配置
        phase1_cfg = config.get("training", {}).get("phase1", {})
        self.batch_size = phase1_cfg.get("batch_size", 256)
        self.num_workers = phase1_cfg.get("num_workers", 4)
        self.prefetch_factor = phase1_cfg.get("prefetch_factor", 2)

        # 设备
        device_str = config.get("device", "cuda")
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        # 查找数据文件
        data_files = list(self.data_dir.glob("*.pkl"))

        if not data_files:
            logger.warning(f"未找到数据文件: {self.data_dir}")
            self.train_dataset = None
            return

        # 加载最新的数据文件
        data_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        data_file = data_files[0]

        logger.info(f"加载数据文件: {data_file}")

        with open(data_file, "rb") as f:
            data = pickle.load(f)

        trajectories = data.get("trajectories", {})

        if not trajectories:
            logger.warning("数据文件中没有轨迹数据")
            self.train_dataset = None
            return

        logger.info(f"加载了 {len(trajectories)} 个车辆轨迹")

        # 创建数据集
        self.train_dataset = TrajectoryDataset(
            trajectories=trajectories,
            future_steps=self.config.get("model", {}).get("world_model", {}).get("future_steps", 5),
            sequence_length=10,
        )

        logger.info(f"数据集大小: {len(self.train_dataset)}")

    def train_dataloader(self) -> Optional[DataLoader]:
        """返回训练数据加载器"""
        if self.train_dataset is None:
            return None

        collator = GraphCollator(device=self.device)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=True,
            collate_fn=collator,
        )


def collect_data(
    config: Dict[str, Any],
    num_episodes: int = 5,
    max_steps: int = 3600,
    output_dir: str = "data",
) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
    """
    并行收集训练数据

    Args:
        config: 环境配置
        num_episodes: episode数量
        max_steps: 每个episode最大步数
        output_dir: 输出目录

    Returns:
        (trajectories, stats)
    """
    from multiprocessing import Pool, cpu_count

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_workers = config.get("environment", {}).get("num_parallel_workers", cpu_count())

    logger.info(f"开始并行数据收集 ({num_workers} workers)")
    logger.info(f"  - Episodes: {num_episodes}")
    logger.info(f"  - Max steps: {max_steps}")

    # 设置全局配置（用于multiprocessing）
    _set_collector_config(config)

    # 并行收集
    timeout = config.get("training", {}).get("phase1", {}).get("data_collection_timeout", 180)

    # 为每个episode分配不同端口，避免冲突
    tasks = []
    for i in range(num_episodes):
        # 每个episode使用不同的基础端口
        port = 8813 + i * 10  # 每个episode间隔10个端口
        tasks.append((i, max_steps, timeout, port))

    all_trajectories = {}
    all_stats = []

    with Pool(processes=min(num_workers, num_episodes)) as pool:
        results = pool.map(_collect_episode_worker, tasks)

        for trajectories, stats in results:
            all_trajectories.update(trajectories)
            all_stats.append(stats)

            if stats["success"]:
                logger.info(f"  Episode {stats['episode_id']}: "
                          f"{stats['total_vehicles']} 辆车, "
                          f"{stats['total_steps']} 步")

    # 保存数据
    timestamp = int(time.time())
    filepath = output_dir / f"parallel_data_{timestamp}.pkl"

    with open(filepath, "wb") as f:
        pickle.dump({
            "trajectories": all_trajectories,
            "stats": {
                "total_episodes": num_episodes,
                "successful": sum(1 for s in all_stats if s["success"]),
                "total_steps": sum(s["total_steps"] for s in all_stats),
                "total_vehicles": len(all_trajectories),
            },
        }, f)

    logger.info(f"数据已保存: {filepath}")

    total_stats = {
        "total_episodes": num_episodes,
        "successful_episodes": sum(1 for s in all_stats if s["success"]),
        "total_steps": sum(s["total_steps"] for s in all_stats),
        "total_vehicles": len(all_trajectories),
    }

    return all_trajectories, total_stats
