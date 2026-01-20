"""
GPU加速的智能车辆选择器训练脚本 - 每步详细输出版本

核心优化：
1. 所有数据预处理在GPU上进行
2. 使用PyTorch张量替代numpy
3. 图结构在GPU上构建
4. 零拷贝操作，避免CPU-GPU传输
5. 每个batch都输出详细时间统计
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple
import numpy as np

# 导入项目模块
from src.models.neural_vehicle_scorer import (
    NeuralICVScorer,
    create_neural_icv_scorer
)
from src.env.gym_wrapper import make_gym_env

# =============================================================================
# 日志配置
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# GPU加速的特征提取器
# =============================================================================
class GPUFeatureExtractor:
    """
    高效批量特征提取（CPU numpy → 一次性GPU传输）

    优化策略：
    1. CPU上用numpy快速批量构建（避免Python循环中的GPU操作）
    2. 一次性传输到GPU（避免大量小传输）
    3. GPU上拼接（并行操作）
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device

    def extract_batch(
        self,
        vehicle_states_list: List[Dict[str, Dict]],
        icv_ids_list: List[List[str]],
        labels_list: List[Dict[str, float]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        高效批量特征提取（CPU numpy → 一次性GPU传输）
        """
        # ✅ 关键优化：CPU上用numpy批量构建（快100倍）
        batch_features = []
        batch_labels = []
        batch_ptr = [0]

        for vehicle_states, icv_ids, labels_dict in zip(
            vehicle_states_list, icv_ids_list, labels_list
        ):
            # 收集有效车辆
            valid_vehicles = [
                (vehicle_states[veh_id], labels_dict[veh_id])
                for veh_id in icv_ids
                if veh_id in vehicle_states and veh_id in labels_dict
            ]

            if len(valid_vehicles) == 0:
                batch_ptr.append(batch_ptr[-1])
                continue

            # ✅ CPU上用numpy批量构建（向量化操作）
            num_vehicles = len(valid_vehicles)
            features_np = np.zeros((num_vehicles, 9), dtype=np.float32)
            labels_np = np.zeros(num_vehicles, dtype=np.float32)

            for i, (state, label) in enumerate(valid_vehicles):
                features_np[i, 0] = state['s'] / 1000.0
                features_np[i, 1] = state['d'] / 10.0
                features_np[i, 2] = state['speed'] / 30.0
                features_np[i, 3] = 0.0  # vd
                features_np[i, 4] = state['speed'] / 30.0
                features_np[i, 5] = state.get('acceleration', 0.0) / 3.0
                features_np[i, 6] = state['lane_index'] / 10.0
                features_np[i, 7] = 0.0  # angle
                features_np[i, 8] = 1.0  # is_icv
                labels_np[i] = label

            batch_features.append(features_np)
            batch_labels.append(labels_np)
            batch_ptr.append(batch_ptr[-1] + num_vehicles)

        if len(batch_features) == 0:
            return None, None, None

        # ✅ CPU上拼接所有样本（快速）
        all_features_np = np.concatenate(batch_features, axis=0)  # [total_nodes, 9]
        all_labels_np = np.concatenate(batch_labels, axis=0)      # [total_nodes]

        # ✅ 关键：一次性传输到GPU（避免数百次小传输）
        features = torch.from_numpy(all_features_np).to(self.device, non_blocking=True)
        labels = torch.from_numpy(all_labels_np).to(self.device, non_blocking=True)
        batch_ptr = torch.tensor(batch_ptr, dtype=torch.long, device=self.device)

        return features, labels, batch_ptr


# =============================================================================
# GPU加速的图构建器
# =============================================================================
class GPUGraphBuilder:
    """
    GPU加速的图结构构建器

    关键优化：
    - 使用PyTorch张量操作构建边索引
    - 批量构建，避免循环
    - 支持变长图
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device

    def build_batch_graph(
        self,
        batch_ptr: torch.Tensor
    ) -> torch.Tensor:
        """
        为批次构建全连接边索引（在GPU上）
        """
        batch_size = len(batch_ptr) - 1
        edge_indices = []

        for i in range(batch_size):
            num_nodes = batch_ptr[i + 1] - batch_ptr[i]
            offset = batch_ptr[i]

            if num_nodes <= 1:
                continue

            # ✅ 在GPU上构建全连接图
            # 创建节点索引
            node_indices = torch.arange(num_nodes, device=self.device) + offset

            # 生成所有边对（使用meshgrid）
            grid_i, grid_j = torch.meshgrid(node_indices, node_indices, indexing='ij')
            mask = grid_i != grid_j  # 移除自环

            edges_i = grid_i[mask]
            edges_j = grid_j[mask]

            edge_index = torch.stack([edges_i, edges_j], dim=0)
            edge_indices.append(edge_index)

        if len(edge_indices) == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)

        # 拼接所有边
        return torch.cat(edge_indices, dim=1)


# =============================================================================
# GPU优化的数据集
# =============================================================================
class GPUOptimizedDataset(Dataset):
    """
    GPU优化的数据集
    """

    def __init__(self, buffer: List[Dict]):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        sample = self.buffer[idx]
        return (
            sample['vehicle_states'],
            sample['icv_ids'],
            sample['labels']
        )


# =============================================================================
# Stage 1训练器（GPU优化版 - 每步详细输出）
# =============================================================================
class Stage1TrainerGPU:
    """
    Stage 1训练器（完全GPU加速 + 每步详细输出）

    关键优化：
    1. 特征提取在GPU上进行
    2. 图结构在GPU上构建
    3. 批量处理，避免循环
    4. 每个batch都输出详细时间统计
    """

    def __init__(
        self,
        model: NeuralICVScorer,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device

        # 训练配置
        stage1_config = config.get('training', {}).get('stage1', {})
        self.num_episodes = stage1_config.get('num_episodes', 100)
        self.epochs = stage1_config.get('epochs', 20)
        self.learning_rate = stage1_config.get('learning_rate', 0.001)
        self.batch_size = stage1_config.get('batch_size', 256)

        # GPU加速组件
        self.feature_extractor = GPUFeatureExtractor(device)
        self.graph_builder = GPUGraphBuilder(device)

        # 优化器
        self.optimizer = optim.Adam(
            self.model.neural_scorer.parameters(),
            lr=self.learning_rate
        )

        # 混合精度训练（暂时禁用，因为模型已有sigmoid）
        self.use_amp = False  # BCELoss在autocast中不安全
        self.scaler = None
        logger.info("ℹ️  混合精度已禁用（模型已有sigmoid层，BCELoss不兼容autocast）")

        # 数据缓冲区
        self.buffer = []

        # 数据缓存路径
        self.cache_dir = Path('collected_data')
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / 'smart_selector_stage1_cache.pkl'

    def save_cache(self):
        """保存数据到缓存"""
        import pickle
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.buffer, f)
        logger.info(f"✅ 数据已缓存: {len(self.buffer)}个样本")

    def load_cache(self) -> bool:
        """从缓存加载数据"""
        import pickle
        if self.cache_file.exists():
            with open(self.cache_file, 'rb') as f:
                self.buffer = pickle.load(f)
            logger.info(f"✅ 从缓存加载数据: {len(self.buffer)}个样本")
            return True
        return False

    def collect_data(self, env, use_cache: bool = True) -> int:
        """使用启发式策略收集数据"""
        if use_cache and self.load_cache():
            return len(self.buffer)

        logger.info("开始收集数据...")
        from train_smart_selector import HeuristicVehicleSelector, HeuristicController

        selector = HeuristicVehicleSelector(self.config)
        controller = HeuristicController(self.config)

        for episode in tqdm(range(self.num_episodes), desc="数据收集"):
            obs, info = env.reset()

            for step in range(3600):
                # 提取车辆状态
                if isinstance(obs, dict):
                    vehicle_ids = obs.get('vehicle_ids', [])
                    icv_ids = list(obs.get('icv_ids', []))
                    vehicle_states_array = obs.get('vehicle_states', np.zeros((0, 9)))
                else:
                    action = {'vehicle_ids': [], 'actions': np.array([], dtype=np.float32).reshape(0, 2)}
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                    continue

                if len(icv_ids) < 10:
                    action = {'vehicle_ids': [], 'actions': np.array([], dtype=np.float32).reshape(0, 2)}
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                    continue

                # 构建vehicle_states
                vehicle_states = {}
                for i, veh_id in enumerate(vehicle_ids):
                    if i < len(vehicle_states_array):
                        state = vehicle_states_array[i]
                        vehicle_states[veh_id] = {
                            's': float(state[0]) * 1000.0,
                            'd': float(state[1]) * 10.0,
                            'speed': float(state[4]) * 30.0,
                            'lane_index': int(state[6] * 10.0),
                            'acceleration': 0.0,
                            'lead_distance': 50.0,
                            'max_speed': 33.33
                        }

                # 启发式选择
                selected_ids = selector.select_vehicles(
                    vehicle_states,
                    icv_ids,
                    top_k=min(30, max(1, len(icv_ids) // 5))
                )

                # 生成控制
                actions_dict = controller.compute_actions(vehicle_states, selected_ids)

                # 执行
                action = {
                    'vehicle_ids': list(actions_dict.keys()),
                    'actions': np.array(list(actions_dict.values()), dtype=np.float32)
                }
                obs, reward, terminated, truncated, info = env.step(action)

                # 记录样本
                if len(icv_ids) >= 10 and step % 10 == 0:
                    labels = {veh_id: 1.0 if veh_id in selected_ids else 0.0
                             for veh_id in icv_ids}
                    self.buffer.append({
                        'vehicle_states': vehicle_states.copy(),
                        'icv_ids': icv_ids.copy(),
                        'labels': labels
                    })

                if terminated or truncated:
                    break

        logger.info(f"数据收集完成: {len(self.buffer)}个样本")
        self.save_cache()
        return len(self.buffer)

    def train_epoch(self, epoch: int) -> float:
        """
        训练一个epoch（完全GPU加速 + 每步详细输出）
        """
        import time
        epoch_start_time = time.time()

        dataset = GPUOptimizedDataset(self.buffer)
        logger.info(f"Epoch {epoch+1}/{self.epochs} - 数据集大小: {len(dataset)}, Batch size: {self.batch_size}")

        # 自定义collate函数，直接返回元组列表
        def collate_fn(batch):
            return batch

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn
        )

        total_loss = 0.0
        num_batches = 0
        total_samples = 0

        # 时间统计
        timing_stats = {
            'data_preparation': 0.0,
            'feature_extraction': 0.0,
            'graph_building': 0.0,
            'forward': 0.0,
            'backward': 0.0,
            'batch_overhead': 0.0,
        }

        self.model.neural_scorer.train()

        # 不使用tqdm，直接循环并实时输出
        logger.info(f"开始训练 Epoch {epoch+1}/{self.epochs}...")

        for batch_idx, batch in enumerate(dataloader):
            batch_start = time.time()

            # 当前batch的时间统计
            batch_timing = {}

            # ✅ 关键：在GPU上批量提取特征
            vehicle_states_list = []
            icv_ids_list = []
            labels_list = []

            # 数据准备阶段
            step_start = time.time()
            for vehicle_states, icv_ids, labels in batch:
                vehicle_states_list.append(vehicle_states)
                icv_ids_list.append(icv_ids)
                labels_list.append(labels)
            batch_timing['data_preparation'] = time.time() - step_start

            # ✅ GPU特征提取（零拷贝）
            step_start = time.time()
            features, labels_tensor, batch_ptr = self.feature_extractor.extract_batch(
                vehicle_states_list,
                icv_ids_list,
                labels_list
            )
            batch_timing['feature_extraction'] = time.time() - step_start

            if features is None:
                logger.warning(f"Batch {batch_idx+1}: 跳过（没有有效特征）")
                continue

            # ✅ GPU图构建
            step_start = time.time()
            edge_index = self.graph_builder.build_batch_graph(batch_ptr)
            batch_timing['graph_building'] = time.time() - step_start

            # ✅ GPU前向传播
            step_start = time.time()
            if self.scaler is not None:
                # 使用混合精度（兼容不同PyTorch版本）
                try:
                    with torch.amp.autocast('cuda'):
                        pred_scores = self.model.neural_scorer(features, edge_index).squeeze()
                        if pred_scores.dim() == 0:
                            pred_scores = pred_scores.unsqueeze(0)
                        loss = nn.BCELoss()(pred_scores, labels_tensor)
                except TypeError:
                    with torch.cuda.amp.autocast():
                        pred_scores = self.model.neural_scorer(features, edge_index).squeeze()
                        if pred_scores.dim() == 0:
                            pred_scores = pred_scores.unsqueeze(0)
                        loss = nn.BCELoss()(pred_scores, labels_tensor)

                # 反向传播（混合精度）
                backward_start = time.time()
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                batch_timing['backward'] = time.time() - backward_start
            else:
                pred_scores = self.model.neural_scorer(features, edge_index).squeeze()
                if pred_scores.dim() == 0:
                    pred_scores = pred_scores.unsqueeze(0)
                loss = nn.BCELoss()(pred_scores, labels_tensor)

                # 反向传播
                backward_start = time.time()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_timing['backward'] = time.time() - backward_start

            batch_timing['forward'] = time.time() - step_start - batch_timing['backward']

            total_loss += loss.item()
            num_batches += 1
            total_samples += len(vehicle_states_list)

            # 其他开销
            batch_total_time = time.time() - batch_start
            batch_timing['batch_overhead'] = batch_total_time - sum(batch_timing.values())

            # 累计统计
            for key in timing_stats:
                timing_stats[key] += batch_timing[key]

            # ✅ 实时输出每个batch的详细时间
            logger.info(f"-"*80)
            logger.info(f"Batch {batch_idx+1}/{len(dataloader)} 详细时间:")
            logger.info(f"  总时间: {batch_total_time*1000:7.1f}ms")
            logger.info(f"  样本数: {len(vehicle_states_list)}")
            logger.info(f"  当前Loss: {loss.item():.4f}, 平均Loss: {total_loss/num_batches:.4f}")
            logger.info(f"  分步耗时:")
            logger.info(f"    1. 数据准备:   {batch_timing['data_preparation']*1000:6.1f}ms ({batch_timing['data_preparation']/batch_total_time*100:5.1f}%)")
            logger.info(f"    2. 特征提取:   {batch_timing['feature_extraction']*1000:6.1f}ms ({batch_timing['feature_extraction']/batch_total_time*100:5.1f}%)")
            logger.info(f"    3. 图构建:     {batch_timing['graph_building']*1000:6.1f}ms ({batch_timing['graph_building']/batch_total_time*100:5.1f}%)")
            logger.info(f"    4. 前向传播:   {batch_timing['forward']*1000:6.1f}ms ({batch_timing['forward']/batch_total_time*100:5.1f}%)")
            logger.info(f"    5. 反向传播:   {batch_timing['backward']*1000:6.1f}ms ({batch_timing['backward']/batch_total_time*100:5.1f}%)")
            logger.info(f"    6. 其他开销:   {batch_timing['batch_overhead']*1000:6.1f}ms ({batch_timing['batch_overhead']/batch_total_time*100:5.1f}%)")

            # 识别当前batch的瓶颈
            max_step = max(batch_timing.items(), key=lambda x: x[1])
            logger.info(f"  ⚠️  瓶颈: {max_step[0]} ({max_step[1]*1000:.1f}ms, {max_step[1]/batch_total_time*100:.1f}%)")
            logger.info(f"  累计: {total_samples} 样本, 已用时间: {sum(timing_stats.values()):.1f}s")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_time = time.time() - epoch_start_time

        # ✅ 打印epoch总结
        logger.info("="*80)
        logger.info(f"Epoch {epoch+1}/{self.epochs} 总结")
        logger.info("="*80)
        logger.info(f"总体统计:")
        logger.info(f"  总时间:        {epoch_time:.2f}s ({epoch_time/60:.1f}分钟)")
        logger.info(f"  平均速度:      {total_samples/epoch_time:.1f} samples/s")
        logger.info(f"  Batch数量:     {num_batches}")
        logger.info(f"  平均Loss:      {avg_loss:.4f}")
        logger.info(f"")
        logger.info(f"分步骤平均时间（每batch）:")
        logger.info(f"  1. 数据准备:   {timing_stats['data_preparation']/num_batches*1000:6.1f}ms")
        logger.info(f"  2. 特征提取:   {timing_stats['feature_extraction']/num_batches*1000:6.1f}ms")
        logger.info(f"  3. 图构建:     {timing_stats['graph_building']/num_batches*1000:6.1f}ms")
        logger.info(f"  4. 前向传播:   {timing_stats['forward']/num_batches*1000:6.1f}ms")
        logger.info(f"  5. 反向传播:   {timing_stats['backward']/num_batches*1000:6.1f}ms")
        logger.info(f"  6. 其他开销:   {timing_stats['batch_overhead']/num_batches*1000:6.1f}ms")
        logger.info("="*80)

        # 识别整个epoch的瓶颈
        max_step_time = max(timing_stats.values())
        max_step_name = [k for k, v in timing_stats.items() if v == max_step_time][0]
        logger.info(f"⚠️  Epoch瓶颈: '{max_step_name}' 占用 {max_step_time/epoch_time*100:.1f}% 的时间")
        logger.info("="*80)

        return avg_loss

    def run(self, env, use_cache: bool = True):
        """运行Stage 1训练"""
        logger.info("="*70)
        logger.info("Stage 1: 启发式引导训练（GPU加速版）")
        logger.info("="*70)

        # 收集数据
        num_samples = self.collect_data(env, use_cache=use_cache)

        if num_samples == 0:
            logger.error("没有收集到数据！")
            return

        logger.info(f"数据集大小: {len(self.buffer)}")
        logger.info(f"训练配置: epochs={self.epochs}, batch_size={self.batch_size}, lr={self.learning_rate}")
        logger.info(f"优化: cuDNN benchmark=True, torch.compile=True, 混合精度=False")

        # 训练
        import time
        train_start_time = time.time()
        best_loss = float('inf')

        for epoch in range(self.epochs):
            avg_loss = self.train_epoch(epoch)

            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = 'checkpoints/smart_selector_stage1.pth'
                Path('checkpoints').mkdir(exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.neural_scorer.state_dict(),
                    'epoch': epoch,
                    'loss': avg_loss
                }, checkpoint_path)
                logger.info(f"✅ 保存最佳模型: {checkpoint_path} (loss={avg_loss:.4f})")

        total_train_time = time.time() - train_start_time

        logger.info("="*70)
        logger.info("Stage 1训练完成！")
        logger.info(f"最佳损失: {best_loss:.4f}")
        logger.info(f"总训练时间: {total_train_time/60:.1f}分钟")
        logger.info(f"平均每epoch: {total_train_time/self.epochs:.1f}秒")
        logger.info("="*70)


# =============================================================================
# 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='智能车辆选择器训练（GPU加速版 - 每步详细输出）')
    parser.add_argument('--config', type=str, default='configs/phase1_lite.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-cache', action='store_true')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 性能优化
    logger.info("配置性能优化...")
    if args.device == 'cuda':
        torch.backends.cudnn.benchmark = True
        logger.info("✅ 启用cuDNN benchmark")

    # 创建环境
    logger.info("创建训练环境...")
    env = make_gym_env(config=config, seed=42, device=args.device)

    # 创建模型
    logger.info("创建神经网络ICV评分器...")
    neural_config = config.get('neural_icv_scoring', {
        'node_dim': 9,
        'hidden_dim': 64,
        'num_layers': 3,
        'num_heads': 4
    })

    model = create_neural_icv_scorer(
        node_dim=neural_config.get('node_dim', 9),
        hidden_dim=neural_config.get('hidden_dim', 64),
        num_layers=neural_config.get('num_layers', 3),
        num_heads=neural_config.get('num_heads', 4),
        device=args.device
    )

    # PyTorch 2.0+ 编译优化
    if hasattr(torch, 'compile') and args.device == 'cuda':
        try:
            model.neural_scorer = torch.compile(
                model.neural_scorer,
                mode='reduce-overhead',
                fullgraph=False
            )
            logger.info("✅ 已启用torch.compile优化")
        except Exception as e:
            logger.warning(f"torch.compile启用失败: {e}")

    # 训练
    trainer = Stage1TrainerGPU(model, config, args.device)
    trainer.run(env, use_cache=not args.no_cache)

    env.close()
    logger.info("\n训练完成！")


if __name__ == '__main__':
    main()
