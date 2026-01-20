"""
智能车辆选择器训练脚本

核心功能：
1. Stage 1: 启发式引导（监督学习，训练神经网络评分器）
2. 与现有训练系统集成（使用v5_ppo_trainer.py进行Stage 2强化学习）
3. 提供启发式车辆选择器和控制器（数据生成和评估）

使用方法：
    # Stage 1: 训练神经网络评分器模仿启发式选择
    python train_smart_selector.py --config configs/phase1_lite.yaml --stage 1
    
    # Stage 2: 使用train_phase1_lite.py进行强化学习微调
    python train_phase1_lite.py --config configs/phase1_lite.yaml --stage 2

设计说明：
- Stage 1在此脚本中完成（监督学习，独立训练）
- Stage 2使用现有的v5_ppo_trainer.py（避免重复实现）
- 启发式选择器可作为baseline或数据生成器使用
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# 导入项目模块
from src.models.neural_vehicle_scorer import (
    NeuralICVScorer,
    create_neural_icv_scorer
)
from src.env.gym_wrapper import make_gym_env
from src.env.vehicle_scoring import (
    RuleBasedScorer,
    UnifiedVehicleScorer,
    create_vehicle_scorer_from_config
)

# =============================================================================
# 日志配置
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 启发式车辆选择器（扩展版，兼容现有系统）
# =============================================================================
class HeuristicVehicleSelector:
    """
    基于交通工程经验的启发式车辆选择器（扩展版）
    
    功能：
    1. 兼容BaseHeuristicSelector（来自vehicle_scoring.py）
    2. 添加额外的选择逻辑（针对训练优化）
    3. 可作为数据生成器或baseline
    
    选择标准：
    1. 瓶颈区域的车辆（高优先级）
    2. 速度慢的车辆（容易造成拥堵）
    3. 即将换道的车辆（影响交通流）
    4. 接近匝道合流点的车辆
    """
    
    def __init__(self, config: Dict, frenet_system=None):
        self.config = config
        self.frenet_system = frenet_system
        
        # 瓶颈区域定义（Frenet坐标系）
        heuristic_config = config.get('heuristic_selector', {})
        self.bottleneck_s_min = heuristic_config.get('bottleneck_s_min', 1000.0)
        self.bottleneck_s_max = heuristic_config.get('bottleneck_s_max', 2000.0)
        
        # 如果有BaseHeuristicSelector，也创建一个作为参考
        if frenet_system is not None:
            from src.env.vehicle_scoring import RuleBasedScorer
            self.rule_scorer = RuleBasedScorer(frenet_system)
        else:
            self.rule_scorer = None
        
    def select_vehicles(
        self,
        vehicle_states: Dict[str, Dict],
        icv_ids: List[str],
        top_k: int = 30,
        use_rule_scorer: bool = False
    ) -> List[str]:
        """
        使用启发式规则选择关键车辆
        
        Args:
            vehicle_states: 所有车辆状态
            icv_ids: 可控制的ICV列表
            top_k: 选择数量
            use_rule_scorer: 是否使用RuleBasedScorer（如果可用）
            
        Returns:
            选中的车辆ID列表
        """
        # 选项1: 使用RuleBasedScorer（更精确）
        if use_rule_scorer and self.rule_scorer is not None:
            try:
                scores = self.rule_scorer.compute_scores(vehicle_states)
                icv_scores = {veh_id: scores.get(veh_id, 0.0) for veh_id in icv_ids}
                sorted_vehicles = sorted(icv_scores.items(), key=lambda x: x[1], reverse=True)
                return [veh_id for veh_id, _ in sorted_vehicles[:top_k]]
            except:
                pass  # Fallback到简化版本
        
        # 选项2: 简化评分（快速）
        scores = {}
        
        for veh_id in icv_ids:
            if veh_id not in vehicle_states:
                continue
                
            state = vehicle_states[veh_id]
            score = 0.0
            
            # 1. 瓶颈区域权重（最重要）
            s = state.get('s', 0.0)
            if self.bottleneck_s_min <= s <= self.bottleneck_s_max:
                score += 10.0
            
            # 2. 速度因子（速度越慢越重要）
            speed = state.get('speed', 0.0)
            max_speed = state.get('max_speed', 33.33)  # 120km/h
            if speed < 0.5 * max_speed:
                score += 5.0 * (1.0 - speed / max_speed)
            
            # 3. 车道因子（内侧车道更重要）
            lane_index = state.get('lane_index', 0)
            if lane_index <= 1:  # 最内侧两个车道
                score += 3.0
            
            # 4. 与前车距离因子（距离近需要控制）
            lead_dist = state.get('lead_distance', 100.0)
            if lead_dist < 30.0:
                score += 4.0 * (1.0 - lead_dist / 30.0)
            
            scores[veh_id] = score
        
        # 按分数排序，选择Top-K
        sorted_vehicles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [veh_id for veh_id, _ in sorted_vehicles[:top_k]]
        
        return selected


# =============================================================================
# 简单控制策略（用于Stage 1数据生成）
# =============================================================================
class HeuristicController:
    """
    基于规则的简单控制策略
    
    控制原则：
    1. 保持安全车距
    2. 平滑加速/减速
    3. 避免频繁换道
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_accel = config.get('max_accel', 2.0)
        self.max_decel = config.get('max_decel', -4.0)
        self.safe_time_gap = config.get('safe_time_gap', 1.5)  # 秒
        
    def compute_actions(
        self,
        vehicle_states: Dict[str, Dict],
        selected_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        为选中车辆生成控制动作
        
        Returns:
            {vehicle_id: [acceleration, lane_change]}
        """
        actions = {}
        
        for veh_id in selected_ids:
            if veh_id not in vehicle_states:
                continue
                
            state = vehicle_states[veh_id]
            
            # IDM (Intelligent Driver Model)
            accel = self._compute_idm_accel(state)
            
            # 换道决策（简化：很少换道以降低成本）
            lane_change = 0  # 0: 保持, -1: 左, 1: 右
            
            # 只有在特殊情况下才换道
            if state.get('lead_distance', 100.0) < 10.0 and state.get('lane_index', 0) > 0:
                # 前车太近，尝试左换道（假设左边是快速车道）
                lane_change = -1
            
            actions[veh_id] = np.array([accel, lane_change], dtype=np.float32)
        
        return actions
    
    def _compute_idm_accel(self, state: Dict) -> float:
        """IDM加速度计算"""
        speed = state.get('speed', 0.0)
        max_speed = state.get('max_speed', 33.33)
        lead_dist = state.get('lead_distance', 100.0)
        lead_speed = state.get('lead_speed', max_speed)
        
        # IDM参数
        desired_time_gap = self.safe_time_gap
        min_spacing = 2.0
        
        # 期望速度项
        speed_term = 1.0 - (speed / max_speed) ** 4
        
        # 距离项
        desired_spacing = min_spacing + speed * desired_time_gap
        spacing_term = (desired_spacing / max(lead_dist, 0.1)) ** 2
        
        # 组合
        accel = self.max_accel * (speed_term - spacing_term)
        
        # 裁剪
        accel = np.clip(accel, self.max_decel, self.max_accel)
        
        return accel


# =============================================================================
# Stage 1: 启发式引导训练
# =============================================================================

class SelectorDataset(torch.utils.data.Dataset):
    """
    可序列化的数据集类（必须定义在模块级别以支持多进程）
    """
    def __init__(self, buffer):
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


class Stage1Trainer:
    """
    Stage 1：使用启发式策略生成训练数据，训练神经网络模仿
    
    目标：让神经网络学习"哪些车重要"的基本模式
    方法：监督学习（模仿启发式选择器）
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
        self.batch_size = stage1_config.get('batch_size', 256)  # 默认增加到256
        
        # 启发式选择器和控制器
        self.heuristic_selector = HeuristicVehicleSelector(config)
        self.heuristic_controller = HeuristicController(config)
        
        # 优化器（优化神经网络评分器内部的neural_scorer）
        self.optimizer = optim.Adam(
            self.model.neural_scorer.parameters(),
            lr=self.learning_rate
        )

        # 混合精度训练（AMP）
        self.use_amp = config.get('advanced', {}).get('amp', True)
        if self.use_amp and device == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("✅ 启用混合精度训练（AMP）")
        else:
            self.scaler = None

        # 数据缓冲区
        self.buffer = []
        
        # 数据缓存路径
        self.cache_dir = Path('collected_data')
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / 'smart_selector_stage1_cache.pkl'
        
    def save_cache(self):
        """保存收集的数据到缓存"""
        import pickle
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.buffer, f)
        logger.info(f"✅ 数据已缓存到: {self.cache_file} ({len(self.buffer)}个样本)")
        
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
        """
        使用启发式策略收集训练数据
        
        关键：记录启发式选择的车辆，让神经网络学习模仿
        """
        # 尝试加载缓存
        if use_cache and self.load_cache():
            logger.info(f"使用缓存数据，跳过数据收集")
            return len(self.buffer)
            
        logger.info("开始收集启发式数据...")
        
        num_samples = 0
        
        for episode in tqdm(range(self.num_episodes), desc="数据收集"):
            obs, info = env.reset()
            
            for step in range(3600):  # 每个episode 3600步
                # 从观测中提取车辆状态
                if isinstance(obs, dict):
                    vehicle_ids = obs.get('vehicle_ids', [])
                    icv_ids = list(obs.get('icv_ids', []))
                    vehicle_states_array = obs.get('vehicle_states', np.zeros((0, 9)))
                else:
                    # 如果观测不是字典格式，跳过
                    action = {'vehicle_ids': [], 'actions': np.array([], dtype=np.float32).reshape(0, 2)}
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                    continue
                
                # 如果ICV数量太少，跳过
                if len(icv_ids) < 10:
                    action = {'vehicle_ids': [], 'actions': np.array([], dtype=np.float32).reshape(0, 2)}
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                    continue
                
                # 构建vehicle_states字典（启发式选择器需要）
                vehicle_states = {}
                for i, veh_id in enumerate(vehicle_ids):
                    if i < len(vehicle_states_array):
                        state = vehicle_states_array[i]
                        vehicle_states[veh_id] = {
                            's': float(state[0]) * 1000.0,  # 反归一化
                            'd': float(state[1]) * 10.0,
                            'speed': float(state[4]) * 30.0,
                            'lane_index': int(state[6] * 10.0),
                            'position': (float(state[0]) * 1000.0, float(state[1]) * 10.0),
                            'lane_id': f"lane_{int(state[6] * 10.0)}",
                            'lead_distance': 50.0,
                            'max_speed': 33.33
                        }
                
                # 使用启发式选择器选择车辆
                selected_ids = self.heuristic_selector.select_vehicles(
                    vehicle_states,
                    icv_ids,
                    top_k=min(30, max(1, len(icv_ids) // 5))  # 至少选1个
                )
                
                # 生成控制动作
                actions_dict = self.heuristic_controller.compute_actions(
                    vehicle_states,
                    selected_ids
                )
                
                # 执行动作
                action = {
                    'vehicle_ids': list(actions_dict.keys()),
                    'actions': np.array(list(actions_dict.values()), dtype=np.float32)
                }
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 记录训练样本（每隔10步记录一次，减少冗余）
                if len(icv_ids) >= 10 and step % 10 == 0:
                    # 创建标签：选中的车辆标记为1，其他为0
                    labels = {veh_id: 1.0 if veh_id in selected_ids else 0.0
                             for veh_id in icv_ids}
                    
                    self.buffer.append({
                        'vehicle_states': vehicle_states.copy(),
                        'icv_ids': icv_ids.copy(),
                        'labels': labels
                    })
                    num_samples += 1
                
                if terminated or truncated:
                    break
        
        logger.info(f"数据收集完成！共收集 {num_samples} 个样本")
        logger.info(f"数据集大小: {len(self.buffer)}")
        
        # 保存缓存
        self.save_cache()
        
        return len(self.buffer)
    
    def train_epoch_batched(self, epoch: int) -> float:
        """
        使用DataLoader批量训练，提高GPU利用率

        性能优化：
        1. 真正的批量处理（而非循环单个样本）
        2. 向量化特征提取
        3. pin_memory + non_blocking传输
        """
        from torch.utils.data import DataLoader

        # 简单collate函数：直接返回batch（不做复杂处理，避免pickle问题）
        def collate_fn(batch):
            return batch

        dataset = SelectorDataset(self.buffer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # 暂时使用0避免pickle问题
            pin_memory=True,  # ✅ 启用pin_memory加速传输
            collate_fn=collate_fn
        )

        total_loss = 0.0
        num_batches = 0
        total_samples_processed = 0

        self.model.neural_scorer.train()

        for batch in dataloader:
            if batch is None:
                continue

            # batch是List[Tuple]，每个元素是(vehicle_states, icv_ids, labels)
            # 在CPU上预处理整个batch（优化版本）
            batch_loss = 0.0
            batch_samples = 0

            # 准备批次数据（在CPU上组装）
            all_features = []
            all_labels = []
            all_edge_indices = []
            node_offsets = [0]  # 用于拼接图的节点偏移

            # 预分配列表大小以减少内存分配开销
            estimated_size = len(batch) * 20  # 估计每个样本20辆车
            all_features_prealloc = []
            all_labels_prealloc = []

            for vehicle_states, icv_ids, labels in batch:
                try:
                    # ✅ 优化1：快速过滤有效车辆
                    valid_vehicles = [
                        (veh_id, vehicle_states[veh_id], labels[veh_id])
                        for veh_id in icv_ids
                        if veh_id in vehicle_states and veh_id in labels
                    ]

                    if len(valid_vehicles) == 0:
                        continue

                    # ✅ 优化2：向量化特征提取（避免Python循环）
                    num_vehicles = len(valid_vehicles)
                    features_array = np.zeros((num_vehicles, 9), dtype=np.float32)
                    labels_array = np.zeros(num_vehicles, dtype=np.float32)

                    for i, (veh_id, state, label) in enumerate(valid_vehicles):
                        # 批量赋值（比逐个append快）
                        features_array[i, 0] = state['s'] / 1000.0
                        features_array[i, 1] = state['d'] / 10.0
                        features_array[i, 2] = state['speed'] / 30.0
                        features_array[i, 3] = 0.0  # vd
                        features_array[i, 4] = state['speed'] / 30.0
                        features_array[i, 5] = state.get('acceleration', 0.0) / 3.0
                        features_array[i, 6] = state['lane_index'] / 10.0
                        features_array[i, 7] = 0.0  # angle
                        features_array[i, 8] = 1.0  # is_icv
                        labels_array[i] = label

                    # ✅ 优化3：构建边索引（向量化）
                    if num_vehicles > 1:
                        # 使用三角索引创建全连接图（无自环）
                        ii, jj = np.triu_indices(num_vehicles, k=1)
                        # 创建双向边（无向图）
                        edge_index = np.stack([
                            np.concatenate([ii, jj]),
                            np.concatenate([jj, ii])
                        ], axis=0).astype(np.int64)
                    else:
                        edge_index = np.zeros((2, 0), dtype=np.int64)

                    # 添加到批次
                    all_features.append(features_array)
                    all_labels.append(labels_array)
                    all_edge_indices.append(edge_index)
                    node_offsets.append(node_offsets[-1] + num_vehicles)
                except Exception as e:
                    logger.warning(f"特征提取失败: {e}")
                    continue

            if len(all_features) == 0:
                continue

            # ✅ 关键优化：在CPU上拼接大批次数据，一次性传输到GPU
            try:
                # 1. 在CPU上拼接所有特征和标签
                batch_features = np.concatenate(all_features, axis=0)  # [total_nodes, 9]
                batch_labels = np.concatenate(all_labels, axis=0)      # [total_nodes]

                # 2. 拼接边索引（需要考虑节点偏移）
                edge_list = []
                for i, edge_index in enumerate(all_edge_indices):
                    if edge_index.shape[1] > 0:
                        offset = node_offsets[i]
                        # 添加偏移量
                        edge_index_offset = edge_index.copy()
                        edge_index_offset[0] += offset
                        edge_index_offset[1] += offset
                        edge_list.append(edge_index_offset)

                if len(edge_list) > 0:
                    batch_edge_index = np.concatenate(edge_list, axis=1)
                else:
                    batch_edge_index = np.zeros((2, 0), dtype=np.int64)

                # 3. ✅ 一次性传输到GPU（使用non_blocking）
                features_tensor = torch.from_numpy(batch_features).to(
                    self.device, non_blocking=True
                )
                labels_tensor = torch.from_numpy(batch_labels).to(
                    self.device, non_blocking=True
                )
                edge_index_tensor = torch.from_numpy(batch_edge_index).to(
                    self.device, non_blocking=True
                )

                # 4. ✅ 批量前向传播（一次处理整个batch）
                # 使用混合精度训练
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        pred_scores = self.model.neural_scorer(
                            features_tensor,
                            edge_index_tensor
                        ).squeeze()

                        if pred_scores.dim() == 0:
                            pred_scores = pred_scores.unsqueeze(0)

                        # 5. 计算损失
                        loss = nn.BCELoss()(pred_scores, labels_tensor)

                    # 6. 反向传播（混合精度）
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    pred_scores = self.model.neural_scorer(
                        features_tensor,
                        edge_index_tensor
                    ).squeeze()

                    if pred_scores.dim() == 0:
                        pred_scores = pred_scores.unsqueeze(0)

                    # 5. 计算损失
                    loss = nn.BCELoss()(pred_scores, labels_tensor)

                    # 6. 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                batch_samples += len(batch)
                total_samples_processed += len(batch)

            except Exception as e:
                import traceback
                logger.warning(f"批次训练出错: {e}\n{traceback.format_exc()}")
                continue

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, Samples: {total_samples_processed}/{len(self.buffer)}, Batches: {num_batches}")
        return avg_loss

    def train_epoch(self, epoch: int) -> float:
        """
        训练一个epoch
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # 随机打乱
        import random
        random.shuffle(self.buffer)
        
        for sample in tqdm(self.buffer, desc=f"Epoch {epoch+1}"):
            vehicle_states = sample['vehicle_states']
            icv_ids = sample['icv_ids']
            labels = sample['labels']
            
            if len(icv_ids) == 0:
                continue
            
            # 计算神经网络评分
            try:
                # 检查样本是否有效
                if len(vehicle_states) == 0 or len(icv_ids) == 0:
                    continue
                
                # ✅ 训练模式：绕过@torch.no_grad()，直接使用neural_scorer
                vehicle_ids = list(vehicle_states.keys())
                num_vehicles = len(vehicle_ids)
                
                # 1. 提取特征
                vehicle_features = []
                for veh_id in vehicle_ids:
                    state = vehicle_states[veh_id]
                    features = self.model._extract_features(state)
                    vehicle_features.append(features)
                
                vehicle_features = torch.tensor(
                    vehicle_features,
                    dtype=torch.float32,
                    device=self.device
                )
                
                # 2. 构建图
                edge_index, edge_attr = self.model.graph_builder.build_graph(
                    vehicle_states,
                    num_vehicles
                )
                
                # 3. 前向传播（带梯度）
                self.model.neural_scorer.train()  # 确保训练模式
                neural_scores = self.model.neural_scorer(
                    vehicle_features,
                    edge_index,
                    edge_attr
                )  # [N], 范围0-1
                
                # 4. 提取ICV的评分和标签
                pred_scores = []
                true_labels = []
                
                for i, veh_id in enumerate(vehicle_ids):
                    if veh_id in icv_ids and veh_id in labels:
                        pred_scores.append(neural_scores[i])
                        true_labels.append(labels[veh_id])
                
                if len(pred_scores) == 0:
                    continue
                
                pred_tensor = torch.stack(pred_scores)
                label_tensor = torch.tensor(true_labels, dtype=torch.float32, device=self.device)
                
                # BCE损失（因为是二分类：是否应该控制）
                loss = nn.BCEWithLogitsLoss()(pred_tensor, label_tensor)
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                import traceback
                logger.warning(f"训练样本出错: {e}")
                logger.debug(f"详细错误:\n{traceback.format_exc()}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        return avg_loss
    
    def run(self, env, use_cache: bool = True):
        """运行Stage 1训练"""
        logger.info("="*70)
        logger.info("Stage 1: 启发式引导训练")
        logger.info("="*70)
        
        # 收集数据
        num_samples = self.collect_data(env, use_cache=use_cache)
        
        if num_samples == 0:
            logger.error("没有收集到数据！")
            return
        
        logger.info(f"数据集大小: {len(self.buffer)}")
        
        # 训练
        best_loss = float('inf')
        
        for epoch in range(self.epochs):
            # 使用批处理训练
            avg_loss = self.train_epoch_batched(epoch)
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = 'checkpoints/smart_selector_stage1.pth'
                Path('checkpoints').mkdir(exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.neural_scorer.state_dict(),  # 保存neural_scorer的state_dict
                    'epoch': epoch,
                    'loss': avg_loss
                }, checkpoint_path)
                logger.info(f"保存最佳模型: {checkpoint_path} (loss={avg_loss:.4f})")
                logger.info(f"保存最佳模型: {checkpoint_path} (loss={avg_loss:.4f})")
        
        logger.info("Stage 1完成！")
        logger.info(f"最佳损失: {best_loss:.4f}")
        logger.info(f"\n下一步: 使用以下命令进行Stage 2强化学习微调:")
        logger.info(f"  python train_phase1_lite.py --config configs/phase1_lite.yaml --stage 2 \\")
        logger.info(f"         --checkpoint checkpoints/smart_selector_stage1.pth")


# =============================================================================
# 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='智能车辆选择器训练（Stage 1）')
    parser.add_argument('--config', type=str, default='configs/phase1_lite.yaml',
                        help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备（cuda或cpu）')
    parser.add_argument('--no-cache', action='store_true',
                        help='不使用缓存数据，重新收集')
    parser.add_argument('--clear-cache', action='store_true',
                        help='清除现有缓存')
    
    args = parser.parse_args()
    
    # 清除缓存
    if args.clear_cache:
        cache_file = Path('collected_data/smart_selector_stage1_cache.pkl')
        if cache_file.exists():
            cache_file.unlink()
            logger.info(f"✅ 已清除缓存: {cache_file}")
        else:
            logger.info("缓存文件不存在")
        return
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # ========== 性能优化配置 ==========
    logger.info("配置性能优化...")

    # 1. 启用cuDNN自动优化（为固定输入大小选择最优算法）
    if args.device == 'cuda':
        torch.backends.cudnn.benchmark = True
        logger.info("✅ 启用cuDNN benchmark模式")

        # 2. 启用确定性模式（可选，用于可复现性，会略微降低性能）
        # torch.backends.cudnn.deterministic = True

    # 3. 设置PyTorch内存优化
    # 减少内存碎片，提高内存分配效率
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    logger.info("✅ 配置CUDA内存分配优化")

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

    # 4. PyTorch 2.0+ 编译优化（实验性，可显著提升性能）
    if hasattr(torch, 'compile') and args.device == 'cuda':
        try:
            logger.info("尝试启用torch.compile优化...")
            model.neural_scorer = torch.compile(
                model.neural_scorer,
                mode='reduce-overhead',  # 减少启动开销
                fullgraph=False  # 不强制整个图编译
            )
            logger.info("✅ 已启用torch.compile优化")
        except Exception as e:
            logger.warning(f"torch.compile启用失败: {e}，使用常规模式")
    else:
        logger.info("跳过torch.compile（不可用或非CUDA设备）")
    
    # 运行Stage 1训练
    trainer = Stage1Trainer(model, config, args.device)
    trainer.run(env, use_cache=not args.no_cache)
    
    # 关闭环境
    env.close()
    
    logger.info("\n训练完成！")
    logger.info("\n" + "="*70)
    logger.info("训练流程总结")
    logger.info("="*70)
    logger.info("✅ Stage 1完成: 神经网络评分器已训练")
    logger.info("   模型保存位置: checkpoints/smart_selector_stage1.pth")
    logger.info("\n下一步:")
    logger.info("  1. 评估Stage 1模型的选择质量")
    logger.info("  2. 运行Stage 2强化学习微调（使用现有的train_phase1_lite.py）:")
    logger.info("     python train_phase1_lite.py --config configs/phase1_lite.yaml --stage 2 \\")
    logger.info("            --checkpoint checkpoints/smart_selector_stage1.pth")
    logger.info("="*70)


if __name__ == '__main__':
    main()
