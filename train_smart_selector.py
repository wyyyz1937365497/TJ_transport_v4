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
    HeuristicVehicleSelector as BaseHeuristicSelector,
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
        
        # 启发式选择器和控制器
        self.heuristic_selector = HeuristicVehicleSelector(config)
        self.heuristic_controller = HeuristicController(config)
        
        # 优化器（优化神经网络评分器内部的neural_scorer）
        self.optimizer = optim.Adam(
            self.model.neural_scorer.parameters(),
            lr=self.learning_rate
        )
        
        # 数据缓冲区
        self.buffer = []
        
    def collect_data(self, env) -> int:
        """
        使用启发式策略收集训练数据
        
        关键：记录启发式选择的车辆，让神经网络学习模仿
        """
        logger.info("开始收集启发式数据...")
        
        num_samples = 0
        
        for episode in tqdm(range(self.num_episodes), desc="数据收集"):
            obs, info = env.reset()
            
            for step in range(3600):  # 每个episode 3600步
                if not hasattr(env, 'sumo_env'):
                    break
                    
                sumo_env = env.sumo_env
                
                # 获取车辆状态
                vehicle_states = {}
                icv_ids = []
                
                if hasattr(sumo_env, 'traci_lib'):
                    all_vehicles = sumo_env.traci_lib.vehicle.getIDList()
                    
                    for veh_id in all_vehicles:
                        try:
                            # 基本状态
                            speed = sumo_env.traci_lib.vehicle.getSpeed(veh_id)
                            position = sumo_env.traci_lib.vehicle.getPosition(veh_id)
                            lane_id = sumo_env.traci_lib.vehicle.getLaneID(veh_id)
                            
                            vehicle_states[veh_id] = {
                                'speed': speed,
                                'position': position,
                                'lane_id': lane_id,
                                's': position[0],  # 简化
                                'lane_index': 0,
                                'lead_distance': 50.0,
                                'max_speed': 33.33
                            }
                            
                            # ICV标记
                            if veh_id.startswith('icv_'):
                                icv_ids.append(veh_id)
                        except:
                            pass
                
                if len(icv_ids) < 10:
                    # 车辆太少，跳过
                    action = {'vehicle_ids': [], 'actions': []}
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                    continue
                
                # 使用启发式选择器选择车辆
                selected_ids = self.heuristic_selector.select_vehicles(
                    vehicle_states,
                    icv_ids,
                    top_k=min(30, len(icv_ids) // 5)
                )
                
                # 生成控制动作
                actions_dict = self.heuristic_controller.compute_actions(
                    vehicle_states,
                    selected_ids
                )
                
                # 执行动作
                action = {
                    'vehicle_ids': list(actions_dict.keys()),
                    'actions': list(actions_dict.values())
                }
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 记录训练样本
                if len(selected_ids) > 0:
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
        return num_samples
    
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
                scores = self.model.compute_scores(vehicle_states)
                
                # 提取ICV的评分和标签
                pred_scores = []
                true_labels = []
                
                for veh_id in icv_ids:
                    if veh_id in scores and veh_id in labels:
                        pred_scores.append(scores[veh_id] / 53.0)  # 归一化
                        true_labels.append(labels[veh_id])
                
                if len(pred_scores) == 0:
                    continue
                
                pred_tensor = torch.tensor(pred_scores, dtype=torch.float32, device=self.device)
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
                logger.warning(f"训练样本出错: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def run(self, env):
        """运行Stage 1训练"""
        logger.info("="*70)
        logger.info("Stage 1: 启发式引导训练")
        logger.info("="*70)
        
        # 收集数据
        num_samples = self.collect_data(env)
        
        if num_samples == 0:
            logger.error("没有收集到数据！")
            return
        
        logger.info(f"数据集大小: {len(self.buffer)}")
        
        # 训练
        best_loss = float('inf')
        
        for epoch in range(self.epochs):
            avg_loss = self.train_epoch(epoch)
            
            logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = 'checkpoints/smart_selector_stage1.pth'
                Path('checkpoints').mkdir(exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'loss': avg_loss
                }, checkpoint_path)
        
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
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
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
    
    # 运行Stage 1训练
    trainer = Stage1Trainer(model, config, args.device)
    trainer.run(env)
    
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
