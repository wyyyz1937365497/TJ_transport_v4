"""
训练增强模块 - 课程学习、优先经验回放、失败案例库

核心功能：
1. 课程学习（Curriculum Learning）：从简单场景到复杂场景的渐进训练
2. 优先经验回放（Prioritized Experience Replay, PER）：高价值状态的优先采样
3. 失败案例库（Failure Case Bank）：专门训练安全模块

使用方法：
    # 在训练脚本中集成
    from src.training.train_enhancements import CurriculumManager, PrioritizedReplayBuffer, FailureCaseBank

    # 1. 课程学习
    curriculum = CurriculumManager(config)
    difficulty_level = curriculum.get_current_difficulty(epoch)

    # 2. 优先经验回放
    replay_buffer = PrioritizedReplayBuffer(capacity=100000, alpha=0.6)
    replay_buffer.add transition, priority)
    batch, indices, weights = replay_buffer.sample(batch_size, beta=0.4)

    # 3. 失败案例库
    failure_bank = FailureCaseBank(max_size=1000)
    failure_bank.add_failure(transition, failure_type)
    failure_batch = failure_bank.sample(batch_size)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import deque
import pickle


# =============================================================================
# 第一部分：课程学习（Curriculum Learning）
# =============================================================================

@dataclass
class DifficultyLevel:
    """难度级别定义"""
    level: int
    name: str
    description: str

    # 环境参数
    max_vehicles: int  # 最大车辆数
    inflow_rate: float  # 流入率（辆/小时）
    icv_ratio: float  # 智能车渗透率
    disturbance_level: float  # 扰动级别（0-1）

    # 训练参数
    episodes: int  # 该级别训练的episodes
    success_threshold: float  # 晋级阈值（成功率）
    min_reward: float  # 最低奖励要求


class CurriculumManager:
    """
    课程学习管理器

    核心策略：
    1. 从简单场景开始（低流量、高渗透率）
    2. 逐步增加难度（高流量、低渗透率、强扰动）
    3. 根据性能动态调整进度
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        curriculum_config = config.get('training', {}).get('curriculum', {})

        # 定义难度级别
        self.levels = self._define_levels()

        # 当前级别
        self.current_level_idx = 0
        self.episodes_in_current_level = 0
        self.success_count = 0
        self.episode_rewards = deque(maxlen=100)  # 最近100个episode的奖励

        # 晋级策略
        self.auto_advance = curriculum_config.get('auto_advance', True)
        self.min_episodes_per_level = curriculum_config.get('min_episodes_per_level', 50)

    def _define_levels(self) -> List[DifficultyLevel]:
        """定义课程学习的难度级别"""
        return [
            # Level 1: 基础场景 - 低流量、高渗透率
            DifficultyLevel(
                level=1,
                name="基础场景",
                description="低流量、高渗透率、无扰动",
                max_vehicles=10,
                inflow_rate=800,
                icv_ratio=0.3,  # 30% ICV（更容易控制）
                disturbance_level=0.0,
                episodes=100,
                success_threshold=0.7,
                min_reward=-100
            ),

            # Level 2: 中等流量 - 适中流量、适中渗透率
            DifficultyLevel(
                level=2,
                name="中等流量",
                description="中等流量、适中渗透率、弱扰动",
                max_vehicles=15,
                inflow_rate=1200,
                icv_ratio=0.25,  # 25% ICV（赛题标准）
                disturbance_level=0.2,
                episodes=150,
                success_threshold=0.6,
                min_reward=-200
            ),

            # Level 3: 高流量 - 高流量、标准渗透率
            DifficultyLevel(
                level=3,
                name="高流量场景",
                description="高流量、标准渗透率、中等扰动",
                max_vehicles=20,
                inflow_rate=1800,
                icv_ratio=0.25,
                disturbance_level=0.4,
                episodes=200,
                success_threshold=0.5,
                min_reward=-300
            ),

            # Level 4: 极端场景 - 极高流量、低渗透率、强扰动
            DifficultyLevel(
                level=4,
                name="极端场景",
                description="极高流量、低渗透率、强扰动",
                max_vehicles=32,
                inflow_rate=2400,
                icv_ratio=0.15,  # 15% ICV（更难）
                disturbance_level=0.7,
                episodes=250,
                success_threshold=0.4,
                min_reward=-400
            ),

            # Level 5: 赛题场景 - 真实比赛条件
            DifficultyLevel(
                level=5,
                name="赛题场景",
                description="真实比赛条件、突发扰动",
                max_vehicles=32,
                inflow_rate=2000,
                icv_ratio=0.25,  # 赛题标准
                disturbance_level=0.5,
                episodes=300,
                success_threshold=0.5,
                min_reward=-350
            )
        ]

    def get_current_difficulty(self) -> DifficultyLevel:
        """获取当前难度级别"""
        return self.levels[self.current_level_idx]

    def update_progress(self, reward: float, success: bool):
        """
        更新训练进度

        Args:
            reward: episode奖励
            success: 是否成功（根据任务定义）
        """
        self.episodes_in_current_level += 1
        self.episode_rewards.append(reward)

        if success:
            self.success_count += 1

        # 检查是否可以晋级
        if self.auto_advance and self._should_advance():
            self._advance_to_next_level()

    def _should_advance(self) -> bool:
        """判断是否应该晋级"""
        if self.current_level_idx >= len(self.levels) - 1:
            return False  # 已经是最高级别

        current_level = self.levels[self.current_level_idx]

        # 条件1：训练足够的episodes
        if self.episodes_in_current_level < self.min_episodes_per_level:
            return False

        # 条件2：达到最低奖励要求
        avg_reward = np.mean(self.episode_rewards)
        if avg_reward < current_level.min_reward:
            return False

        # 条件3：成功率达标（可选）
        if self.episodes_in_current_level >= current_level.episodes:
            success_rate = self.success_count / self.episodes_in_current_level
            if success_rate >= current_level.success_threshold:
                return True

        return False

    def _advance_to_next_level(self):
        """晋级到下一级别"""
        if self.current_level_idx < len(self.levels) - 1:
            self.current_level_idx += 1
            self.episodes_in_current_level = 0
            self.success_count = 0
            self.episode_rewards.clear()

            new_level = self.levels[self.current_level_idx]
            print(f"\n[CURRICULUM] 🎓 晋升到 Level {new_level.level}: {new_level.name}")
            print(f"   描述: {new_level.description}")
            print(f"   参数: 最大车辆={new_level.max_vehicles}, "
                  f"流量={new_level.inflow_rate}, ICV比率={new_level.icv_ratio}")

    def get_env_config_for_current_level(self, base_config: Dict) -> Dict:
        """
        根据当前难度级别生成环境配置

        Args:
            base_config: 基础环境配置

        Returns:
            更新后的环境配置
        """
        current_level = self.get_current_difficulty()

        env_config = base_config.copy()
        env_config['max_vehicles'] = current_level.max_vehicles
        env_config['inflow_rate'] = current_level.inflow_rate
        env_config['icv_ratio'] = current_level.icv_ratio

        # 添加扰动级别
        env_config['disturbance_level'] = current_level.disturbance_level

        return env_config

    def get_stats(self) -> Dict[str, Any]:
        """获取课程学习统计信息"""
        current_level = self.get_current_difficulty()

        return {
            'current_level': current_level.level,
            'level_name': current_level.name,
            'episodes_in_level': self.episodes_in_current_level,
            'total_levels': len(self.levels),
            'success_count': self.success_count,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'progress': (self.current_level_idx + 1) / len(self.levels)
        }


# =============================================================================
# 第二部分：优先经验回放（Prioritized Experience Replay）
# =============================================================================

class SumTree:
    """
    SumTree数据结构 - 用于高效优先级采样

    特性：
    - 叶子节点存储经验的优先级
    - 内部节点存储子树的优先级之和
    - 采样时间复杂度：O(log N)
    - 更新时间复杂度：O(log N)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # 完全二叉树
        self.data = np.zeros(capacity, dtype=object)  # 存储经验
        self.write = 0  # 写入位置
        self.n_entries = 0  # 当前存储的经验数

    def _propagate(self, idx: int, change: float):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """
        根据优先级采样（向下遍历）

        Args:
            idx: 当前节点索引
            s: 累积优先级

        Returns:
            叶子节点索引
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """获取总优先级"""
        return self.tree[0]

    def add(self, priority: float, data: Any):
        """
        添加新经验

        Args:
            priority: 优先级
            data: 经验数据
        """
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float):
        """
        更新优先级

        Args:
            idx: 叶子节点索引
            priority: 新的优先级
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        """
        根据优先级采样

        Args:
            s: 累积优先级值（随机采样）

        Returns:
            (idx, priority, data)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_idx]


@dataclass
class Transition:
    """经验回放Transition"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    # 交通场景特有的额外信息
    info: Dict[str, Any] = field(default_factory=dict)


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区（PER）

    核心特性：
    1. 基于TD误差的优先级采样
    2. 使用SumTree实现高效采样
    3. 重要性采样权重（避免偏差）
    4. 支持高价值状态优先采样（拥堵临界点）

    优先级策略：
    - 方案A: 基于TD误差（标准PER）
    - 方案B: 基于拥堵状态（交通场景特定）
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6
    ):
        """
        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数（0=均匀采样，1=完全按优先级）
            beta_start: 重要性采样权重初始值
            beta_frames: beta线性增长到1的帧数
            epsilon: 避免除零的小常数
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon

        self.sum_tree = SumTree(capacity)
        self.max_priority = 1.0

        # 帧计数器（用于计算beta）
        self.frame = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
        info: Optional[Dict] = None
    ):
        """
        添加新经验

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
            priority: 优先级（None则使用最大优先级）
            info: 额外信息
        """
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info or {}
        )

        if priority is None:
            priority = self.max_priority

        self.sum_tree.add(priority, transition)

    def sample(self, batch_size: int) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        采样一个batch

        Args:
            batch_size: batch大小

        Returns:
            batch_dict: 包含states, actions, rewards, next_states, dones的字典
            indices: 采样的索引
            weights: 重要性采样权重
        """
        batch_data = []
        indices = []
        priorities = []

        # 计算当前beta
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

        # 分段采样（避免总是采样高优先级）
        segment_priority = self.sum_tree.total() / batch_size

        for i in range(batch_size):
            a = segment_priority * i
            b = segment_priority * (i + 1)

            s = np.random.uniform(a, b)
            idx, priority, data = self.sum_tree.get(s)

            batch_data.append(data)
            indices.append(idx)
            priorities.append(priority)

        # 计算重要性采样权重
        sampling_probabilities = np.array(priorities) / self.sum_tree.total()
        is_weights = np.power(self.sum_tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max()  # 归一化

        # 打包成batch
        states = np.array([t.state for t in batch_data])
        actions = np.array([t.action for t in batch_data])
        rewards = np.array([t.reward for t in batch_data])
        next_states = np.array([t.next_state for t in batch_data])
        dones = np.array([t.done for t in batch_data])

        batch_dict = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }

        self.frame += 1

        return batch_dict, np.array(indices), is_weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        更新采样的优先级

        Args:
            indices: 索引数组
            priorities: 新的优先级数组
        """
        for idx, priority in zip(indices, priorities):
            self.update_priority(idx, priority)

    def update_priority(self, idx: int, priority: float):
        """更新单个优先级"""
        priority = (priority + self.epsilon) ** self.alpha
        self.sum_tree.update(idx, priority)
        self.max_priority = max(self.max_priority, priority)

    def compute_traffic_priority(
        self,
        reward: float,
        info: Dict[str, Any],
        td_error: Optional[float] = None
    ) -> float:
        """
        计算交通场景特定的优先级

        策略：
        1. 高TD误差 → 高优先级（标准PER）
        2. 接近拥堵临界点 → 高优先级
        3. 安全事件 → 超高优先级
        4. 低奖励（失败案例）→ 中等优先级

        Args:
            reward: 奖励
            info: 额外信息（包含交通状态）
            td_error: TD误差（可选）

        Returns:
            优先级（0-1之间）
        """
        priority = 0.5  # 基础优先级

        # 1. 基于TD误差（如果提供）
        if td_error is not None:
            priority = max(priority, abs(td_error))

        # 2. 基于奖励（低奖励 = 可能失败）
        if reward < -200:
            priority = max(priority, 0.7)

        # 3. 接近拥堵临界点
        if 'avg_speed' in info:
            avg_speed = info['avg_speed']
            # 如果速度低于5 m/s，说明接近拥堵
            if avg_speed < 5.0:
                priority = max(priority, 0.8)

        # 4. 安全事件（极高优先级）
        if info.get('has_emergency_braking', False):
            priority = 1.0
        elif info.get('has_conflict', False):
            priority = 0.9

        # 5. 干预成本（高成本 = 重要学习案例）
        if 'num_controlled' in info:
            num_controlled = info['num_controlled']
            if num_controlled > 5:  # 控制了很多车
                priority = max(priority, 0.6)

        return min(priority, 1.0)

    def __len__(self) -> int:
        return self.sum_tree.n_entries


# =============================================================================
# 第三部分：失败案例库（Failure Case Bank）
# =============================================================================

@dataclass
class FailureCase:
    """失败案例"""
    state: np.ndarray
    action: np.ndarray
    failure_type: str  # 失败类型
    severity: float  # 严重程度（0-1）
    info: Dict[str, Any]

    # 标签（用于监督学习）
    safety_label: int  # 0=安全, 1=危险
    recommended_action: Optional[np.ndarray] = None  # 推荐的正确动作


class FailureCaseBank:
    """
    失败案例库

    核心功能：
    1. 收集失败案例（碰撞、急刹、严重拥堵）
    2. 按严重程度和类型分类
    3. 优先采样高风险案例
    4. 支持安全模块的专门训练

    失败类型：
    - 'collision': 碰撞
    - 'emergency_braking': 急刹
    - 'severe_congestion': 严重拥堵
    - 'high_intervention_cost': 高干预成本
    - 'low_efficiency': 低效率
    """

    def __init__(self, max_size: int = 1000):
        """
        Args:
            max_size: 最大存储案例数
        """
        self.max_size = max_size
        self.cases = deque(maxlen=max_size)

        # 按类型分类的索引
        self.collision_cases = deque(maxlen=200)
        self.emergency_braking_cases = deque(maxlen=300)
        self.congestion_cases = deque(maxlen=300)
        self.inefficiency_cases = deque(maxlen=200)

    def add_failure(
        self,
        state: np.ndarray,
        action: np.ndarray,
        failure_type: str,
        severity: float = 0.5,
        info: Optional[Dict] = None,
        recommended_action: Optional[np.ndarray] = None
    ):
        """
        添加失败案例

        Args:
            state: 状态
            action: 导致失败的动作
            failure_type: 失败类型
            severity: 严重程度（0-1）
            info: 额外信息
            recommended_action: 推荐的正确动作（可选）
        """
        # 安全标签：危险=1，安全=0
        safety_label = 1 if severity > 0.5 else 0

        case = FailureCase(
            state=state,
            action=action,
            failure_type=failure_type,
            severity=severity,
            info=info or {},
            safety_label=safety_label,
            recommended_action=recommended_action
        )

        self.cases.append(case)

        # 分类存储
        if failure_type == 'collision':
            self.collision_cases.append(case)
        elif failure_type == 'emergency_braking':
            self.emergency_braking_cases.append(case)
        elif failure_type == 'severe_congestion':
            self.congestion_cases.append(case)
        elif failure_type == 'low_efficiency':
            self.inefficiency_cases.append(case)

    def detect_failure(
        self,
        state: Dict[str, Any],
        action: np.ndarray,
        next_state: Dict[str, Any],
        reward: float
    ) -> Optional[Tuple[str, float]]:
        """
        自动检测失败案例

        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 下一个状态
            reward: 奖励

        Returns:
            (failure_type, severity) 或 None
        """
        # 1. 检测碰撞
        if 'collision' in next_state and next_state['collision']:
            return 'collision', 1.0

        # 2. 检测急刹
        if 'emergency_braking_count' in next_state:
            if next_state['emergency_braking_count'] > 0:
                return 'emergency_braking', 0.8

        # 3. 检测严重拥堵
        if 'avg_speed' in next_state:
            if next_state['avg_speed'] < 2.0:  # 几乎停止
                return 'severe_congestion', 0.9

        # 4. 检测低效率
        if reward < -500:
            return 'low_efficiency', 0.6

        # 5. 检测高干预成本
        if 'num_controlled' in next_state:
            if next_state['num_controlled'] > 10:  # 控制了太多车
                return 'high_intervention_cost', 0.5

        return None

    def sample(
        self,
        batch_size: int,
        failure_type: Optional[str] = None,
        min_severity: float = 0.0
    ) -> List[FailureCase]:
        """
        采样失败案例

        Args:
            batch_size: batch大小
            failure_type: 失败类型（None则从所有类型采样）
            min_severity: 最小严重程度

        Returns:
            失败案例列表
        """
        # 选择案例池
        if failure_type == 'collision':
            pool = self.collision_cases
        elif failure_type == 'emergency_braking':
            pool = self.emergency_braking_cases
        elif failure_type == 'severe_congestion':
            pool = self.congestion_cases
        elif failure_type == 'low_efficiency':
            pool = self.inefficiency_cases
        else:
            pool = self.cases

        # 过滤严重程度
        filtered = [c for c in pool if c.severity >= min_severity]

        if len(filtered) == 0:
            return []

        # 按严重程度加权采样
        severities = np.array([c.severity for c in filtered])
        probs = severities / severities.sum()

        indices = np.random.choice(
            len(filtered),
            size=min(batch_size, len(filtered)),
            replace=False,
            p=probs
        )

        return [filtered[i] for i in indices]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_cases': len(self.cases),
            'collision_cases': len(self.collision_cases),
            'emergency_braking_cases': len(self.emergency_braking_cases),
            'congestion_cases': len(self.congestion_cases),
            'inefficiency_cases': len(self.inefficiency_cases),
            'avg_severity': np.mean([c.severity for c in self.cases]) if self.cases else 0.0
        }


# =============================================================================
# 第四部分：集成训练器（使用所有增强功能）
# =============================================================================

class EnhancedTrainingManager:
    """
    增强训练管理器 - 集成所有增强功能

    使用方法：
        manager = EnhancedTrainingManager(config)

        # 训练循环
        for episode in range(num_episodes):
            # 1. 获取当前难度级别的环境配置
            env_config = manager.get_curriculum_env_config()

            # 2. 运行episode
            for step in range(max_steps):
                action = model.predict(state)
                next_state, reward, done, info = env.step(action)

                # 3. 检测失败案例
                failure_info = manager.detect_failure(state, action, next_state, reward)
                if failure_info:
                    manager.add_failure_case(...)

                # 4. 存储到优先经验回放
                priority = manager.compute_priority(reward, info, td_error)
                manager.replay_buffer.add(..., priority=priority)

                state = next_state

            # 5. 更新课程学习进度
            manager.update_curriculum_progress(episode_reward, success)

            # 6. 训练时混合采样
            normal_batch = manager.replay_buffer.sample(batch_size // 2)
            failure_batch = manager.failure_bank.sample(batch_size // 2)

    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 1. 课程学习
        curriculum_config = config.get('training', {}).get('curriculum', {})
        if curriculum_config.get('enabled', False):
            self.curriculum = CurriculumManager(config)
        else:
            self.curriculum = None

        # 2. 优先经验回放
        per_config = config.get('training', {}).get('prioritized_replay', {})
        if per_config.get('enabled', False):
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=per_config.get('capacity', 100000),
                alpha=per_config.get('alpha', 0.6),
                beta_start=per_config.get('beta_start', 0.4),
                beta_frames=per_config.get('beta_frames', 100000)
            )
        else:
            self.replay_buffer = None

        # 3. 失败案例库
        failure_config = config.get('training', {}).get('failure_bank', {})
        if failure_config.get('enabled', False):
            self.failure_bank = FailureCaseBank(
                max_size=failure_config.get('max_size', 1000)
            )
        else:
            self.failure_bank = None

    def get_curriculum_env_config(self, base_config: Dict) -> Dict:
        """获取当前课程学习级别的环境配置"""
        if self.curriculum is None:
            return base_config

        return self.curriculum.get_env_config_for_current_level(base_config)

    def update_curriculum_progress(self, reward: float, success: bool):
        """更新课程学习进度"""
        if self.curriculum is not None:
            self.curriculum.update_progress(reward, success)

    def detect_failure(
        self,
        state: Dict,
        action: np.ndarray,
        next_state: Dict,
        reward: float
    ):
        """检测失败案例"""
        if self.failure_bank is None:
            return None

        return self.failure_bank.detect_failure(state, action, next_state, reward)

    def add_failure_case(
        self,
        state: np.ndarray,
        action: np.ndarray,
        failure_type: str,
        severity: float,
        info: Dict
    ):
        """添加失败案例"""
        if self.failure_bank is not None:
            self.failure_bank.add_failure(state, action, failure_type, severity, info)

    def compute_priority(
        self,
        reward: float,
        info: Dict,
        td_error: Optional[float] = None
    ) -> float:
        """计算优先级"""
        if self.replay_buffer is None:
            return 0.5

        return self.replay_buffer.compute_traffic_priority(reward, info, td_error)

    def get_stats(self) -> Dict[str, Any]:
        """获取所有增强功能的统计信息"""
        stats = {}

        if self.curriculum is not None:
            stats['curriculum'] = self.curriculum.get_stats()

        if self.replay_buffer is not None:
            stats['replay_buffer'] = {
                'size': len(self.replay_buffer),
                'frame': self.replay_buffer.frame
            }

        if self.failure_bank is not None:
            stats['failure_bank'] = self.failure_bank.get_stats()

        return stats


# =============================================================================
# 辅助函数
# =============================================================================

def create_enhanced_training_manager(config: Dict[str, Any]) -> EnhancedTrainingManager:
    """
    工厂函数：创建增强训练管理器

    Args:
        config: 配置字典

    Returns:
        EnhancedTrainingManager实例
    """
    return EnhancedTrainingManager(config)


def save_failure_cases(failure_bank: FailureCaseBank, path: str):
    """保存失败案例库到文件"""
    with open(path, 'wb') as f:
        pickle.dump(failure_bank, f)


def load_failure_cases(path: str) -> FailureCaseBank:
    """从文件加载失败案例库"""
    with open(path, 'rb') as f:
        return pickle.load(f)
