"""
权重转换工具 - 用于在不同训练阶段之间传递模型权重

支持的权重传递路径:
1. Phase 1 (监督学习) → Phase 2 (SB3 简化策略)
2. Phase 2 (SB3 简化策略) → Phase 3 (SB3 完整策略)
3. Phase 3 (SB3 完整策略) → Phase 4 (CPO 约束优化)
"""

import os
import torch
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class WeightTransferError(Exception):
    """权重传递失败异常"""
    pass


class WeightTransfer:
    """
    权重传递管理器

    职责：
    1. 权重映射和转换
    2. 形状验证
    3. 兼容性检查
    4. 详细的日志记录
    """

    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: 是否打印详细日志
        """
        self.verbose = verbose
        self.transfer_history = []

    def log(self, message: str):
        """打印日志"""
        if self.verbose:
            logger.info(message)
            print(message)

    # ============================================================
    # Phase 1 → Phase 2: 监督学习 → SB3 简化策略
    # ============================================================

    def phase1_to_sb3(
        self,
        sb3_policy: torch.nn.Module,
        phase1_checkpoint: str,
        device: torch.device
    ) -> int:
        """
        将 Phase 1 训练的权重加载到 SB3 策略

        映射关系：
            Phase 1 (TrafficController)      →  Phase 2 (SimpleActorCriticPolicy)
            ────────────────────────────          ───────────────────────────────
            risk_gnn.*                           features_extractor.risk_gnn.*
            world_model.*                        features_extractor.world_model.*

        Args:
            sb3_policy: SB3 策略网络 (SimpleActorCriticPolicy)
            phase1_checkpoint: Phase 1 检查点路径
            device: 目标设备

        Returns:
            成功传递的权重数量

        Raises:
            WeightTransferError: 权重传递失败
        """
        self.log(f"\n{'='*70}")
        self.log(f"🔄 Phase 1 → Phase 2 权重传递")
        self.log(f"{'='*70}")
        self.log(f"   源文件: {phase1_checkpoint}")
        self.log(f"   目标: SB3 SimpleActorCriticPolicy")

        # 1. 加载 Phase 1 检查点
        if not os.path.exists(phase1_checkpoint):
            raise WeightTransferError(f"Phase 1 检查点不存在: {phase1_checkpoint}")

        checkpoint = torch.load(phase1_checkpoint, map_location=device)

        # 提取模型状态字典
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                phase1_state = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                phase1_state = checkpoint['state_dict']
            else:
                phase1_state = checkpoint
        else:
            phase1_state = checkpoint

        self.log(f"   源权重数量: {len(phase1_state)}")

        # 2. 获取 SB3 策略的状态字典
        sb3_state = sb3_policy.state_dict()
        self.log(f"   目标权重数量: {len(sb3_state)}")

        # 3. 定义权重映射规则
        # Phase 1 的权重前缀 → Phase 2 的权重前缀
        weight_mappings = [
            # GNN 层
            ('risk_gnn.', 'features_extractor.risk_gnn.'),
            # World Model 层
            ('world_model.', 'features_extractor.world_model.'),
        ]

        # 4. 执行权重传递
        loaded_count = 0
        skipped_count = 0
        failed_count = 0

        for old_prefix, new_prefix in weight_mappings:
            self.log(f"\n   📦 映射: {old_prefix} → {new_prefix}")

            for phase1_key, phase1_param in phase1_state.items():
                if not phase1_key.startswith(old_prefix):
                    continue

                # 转换键名
                new_key = phase1_key.replace(old_prefix, new_prefix)

                # 检查目标键是否存在
                if new_key not in sb3_state:
                    self.log(f"      ⚠️  跳过: {phase1_key} → {new_key} (目标不存在)")
                    skipped_count += 1
                    continue

                # 检查形状是否匹配
                target_shape = sb3_state[new_key].shape
                source_shape = phase1_param.shape

                if source_shape != target_shape:
                    self.log(f"      ❌ 失败: {phase1_key}")
                    self.log(f"         源形状: {source_shape}")
                    self.log(f"         目标形状: {target_shape}")
                    failed_count += 1
                    continue

                # 传递权重
                sb3_state[new_key] = phase1_param.to(device)
                loaded_count += 1

                if self.verbose:
                    self.log(f"      ✅ {phase1_key} → {new_key}")

        # 5. 加载更新后的状态字典
        sb3_policy.load_state_dict(sb3_state)

        # 6. 记录传递历史
        self.transfer_history.append({
            'source': 'phase1',
            'target': 'phase2_sb3',
            'loaded': loaded_count,
            'skipped': skipped_count,
            'failed': failed_count
        })

        # 7. 打印总结
        self.log(f"\n{'='*70}")
        self.log(f"📊 权重传递总结")
        self.log(f"{'='*70}")
        self.log(f"   ✅ 成功: {loaded_count}")
        self.log(f"   ⚠️  跳过: {skipped_count}")
        self.log(f"   ❌ 失败: {failed_count}")
        self.log(f"{'='*70}\n")

        if loaded_count == 0:
            raise WeightTransferError("没有成功传递任何权重！")

        return loaded_count

    # ============================================================
    # Phase 2 → Phase 3: SB3 简化策略 → SB3 完整策略
    # ============================================================

    def sb3_to_full_policy(
        self,
        full_policy: torch.nn.Module,
        phase2_sb3_model,
        device: torch.device
    ) -> int:
        """
        将 Phase 2 SB3 简化策略的权重加载到 Phase 3 完整策略

        映射关系：
            Phase 2 (SimpleActorCriticPolicy)    →  Phase 3 (FullTrafficActorCriticPolicy)
            ──────────────────────────────          ───────────────────────────────────────
            features_extractor.vehicle_encoder     traffic_controller.risk_gnn (部分)
            features_extractor.global_encoder      traffic_controller.world_model (部分)
            action_net                             traffic_controller.controller
            value_net                               (复用)

        Args:
            full_policy: 完整 SB3 策略 (FullTrafficActorCriticPolicy)
            phase2_sb3_model: Phase 2 的 SB3 PPO 模型
            device: 目标设备

        Returns:
            成功传递的权重数量
        """
        self.log(f"\n{'='*70}")
        self.log(f"🔄 Phase 2 → Phase 3 权重传递")
        self.log(f"{'='*70}")
        self.log(f"   源: SB3 SimpleActorCriticPolicy")
        self.log(f"   目标: SB3 FullTrafficActorCriticPolicy")

        # 1. 获取 Phase 2 策略的状态字典
        phase2_policy = phase2_sb3_model.policy
        phase2_state = phase2_policy.state_dict()
        self.log(f"   源权重数量: {len(phase2_state)}")

        # 2. 获取完整策略的状态字典
        full_state = full_policy.state_dict()
        self.log(f"   目标权重数量: {len(full_state)}")

        # 3. 定义权重映射规则
        # 这里需要根据实际的完整策略架构进行调整
        weight_mappings = self._get_phase2_to_phase3_mappings()

        # 4. 执行权重传递
        loaded_count = 0
        skipped_count = 0
        failed_count = 0

        for pattern, target_prefix in weight_mappings.items():
            self.log(f"\n   📦 映射模式: {pattern}")

            for phase2_key, phase2_param in phase2_state.items():
                if not phase2_key.startswith(pattern):
                    continue

                # 根据模式生成目标键名
                new_key = self._transform_key(phase2_key, pattern, target_prefix)

                if new_key is None:
                    continue

                # 检查目标是否存在
                if new_key not in full_state:
                    self.log(f"      ⚠️  跳过: {phase2_key} → {new_key} (目标不存在)")
                    skipped_count += 1
                    continue

                # 检查形状
                if phase2_param.shape != full_state[new_key].shape:
                    self.log(f"      ❌ 失败: {phase2_key}")
                    self.log(f"         源形状: {phase2_param.shape}")
                    self.log(f"         目标形状: {full_state[new_key].shape}")
                    failed_count += 1
                    continue

                # 传递权重
                full_state[new_key] = phase2_param.to(device)
                loaded_count += 1

                if self.verbose:
                    self.log(f"      ✅ {phase2_key} → {new_key}")

        # 5. 加载更新后的状态字典
        full_policy.load_state_dict(full_state)

        # 6. 记录历史
        self.transfer_history.append({
            'source': 'phase2_sb3',
            'target': 'phase3_full',
            'loaded': loaded_count,
            'skipped': skipped_count,
            'failed': failed_count
        })

        # 7. 打印总结
        self.log(f"\n{'='*70}")
        self.log(f"📊 权重传递总结")
        self.log(f"{'='*70}")
        self.log(f"   ✅ 成功: {loaded_count}")
        self.log(f"   ⚠️  跳过: {skipped_count}")
        self.log(f"   ❌ 失败: {failed_count}")
        self.log(f"{'='*70}\n")

        return loaded_count

    def _get_phase2_to_phase3_mappings(self) -> Dict[str, str]:
        """
        获取 Phase 2 → Phase 3 的权重映射规则

        Returns:
            模式到目标前缀的映射字典
        """
        # 这个映射需要根据实际的完整策略架构来定义
        # 示例映射：
        return {
            # 如果完整策略有兼容的特征提取器
            'features_extractor.': 'traffic_controller.',

            # Actor 和 Critic 网络可能需要特殊处理
            # 'action_net.': 'actor_mean.',
            # 'value_net.': 'critic.',
        }

    def _transform_key(self, key: str, pattern: str, target_prefix: str) -> Optional[str]:
        """
        转换键名

        Args:
            key: 原始键名
            pattern: 匹配模式
            target_prefix: 目标前缀

        Returns:
            转换后的键名，如果不匹配则返回 None
        """
        if key.startswith(pattern):
            suffix = key[len(pattern):]
            return target_prefix + suffix
        return None

    # ============================================================
    # Phase 3 → Phase 4: 完整策略 → CPO 约束优化
    # ============================================================

    def phase3_to_cpo(
        self,
        cpo_model,
        phase3_sb3_model,
        device: torch.device
    ) -> int:
        """
        将 Phase 3 完整策略的权重加载到 Phase 4 CPO 模型

        由于 Phase 3 和 Phase 4 使用相同的策略架构，
        这个传递相对简单，主要是复制策略权重。

        Args:
            cpo_model: Phase 4 的 CPO 模型
            phase3_sb3_model: Phase 3 的 SB3 PPO 模型
            device: 目标设备

        Returns:
            成功传递的权重数量
        """
        self.log(f"\n{'='*70}")
        self.log(f"🔄 Phase 3 → Phase 4 权重传递")
        self.log(f"{'='*70}")
        self.log(f"   源: SB3 PPO (Phase 3)")
        self.log(f"   目标: CPO (Phase 4)")

        # 获取状态字典
        phase3_state = phase3_sb3_model.policy.state_dict()
        cpo_state = cpo_model.policy.state_dict()

        self.log(f"   源权重数量: {len(phase3_state)}")
        self.log(f"   目标权重数量: {len(cpo_state)}")

        # 直接复制所有匹配的权重
        loaded_count = 0
        skipped_count = 0

        for key, param in phase3_state.items():
            if key in cpo_state:
                if param.shape == cpo_state[key].shape:
                    cpo_state[key] = param.to(device)
                    loaded_count += 1
                    if self.verbose:
                        self.log(f"   ✅ {key}")
                else:
                    self.log(f"   ⚠️  跳过: {key} (形状不匹配)")
                    skipped_count += 1
            else:
                skipped_count += 1

        # 加载权重
        cpo_model.policy.load_state_dict(cpo_state)

        # 记录历史
        self.transfer_history.append({
            'source': 'phase3_sb3',
            'target': 'phase4_cpo',
            'loaded': loaded_count,
            'skipped': skipped_count,
            'failed': 0
        })

        # 打印总结
        self.log(f"\n{'='*70}")
        self.log(f"📊 权重传递总结")
        self.log(f"{'='*70}")
        self.log(f"   ✅ 成功: {loaded_count}")
        self.log(f"   ⚠️  跳过: {skipped_count}")
        self.log(f"{'='*70}\n")

        return loaded_count

    # ============================================================
    # 工具方法
    # ============================================================

    def get_transfer_summary(self) -> Dict[str, Any]:
        """
        获取权重传递历史摘要

        Returns:
            包含所有传递记录的字典
        """
        return {
            'total_transfers': len(self.transfer_history),
            'transfers': self.transfer_history
        }

    def save_transfer_report(self, filepath: str):
        """
        保存权重传递报告到文件

        Args:
            filepath: 报告文件路径
        """
        import json
        from datetime import datetime

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_transfer_summary()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.log(f"📄 权重传递报告已保存: {filepath}")


def validate_checkpoint(
    checkpoint_path: str,
    expected_phase: Optional[int] = None
) -> Tuple[bool, str]:
    """
    验证检查点文件的完整性

    Args:
        checkpoint_path: 检查点路径
        expected_phase: 期望的阶段编号（可选）

    Returns:
        (是否有效, 错误消息)
    """
    if not os.path.exists(checkpoint_path):
        return False, f"文件不存在: {checkpoint_path}"

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 检查是否包含模型状态
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # 检查阶段信息
            if expected_phase is not None:
                phase = checkpoint.get('phase', checkpoint.get('training_phase'))
                if phase != expected_phase:
                    return False, f"阶段不匹配: 期望 {expected_phase}, 实际 {phase}"

            # 检查状态字典是否为空
            if len(state_dict) == 0:
                return False, "状态字典为空"

        return True, "检查点有效"

    except Exception as e:
        return False, f"加载失败: {str(e)}"


# 便捷函数
def transfer_phase1_to_sb3(
    sb3_policy: torch.nn.Module,
    phase1_checkpoint: str,
    device: torch.device,
    verbose: bool = True
) -> int:
    """Phase 1 → Phase 2 权重传递的便捷函数"""
    transfer = WeightTransfer(verbose=verbose)
    return transfer.phase1_to_sb3(sb3_policy, phase1_checkpoint, device)


def transfer_sb3_to_full_policy(
    full_policy: torch.nn.Module,
    phase2_sb3_model,
    device: torch.device,
    verbose: bool = True
) -> int:
    """Phase 2 → Phase 3 权重传递的便捷函数"""
    transfer = WeightTransfer(verbose=verbose)
    return transfer.sb3_to_full_policy(full_policy, phase2_sb3_model, device)


def transfer_phase3_to_cpo(
    cpo_model,
    phase3_sb3_model,
    device: torch.device,
    verbose: bool = True
) -> int:
    """Phase 3 → Phase 4 权重传递的便捷函数"""
    transfer = WeightTransfer(verbose=verbose)
    return transfer.phase3_to_cpo(cpo_model, phase3_sb3_model, device)
