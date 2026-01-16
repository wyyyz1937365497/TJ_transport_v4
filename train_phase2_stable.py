"""
Phase 2 训练脚本 - 稳定版本（融合 Multi-Stage Training）

采用以下策略避免 Windows SubprocVecEnv 的 BrokenPipeError：
1. 默认使用固定的比赛环境配置（Level 5）
2. 支持多阶段课程学习（每个阶段独立运行）
3. 可选启用动态课程切换（警告：Windows 上不稳定）

使用方法：
    # 方案A：直接在比赛环境训练（推荐，最快）
    python train_phase2_stable.py

    # 方案B：多阶段渐进训练（推荐，更稳定）
    python train_phase2_stable.py --stage 1
    python train_phase2_stable.py --stage 2 --prev-checkpoint checkpoints/competition/curriculum/level1/ppo.zip
    python train_phase2_stable.py --stage 3 --prev-checkpoint checkpoints/competition/curriculum/level2/ppo.zip
    python train_phase2_stable.py --stage 4 --prev-checkpoint checkpoints/competition/curriculum/level3/ppo.zip
    python train_phase2_stable.py --stage 5 --prev-checkpoint checkpoints/competition/curriculum/level4/ppo.zip

    # 使用 Phase 1 权重初始化
    python train_phase2_stable.py --phase1-checkpoint checkpoints/competition/phase1/world_model_final.pth

    # 实验性：启用动态课程切换（不推荐）
    python train_phase2_stable.py --enable-curriculum
"""

import os
import sys
import argparse
import yaml
import torch
import time
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.train_enhancements import EnhancedTrainingManager
from src.models.ideal_policy_v4 import create_ideal_traffic_policy_v4
from src.env.vec_env import create_parallel_envs
from src.utils.helpers import get_device
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


# 课程级别配置（与 CurriculumManager 中的定义一致）
# 训练步数分配：前4个阶段各占6.3%，最后阶段（比赛环境）占剩余74.8%
CURRICULUM_LEVELS = [
    {
        'level': 1,
        'name': '基础场景',
        'description': '低流量、高渗透率、无扰动',
        'max_vehicles': 10,
        'inflow_rate': 800,
        'icv_ratio': 0.3,
        'disturbance_level': 0.0,
        'total_timesteps': 126000,  # 6.3% = 126k步
        'checkpoint': 'checkpoints/competition/curriculum/level1/ppo.zip'
    },
    {
        'level': 2,
        'name': '中等流量',
        'description': '中等流量、适中渗透率、弱扰动',
        'max_vehicles': 15,
        'inflow_rate': 1200,
        'icv_ratio': 0.25,
        'disturbance_level': 0.2,
        'total_timesteps': 126000,  # 6.3% = 126k步
        'checkpoint': 'checkpoints/competition/curriculum/level2/ppo.zip'
    },
    {
        'level': 3,
        'name': '高流量场景',
        'description': '高流量、标准渗透率、中等扰动',
        'max_vehicles': 20,
        'inflow_rate': 1800,
        'icv_ratio': 0.25,
        'disturbance_level': 0.4,
        'total_timesteps': 126000,  # 6.3% = 126k步
        'checkpoint': 'checkpoints/competition/curriculum/level3/ppo.zip'
    },
    {
        'level': 4,
        'name': '极端场景',
        'description': '极高流量、低渗透率、强扰动',
        'max_vehicles': 32,
        'inflow_rate': 2400,
        'icv_ratio': 0.15,
        'disturbance_level': 0.7,
        'total_timesteps': 126000,  # 6.3% = 126k步
        'checkpoint': 'checkpoints/competition/curriculum/level4/ppo.zip'
    },
    {
        'level': 5,
        'name': '赛题场景',
        'description': '真实比赛条件、突发扰动',
        'max_vehicles': 32,
        'inflow_rate': 2000,
        'icv_ratio': 0.25,
        'disturbance_level': 0.5,
        'total_timesteps': 1496000,  # 74.8% = 1,496k步（比赛环境，充分训练）
        'checkpoint': 'checkpoints/competition/phase2/shielded_ppo.zip'  # 默认路径
    },
]


class EnhancedTrainingCallback(BaseCallback):
    """增强训练回调 - 进度跟踪和时间统计"""

    def __init__(self, enhanced_manager, total_timesteps, verbose=1):
        super().__init__(verbose)
        self.enhanced_manager = enhanced_manager
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.last_print_time = None
        self.print_interval = 60  # 每60秒打印一次

    def _on_training_start(self):
        """训练开始时记录时间"""
        self.start_time = time.time()
        self.last_print_time = self.start_time
        print(f"\n[TRAIN] Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def _on_step(self):
        """每步调用"""
        if self.verbose == 0:
            return True

        # 定期打印进度
        current_time = time.time()
        if current_time - self.last_print_time >= self.print_interval:
            elapsed = current_time - self.start_time
            progress = self.num_timesteps / self.total_timesteps * 100

            # 计算ETA
            if progress > 0:
                remaining_time = elapsed * (100 - progress) / progress
            else:
                remaining_time = 0

            # 格式化时间
            elapsed_str = self._format_time(elapsed)
            remaining_str = self._format_time(remaining_time)

            # 打印进度
            print(f"[PROGRESS] {self.num_timesteps:,}/{self.total_timesteps:,} steps ({progress:.1f}%) | "
                  f"Elapsed: {elapsed_str} | ETA: {remaining_str}", flush=True)

            self.last_print_time = current_time

        return True

    def _on_training_end(self):
        """训练结束时处理"""
        print()  # 换行
        print(f"\n[DONE] Training completed at step {self.num_timesteps:,}")
        elapsed = time.time() - self.start_time
        print(f"[TIME] Total training time: {self._format_time(elapsed)}")

    def _format_time(self, seconds):
        """格式化时间显示"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


def load_config(config_path: str = 'configs/competition.yaml') -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_phase2(
    config: Dict[str, Any],
    phase1_checkpoint: Optional[str] = None,
    enable_curriculum: bool = False,
    stage: Optional[int] = None,
    prev_checkpoint: Optional[str] = None
) -> str:
    """
    Phase 2 PPO 训练

    Args:
        config: 训练配置
        phase1_checkpoint: Phase 1 检查点路径（可选）
        enable_curriculum: 是否启用动态课程切换（警告：Windows 上不稳定）
        stage: 课程阶段（1-5），None 表示使用 Level 5（比赛环境）
        prev_checkpoint: 前一阶段的检查点路径（用于多阶段训练）

    Returns:
        训练完成的模型路径
    """
    print("\n" + "="*80)
    print("[PHASE 2] PPO Training - Stable Version (Multi-Stage Supported)")
    print("="*80)

    # 确定训练模式和配置
    if stage is not None:
        # 多阶段训练模式
        if stage < 1 or stage > 5:
            print(f"\n[ERROR] Invalid stage: {stage}. Must be between 1 and 5.")
            return None

        stage_config = CURRICULUM_LEVELS[stage - 1]

        print(f"\n[MODE] Multi-Stage Training - Stage {stage}")
        print(f"[INFO] Level: {stage_config['name']}")
        print(f"[INFO] Description: {stage_config['description']}")
        print(f"[INFO] Parameters:")
        print(f"  - max_vehicles: {stage_config['max_vehicles']}")
        print(f"  - inflow_rate: {stage_config['inflow_rate']}")
        print(f"  - icv_ratio: {stage_config['icv_ratio']}")
        print(f"  - disturbance_level: {stage_config['disturbance_level']}")
        print(f"  - total_timesteps: {stage_config['total_timesteps']:,}")

        if prev_checkpoint:
            print(f"\n[LOAD] Will load weights from: {prev_checkpoint}")
        elif phase1_checkpoint and stage == 1:
            print(f"\n[LOAD] Will load Phase 1 weights from: {phase1_checkpoint}")

        enable_curriculum = False  # 多阶段模式禁用动态切换
    elif enable_curriculum:
        # 动态课程切换模式（不推荐）
        print("\n[WARNING] Dynamic curriculum learning is ENABLED")
        print("[WARNING] This may cause BrokenPipeError on Windows!")
        print("[WARNING] Consider using --stage 1..5 for multi-stage training instead")
        print("[WARNING] Or run without --enable-curriculum to use fixed competition environment\n")

        # 临时启用课程学习
        if 'training' not in config:
            config['training'] = {}
        if 'curriculum' not in config['training']:
            config['training']['curriculum'] = {}
        config['training']['curriculum']['enabled'] = True
    else:
        # 固定比赛环境模式（默认，推荐）
        print("\n[MODE] Fixed Competition Environment (Level 5)")
        print("[INFO] This is the STABLE option for Windows")
        print("[INFO] Dynamic curriculum switching is DISABLED")
        print("[INFO] Use --stage 1..5 for multi-stage training")
        print("[INFO] Use --enable-curriculum for dynamic switching (experimental)\n")

        # 禁用课程学习
        if 'training' not in config:
            config['training'] = {}
        if 'curriculum' not in config['training']:
            config['training']['curriculum'] = {}
        config['training']['curriculum']['enabled'] = False

        # 使用 Level 5 配置
        stage = 5
        stage_config = CURRICULUM_LEVELS[4]
        prev_checkpoint = None  # 固定环境模式不使用前阶段权重

    # 初始化增强训练管理器
    enhanced_manager = EnhancedTrainingManager(config)

    # 获取设备
    device = get_device()

    # 配置 - 基础配置
    phase2_config = config.get('training', {}).get('phase2', {})

    # 如果是多阶段模式，使用阶段配置覆盖基础配置
    if stage is not None:
        total_timesteps = stage_config['total_timesteps']
        model_path = stage_config['checkpoint']

        # 环境配置
        env_config = config.get('environment', {}).copy()
        env_config['max_vehicles'] = stage_config['max_vehicles']
        env_config['inflow_rate'] = stage_config['inflow_rate']
        env_config['icv_ratio'] = stage_config['icv_ratio']
        env_config['disturbance_level'] = stage_config['disturbance_level']
    else:
        # 固定环境模式或动态课程模式
        total_timesteps = phase2_config.get('total_timesteps', 2000000)
        model_path = phase2_config.get('phase2_model_path', 'checkpoints/competition/phase2/shielded_ppo.zip')
        env_config = config.get('environment', {}).copy()

        # 如果不使用课程学习，直接使用比赛环境配置（Level 5）
        if not enable_curriculum:
            env_config['max_vehicles'] = 32
            env_config['inflow_rate'] = 2000
            env_config['icv_ratio'] = 0.25
            env_config['disturbance_level'] = 0.5
        else:
            # 应用课程学习的当前配置
            if enhanced_manager.curriculum:
                env_config = enhanced_manager.get_curriculum_env_config(env_config)
                current_level = enhanced_manager.curriculum.get_current_difficulty()
                print(f"\n[CURRICULUM] Starting at Level {current_level.level}: {current_level.name}")
                print(f"  Config: {current_level.max_vehicles} vehicles, {current_level.inflow_rate} flow")

    # 检查点目录
    checkpoint_dir = os.path.dirname(model_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_envs = phase2_config.get('num_envs', 24)
    n_steps = phase2_config.get('n_steps', 16384)

    print(f"\n[PHASE 2] Configuration:")
    print(f"  Total Timesteps: {total_timesteps:,}")
    print(f"  Parallel Envs: {num_envs}")
    print(f"  Steps per Rollout: {n_steps}")
    print(f"  Batch Size: {phase2_config.get('batch_size', 32768)}")
    print(f"  Learning Rate: {phase2_config.get('learning_rate', 3e-4)}")
    print(f"  Checkpoint: {model_path}")
    print(f"  Environment: max_vehicles={env_config['max_vehicles']}, inflow_rate={env_config['inflow_rate']}")

    # 明确传递device参数
    device_str = str(device)
    print(f"[ENV] Using device: {device_str}")

    # 创建环境
    print(f"\n[ENV] Creating {num_envs} parallel environments...")
    vec_env_wrapper = create_parallel_envs(
        config=env_config,
        num_envs=num_envs,
        seed=config.get('seed', 42),
        device=device_str
    )

    vec_env = vec_env_wrapper.vec_env
    print(f"[OK] Environments created\n")

    # 创建策略
    print("[MODEL] Creating policy network...")
    policy_class = create_ideal_traffic_policy_v4(config)

    # 确保learning_rate为float
    learning_rate = float(phase2_config.get('learning_rate', 3e-4))

    model = PPO(
        policy_class,
        vec_env,
        verbose=1,
        tensorboard_log=config.get('paths', {}).get('log_dir', 'logs') + '/phase2',
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=phase2_config.get('batch_size', 32768),
        n_epochs=phase2_config.get('update_epochs', 5),
        gamma=phase2_config.get('gamma', 0.99),
        gae_lambda=phase2_config.get('gae_lambda', 0.95),
        clip_range=phase2_config.get('clip_epsilon', 0.2),
        ent_coef=phase2_config.get('entropy_coef', 0.01),
        vf_coef=phase2_config.get('value_loss_coef', 0.5),
        max_grad_norm=phase2_config.get('max_grad_norm', 0.5),
        seed=config.get('seed', 42),
        device=device_str
    )

    # 加载权重
    # 优先级：prev_checkpoint > phase1_checkpoint
    if prev_checkpoint and os.path.exists(prev_checkpoint):
        # 加载前阶段的完整模型权重
        print(f"\n[LOAD] Loading weights from previous stage: {prev_checkpoint}")
        try:
            model.set_parameters(prev_checkpoint)
            print(f"[OK] Weights loaded successfully from previous stage")
        except Exception as e:
            print(f"[WARN] Failed to load weights: {e}")
            print(f"[INFO] Training from scratch instead")
    elif phase1_checkpoint and os.path.exists(phase1_checkpoint):
        # 加载Phase 1的感知层和预测层权重
        print(f"\n[LOAD] Loading Phase 1 weights from: {phase1_checkpoint}")
        try:
            checkpoint = torch.load(phase1_checkpoint, map_location=device)

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            model_dict = model.policy.state_dict()
            loaded = 0

            for key in state_dict.keys():
                if key.startswith('perception_layer.') or key.startswith('prediction_layer.'):
                    if key in model_dict and state_dict[key].shape == model_dict[key].shape:
                        model_dict[key] = state_dict[key]
                        loaded += 1

            print(f"[OK] Loaded {loaded} weights from Phase 1")
        except Exception as e:
            print(f"[WARN] Failed to load Phase 1 weights: {e}")
            print(f"[INFO] Training from scratch instead")
    elif prev_checkpoint or phase1_checkpoint:
        checkpoint_path = prev_checkpoint if prev_checkpoint else phase1_checkpoint
        print(f"\n[WARN] Checkpoint file not found: {checkpoint_path}")
        print(f"[INFO] Training from scratch")

    # 冻结感知层和预测层
    print(f"\n[FREEZE] Freezing perception and prediction layers...")
    if hasattr(model.policy, 'freeze_perception'):
        model.policy.freeze_perception()
    if hasattr(model.policy, 'freeze_prediction'):
        model.policy.freeze_prediction()
    print(f"[OK] Layers frozen")

    # 创建回调
    callbacks = []

    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # 每50k步保存
        save_path=checkpoint_dir,
        name_prefix="checkpoint",
        save_replay_buffer=False
    )
    callbacks.append(checkpoint_callback)

    # 增强训练回调（进度跟踪）
    enhanced_callback = EnhancedTrainingCallback(
        enhanced_manager=enhanced_manager,
        total_timesteps=total_timesteps,
        verbose=1
    )
    callbacks.append(enhanced_callback)

    # 课程学习回调（仅在启用时添加）
    if enable_curriculum and enhanced_manager.curriculum:
        from train_phase2 import CurriculumLevelCallback

        curriculum_config = config.get('training', {}).get('curriculum', {})

        curriculum_callback = CurriculumLevelCallback(
            config=config,
            enhanced_manager=enhanced_manager,
            num_envs=num_envs,
            check_frequency=curriculum_config.get('check_frequency', 50000),
            verbose=curriculum_config.get('verbose', 1)
        )
        callbacks.append(curriculum_callback)

        print(f"\n[CURRICULUM] Dynamic level switching ENABLED (experimental)")
        print(f"  Check frequency: every {curriculum_config.get('check_frequency', 50000):,} steps")
        print(f"  Min episodes per level: {curriculum_config.get('min_episodes_per_level', 5)}")
    else:
        print(f"\n[INFO] Curriculum learning: DISABLED")
        print(f"[INFO] Using fixed competition environment (recommended for Windows)")

    # 开始训练
    print("\n" + "="*80)
    print("[TRAIN] Starting PPO training...")
    print("="*80)
    print(f"[INFO] Total timesteps: {total_timesteps:,}")
    print(f"[INFO] Parallel environments: {num_envs}")
    print(f"[INFO] Steps per rollout: {n_steps}")

    if enable_curriculum:
        print(f"[WARNING] Dynamic curriculum: ENABLED (may cause issues on Windows)")
    else:
        print(f"[INFO] Fixed competition environment: ENABLED (stable)")

    print()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True  # 使用内置进度条
        )

        # 保存最终模型
        print(f"\n[SAVE] Saving final model...")
        model.save(model_path)
        print(f"[OK] Model saved: {model_path}")

        # 显示最终统计
        print(f"\n" + "="*80)
        print(f"[DONE] Phase 2 Training Complete!")
        print(f"="*80)
        print(f"  Model: {model_path}")

        if stage is not None:
            print(f"  Stage: {stage} - {stage_config['name']}")
            print(f"  Config: {stage_config['max_vehicles']} vehicles, {stage_config['inflow_rate']} flow")

            # 提示下一步
            if stage < 5:
                next_stage = stage + 1
                print(f"\n[INFO] To continue to Stage {next_stage}, run:")
                print(f"  python train_phase2_stable.py --stage {next_stage} --prev-checkpoint {model_path}")
            else:
                print(f"\n[INFO] All curriculum stages completed!")
                print(f"[INFO] Final model: {model_path}")
        elif enable_curriculum and enhanced_manager.curriculum:
            final_level = enhanced_manager.curriculum.get_current_difficulty()
            print(f"  Final Level: {final_level.level} - {final_level.name}")
            print(f"  Config: {final_level.max_vehicles} vehicles, {final_level.inflow_rate} flow")
        else:
            print(f"  Environment: Competition (Level 5)")
            print(f"  Config: 32 vehicles, 2000 flow")

        print(f"="*80 + "\n")

    except BrokenPipeError as e:
        print(f"\n{'='*80}")
        print(f"[ERROR] BrokenPipeError occurred!")
        print(f"="*80)
        print(f"\nThis is a known issue on Windows with dynamic curriculum switching.")
        print(f"\nRecommended solutions:")
        print(f"\n1. Use fixed competition environment (RECOMMENDED, fastest):")
        print(f"   python train_phase2_stable.py")
        print(f"\n2. Use multi-stage training (RECOMMENDED, progressive learning):")
        print(f"   python train_phase2_stable.py --stage 1")
        print(f"   python train_phase2_stable.py --stage 2 --prev-checkpoint checkpoints/competition/curriculum/level1/ppo.zip")
        print(f"   python train_phase2_stable.py --stage 3 --prev-checkpoint checkpoints/competition/curriculum/level2/ppo.zip")
        print(f"   python train_phase2_stable.py --stage 4 --prev-checkpoint checkpoints/competition/curriculum/level3/ppo.zip")
        print(f"   python train_phase2_stable.py --stage 5 --prev-checkpoint checkpoints/competition/curriculum/level4/ppo.zip")
        print(f"\n3. If you really want dynamic curriculum (not recommended):")
        print(f"   Run on Linux, or accept that training may fail")
        print(f"\n{'='*80}\n")
        raise

    # 关闭环境
    vec_env.close()

    return model_path


def main():
    parser = argparse.ArgumentParser(
        description='Phase 2 PPO 训练 - 稳定版本（融合 Multi-Stage Training）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
训练模式：

  方案A：固定环境训练（推荐，最快）
    python train_phase2_stable.py
    - 直接在比赛环境（Level 5）训练
    - 8小时完成训练
    - 最稳定，无 BrokenPipeError

  方案B：多阶段渐进训练（推荐，更稳定）
    python train_phase2_stable.py --stage 1
    python train_phase2_stable.py --stage 2 --prev-checkpoint checkpoints/competition/curriculum/level1/ppo.zip
    python train_phase2_stable.py --stage 3 --prev-checkpoint checkpoints/competition/curriculum/level2/ppo.zip
    python train_phase2_stable.py --stage 4 --prev-checkpoint checkpoints/competition/curriculum/level3/ppo.zip
    python train_phase2_stable.py --stage 5 --prev-checkpoint checkpoints/competition/curriculum/level4/ppo.zip
    - 渐进式学习（从简单到复杂）
    - 每个阶段独立运行，无管道问题
    - 总计约30小时

  组合选项：
    # 使用 Phase 1 权重初始化阶段1
    python train_phase2_stable.py --stage 1 --phase1-checkpoint checkpoints/competition/phase1/world_model_final.pth

    # 固定环境 + Phase 1 权重
    python train_phase2_stable.py --phase1-checkpoint checkpoints/competition/phase1/world_model_final.pth

  实验性：动态课程切换（不推荐）
    python train_phase2_stable.py --enable-curriculum
    - Windows 上可能导致 BrokenPipeError
    - 推荐使用方案B代替
        """
    )

    parser.add_argument(
        '--stage',
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5],
        help='课程阶段（1-5），用于多阶段训练。不指定则使用固定比赛环境（Level 5）'
    )

    parser.add_argument(
        '--prev-checkpoint',
        type=str,
        default=None,
        help='前阶段的检查点路径（用于多阶段训练，从阶段2开始需要）'
    )

    parser.add_argument(
        '--phase1-checkpoint',
        type=str,
        default=None,
        help='Phase 1 检查点路径（用于初始化感知层和预测层）'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/competition.yaml',
        help='配置文件路径（默认: configs/competition.yaml）'
    )

    parser.add_argument(
        '--enable-curriculum',
        action='store_true',
        help='启用动态课程切换（警告：Windows 上不稳定，可能导致 BrokenPipeError）'
    )

    args = parser.parse_args()

    # 参数验证
    if args.prev_checkpoint and args.stage is None:
        print("[ERROR] --prev-checkpoint 只能与 --stage 一起使用")
        print("[INFO] 如果不使用多阶段训练，请删除 --prev-checkpoint 参数")
        return 1

    if args.enable_curriculum and args.stage is not None:
        print("[ERROR] --enable-curriculum 和 --stage 不能同时使用")
        print("[INFO] --stage 已经是分阶段训练，不需要动态切换")
        return 1

    # 加载配置
    print(f"\n[CONFIG] Loading configuration from: {args.config}")
    config = load_config(args.config)
    print(f"[OK] Configuration loaded\n")

    # 开始训练
    try:
        checkpoint = train_phase2(
            config,
            phase1_checkpoint=args.phase1_checkpoint,
            enable_curriculum=args.enable_curriculum,
            stage=args.stage,
            prev_checkpoint=args.prev_checkpoint
        )

        if checkpoint is None:
            print(f"\n[ERROR] Training failed")
            return 1

        print(f"\n[SUCCESS] Training completed successfully!")
        print(f"Model saved to: {checkpoint}")
        return 0
    except KeyboardInterrupt:
        print(f"\n[INFO] Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
