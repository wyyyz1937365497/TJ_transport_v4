"""
测试检查点管理器功能

使用方法:
    python test_checkpoint_manager.py

功能:
    1. 显示训练状态
    2. 检查各阶段是否完成
    3. 获取下一个未完成的阶段
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.training.checkpoint_manager import PhaseCheckpointManager


def main():
    # 创建检查点管理器
    checkpoint_dir = 'checkpoints/competition'
    manager = PhaseCheckpointManager(checkpoint_dir)

    # 显示训练状态
    print("\n" + "="*80)
    print("检查点管理器测试")
    print("="*80)

    manager.print_training_status()

    # 获取训练状态
    status = manager.get_training_status()

    print("\n" + "="*80)
    print("详细状态信息")
    print("="*80)

    for phase_key, phase_info in status['phases'].items():
        print(f"\n{phase_key.upper()}:")
        print(f"  名称: {phase_info['name']}")
        print(f"  完成状态: {phase_info['completed']}")

        if phase_info['completed']:
            print(f"  检查点: {phase_info['checkpoint_path']}")
            if 'timestamp' in phase_info:
                print(f"  时间: {phase_info['timestamp']}")
            if 'metrics' in phase_info:
                print(f"  指标: {phase_info['metrics']}")

        print(f"  前置要求: {phase_info['required_for']}")

    # 获取下一个阶段
    print("\n" + "="*80)
    print("训练进度")
    print("="*80)

    next_phase = manager.get_next_phase()
    if next_phase:
        print(f"下一个阶段: {next_phase}")
        phase_info = manager.PHASES[next_phase]
        print(f"  名称: {phase_info['name']}")
        print(f"  检查点: {phase_info['checkpoint_name']}")
    else:
        print("所有阶段均已完成！")

    # 检查特定阶段
    print("\n" + "="*80)
    print("阶段检查")
    print("="*80)

    for phase in ['phase1', 'phase2', 'phase3', 'phase4']:
        can_skip, checkpoint_path = manager.can_skip_phase(phase)
        status_icon = "✅" if can_skip else "⏳"
        print(f"{status_icon} {phase.upper()}: ", end="")
        if can_skip:
            print(f"已完成 ({checkpoint_path})")
        else:
            print(f"未完成")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
