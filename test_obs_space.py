#!/usr/bin/env python3
"""
测试观测空间维度是否正确
"""

import yaml
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.env.gym_wrapper import GymSumoEnv
from src.constants import MAX_VEHICLES, FEATURES_PER_VEHICLE

def test_obs_space():
    """测试观测空间维度"""

    # 加载配置
    with open('configs/competition.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("观测空间维度测试")
    print("=" * 80)

    # 测试不同stage的配置
    stages = [1, 2, 3, 4, 5]

    for stage in stages:
        print(f"\n--- Stage {stage} ---")

        # 获取该stage的配置
        curriculum_config = config.get('training', {}).get('curriculum', {})
        levels_config = curriculum_config.get('levels', [])

        if stage <= len(levels_config):
            level_config = levels_config[stage - 1]
            max_vehicles = level_config['max_vehicles']

            # 更新环境配置
            env_config = config.get('environment', {}).copy()
            env_config['max_vehicles'] = max_vehicles

            print(f"Max vehicles: {max_vehicles}")

            # 创建环境
            try:
                env = GymSumoEnv(config=env_config)

                # 计算期望的观测维度
                expected_obs_dim = max_vehicles * FEATURES_PER_VEHICLE + 32 + 1

                print(f"  期望观测维度: {expected_obs_dim}")
                print(f"  实际观测维度: {env.observation_space.shape[0]}")
                print(f"  MATCH: {'✅' if expected_obs_dim == env.observation_space.shape[0] else '❌'}")

                # 动作空间
                expected_action_dim = max_vehicles * 2
                actual_action_dim = env.action_space.shape[0]
                print(f"  期望动作维度: {expected_action_dim}")
                print(f"  实际动作维度: {actual_action_dim}")
                print(f"  MATCH: {'✅' if expected_action_dim == actual_action_dim else '❌'}")

            except Exception as e:
                print(f"  ❌ ERROR: {e}")

    print("\n" + "=" * 80)
    print(f"MAX_VEHICLES constant: {MAX_VEHICLES}")
    print(f"FEATURES_PER_VEHICLE constant: {FEATURES_PER_VEHICLE}")
    print("=" * 80)

if __name__ == '__main__':
    test_obs_space()
