#!/usr/bin/env python3
"""
测试ICV比例是否正确应用
"""

import yaml
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.env.gym_wrapper import GymSumoEnv

def test_icv_ratio():
    """测试不同stage的ICV比例"""

    # 加载配置
    with open('configs/competition.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("ICV比例测试")
    print("=" * 80)

    # 测试不同stage
    stages = [1, 2, 5]

    for stage in stages:
        print(f"\n--- Stage {stage} ---")

        # 获取该stage的配置
        curriculum_config = config.get('training', {}).get('curriculum', {})
        levels_config = curriculum_config.get('levels', [])

        if stage <= len(levels_config):
            level_config = levels_config[stage - 1]
            max_vehicles = level_config['max_vehicles']
            icv_ratio = level_config['icv_ratio']

            # 更新环境配置
            env_config = config.get('environment', {}).copy()
            env_config['max_vehicles'] = max_vehicles
            env_config['icv_ratio'] = icv_ratio  # ✅ 关键：设置icv_ratio

            expected_icv_count = int(max_vehicles * icv_ratio)

            print(f"Max vehicles: {max_vehicles}")
            print(f"ICV ratio: {icv_ratio}")
            print(f"Expected ICV count: {expected_icv_count}")

            # 创建环境
            try:
                env = GymSumoEnv(config=env_config)
                obs, info = env.reset()

                # 运行几步让SUMO加载车辆
                for step in range(50):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    if done:
                        obs, info = env.reset()

                # 从观测中提取is_icv标志
                # 观测格式：[max_vehicles * 9 + 32 + 1]
                vehicle_features = obs[:max_vehicles * 9].reshape(max_vehicles, 9)
                is_icv_flags = vehicle_features[:, 8]  # 第9个特征是is_icv

                # 统计实际车辆数（s坐标非零的车辆）
                actual_vehicle_count = (vehicle_features[:, 0].abs() > 1e-6).sum()
                actual_icv_count = is_icv_flags.sum()
                actual_icv_ratio = actual_icv_count / max(actual_vehicle_count, 1) if actual_vehicle_count > 0 else 0

                print(f"Actual vehicles: {actual_vehicle_count:.0f} / {max_vehicles}")
                print(f"Actual ICV count: {actual_icv_count:.0f}")
                print(f"Actual ICV ratio: {actual_icv_ratio:.3f}")
                print(f"  MATCH: {'✅' if abs(actual_icv_ratio - icv_ratio) < 0.15 else '❌ (允许15%误差)'}")

                env.close()
            except Exception as e:
                print(f"  ❌ ERROR: {e}")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    test_icv_ratio()
