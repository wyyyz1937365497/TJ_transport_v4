"""
快速开始脚本
功能：简化版训练流程，用于快速测试
"""

import os
import sys
import json
import time
import torch
import numpy as np

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models import create_model_from_config
from src.env import SumoEnvironment


def main():
    """快速开始函数"""
    print("="*70)
    print("🚀 智能交通控制系统 - 快速开始")
    print("="*70)

    # 简化配置
    config = {
        "device": "cpu",  # 使用CPU避免CUDA问题
        "seed": 42,

        "environment": {
            "sumo_cfg": "仿真环境_初赛_1.0/仿真环境-初赛/sumo.sumocfg",
            "step_length": 0.1,
            "max_steps": 1000,  # 短时间测试
            "control_ratio": 0.25
        },

        "model": {
            "node_dim": 9,
            "edge_dim": 4,
            "gnn_hidden_dim": 64,
            "gnn_output_dim": 256,
            "gnn_layers": 3,
            "gnn_heads": 4,
            "gnn_dropout": 0.1,

            "world_hidden_dim": 128,
            "future_steps": 5,
            "world_dropout": 0.1,

            "controller_hidden_dim": 128,
            "global_dim": 16,
            "action_dim": 2,
            "top_k": 5,
            "controller_dropout": 0.2,

            "interaction_radius": 100.0,
            "max_neighbors": 8,
            "lane_change_distance": 50.0,

            "ttc_threshold": 2.0,
            "thw_threshold": 1.5,
            "max_accel": 2.0,
            "max_decel": -3.0,
            "emergency_decel": -5.0,
            "max_lane_change_speed": 5.0,

            "cost_limit": 0.1,
            "initial_lambda": 1.0,
            "lambda_lr": 0.01
        }
    }

    print(f"\n📋 配置:")
    print(f"   - 设备: {config['device']}")
    print(f"   - 最大步数: {config['environment']['max_steps']}")
    print(f"   - 控制比例: {config['environment']['control_ratio'] * 100}%")

    # 创建模型
    print(f"\n🏗️  创建模型...")
    model = create_model_from_config(config)

    # 创建环境
    print(f"\n🌍 创建SUMO环境...")
    env = SumoEnvironment(config['environment'], use_gui=False)

    # 运行测试episode
    print(f"\n🎮 运行测试episode...")
    observation = env.reset()

    total_reward = 0
    step = 0
    max_steps = config['environment']['max_steps']

    print(f"\n进度:")

    while step < max_steps:
        # 准备batch
        batch = {
            'vehicle_states': observation['vehicle_states'],
            'vehicle_ids': observation['vehicle_ids'],
            'icv_ids': observation['icv_ids'],
            'global_metrics': torch.tensor(
                observation['global_stats'],
                dtype=torch.float32
            ).unsqueeze(0),
            'is_icv': torch.tensor(
                [1 if vid in observation['icv_ids'] else 0 for vid in observation['vehicle_ids']],
                dtype=torch.float32
            ) if observation['vehicle_ids'] else torch.tensor([], dtype=torch.float32)
        }

        # 模型推理
        try:
            output = model(batch)

            # 提取控制动作
            actions = {}
            if output['selected_vehicle_ids']:
                safe_actions = output['safe_actions']
                for i, veh_id in enumerate(output['selected_vehicle_ids']):
                    # 映射加速度
                    accel = safe_actions[i, 0].item()
                    physical_accel = -3.0 + (accel + 1) / 2 * 5.0  # 映射到[-3,2]

                    # 换道
                    lane_change = safe_actions[i, 1].item() > 0.5

                    actions[veh_id] = [physical_accel, lane_change]

        except Exception as e:
            print(f"   ⚠️  推理错误: {e}")
            actions = {}

        # 执行动作
        observation, reward, done, info = env.step(actions)
        total_reward += reward
        step += 1

        # 进度报告
        if step % 100 == 0:
            vehicles = len(observation['vehicle_states'])
            elapsed = step * config['environment']['step_length']
            print(f"   Step {step}/{max_steps} | 车辆数: {vehicles} | 奖励: {total_reward:.2f} | 时间: {elapsed:.1f}s")

        # 检查结束
        if done:
            print(f"\n✅ 仿真自然结束")
            break

    # 最终统计
    print(f"\n📊 测试完成!")
    print(f"   - 总步数: {step}")
    print(f"   - 总奖励: {total_reward:.2f}")
    print(f"   - 平均奖励: {total_reward/step:.3f}")

    env.close()

    print(f"\n✅ 快速开始完成!")
    print(f"\n💡 提示:")
    print(f"   - 使用 'python train.py' 运行完整训练")
    print(f"   - 使用 'python train.py --help' 查看所有选项")


if __name__ == "__main__":
    main()
