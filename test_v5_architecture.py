"""
轻量级架构 v5.0 测试脚本

测试内容:
1. 模型创建和前向传播
2. Top-K车辆选择机制
3. OCR奖励计算
4. 稀疏控制器集成
"""

import sys
import torch
import numpy as np
import yaml
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.models.v5_lightweight import create_lightweight_policy_v5
from src.training.ocr_rewards import create_ocr_reward_calculator, BaselineStatisticsCollector
from src.env.sparse_controller import create_sparse_controller


def test_model_creation():
    """测试1: 模型创建和前向传播"""
    print("\n" + "=" * 70)
    print("测试1: 模型创建和前向传播")
    print("=" * 70)

    # 配置
    config = {
        'model': {
            'gnn': {
                'hidden_dim': 64,
                'output_dim': 64,
                'num_layers': 3,
                'dropout': 0.1,
                'top_k_ratio': 0.05
            }
        }
    }

    # 创建模型
    obs_dim = 321  # 32*9 + 32 + 1
    action_dim = 2

    policy = create_lightweight_policy_v5(obs_dim, action_dim, config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    policy = policy.to(device)

    print(f"✅ 模型创建成功")
    print(f"   设备: {device}")
    print(f"   参数量: {sum(p.numel() for p in policy.parameters()):,}")

    # 测试前向传播
    batch_size = 4
    obs = torch.randn(batch_size, obs_dim).to(device)

    with torch.no_grad():
        actions, values, log_probs = policy(obs, deterministic=False)

    print(f"✅ 前向传播成功")
    print(f"   输入shape: {obs.shape}")
    print(f"   输出shape: actions={actions.shape}, values={values.shape}, log_probs={log_probs.shape}")

    # 测试evaluate_actions
    with torch.no_grad():
        values2, log_probs2, entropy = policy.evaluate_actions(obs, actions)

    print(f"✅ evaluate_actions成功")
    print(f"   熵: {entropy.item():.4f}")

    return policy


def test_top_k_selection(policy):
    """测试2: Top-K车辆选择机制"""
    print("\n" + "=" * 70)
    print("测试2: Top-K车辆选择机制")
    print("=" * 70)

    device = next(policy.parameters()).device

    # 创建模拟观测（需要合理设置num_vehicles）
    obs_dim = 321
    obs = torch.randn(1, obs_dim).to(device)

    # ⭐ 关键修复：设置合理的num_vehicles值（位于观测的最后一个位置）
    # obs格式：[max_vehicles*9 (vehicles) + 32 (global) + 1 (num)]
    # 最后一个元素应该是实际车辆数
    num_vehicles_real = 30  # 假设有30辆车
    obs[0, -1] = num_vehicles_real  # 设置num_vehicles

    # 选择车辆
    with torch.no_grad():
        selected_indices, info = policy.select_vehicles(obs[0].cpu().numpy())

    print(f"✅ Top-K选择成功")
    print(f"   选中的车辆数: {len(selected_indices)}")
    print(f"   车辆索引: {selected_indices}")
    print(f"   K值: {info['k']}")
    print(f"   总车辆数: {info['num_vehicles']}")

    # 验证Top-K比例
    k_ratio = len(selected_indices) / info['num_vehicles']
    print(f"   实际Top-K比例: {k_ratio:.1%}")
    print(f"   目标Top-K比例: 5.0%")

    assert k_ratio <= 0.10, f"Top-K比例过高: {k_ratio:.1%}"
    print(f"✅ Top-K比例验证通过")


def test_ocr_reward_calculator():
    """测试3: OCR奖励计算"""
    print("\n" + "=" * 70)
    print("测试3: OCR奖励计算")
    print("=" * 70)

    # 创建OCR奖励计算器
    calculator = create_ocr_reward_calculator(
        baseline_stats=None,  # 没有基准
        k_penalty=0.1,
        w_efficiency=0.7,
        w_stability=0.3,
    )

    print(f"✅ OCR奖励计算器创建成功")

    # 模拟episode数据
    calculator.reset()

    # 模拟10步的更新
    for step in range(10):
        # 模拟车辆信息
        num_vehicles = np.random.randint(20, 50)
        vehicle_info = []

        for i in range(num_vehicles):
            arrived = np.random.random() < 0.1  # 10%概率到达
            vehicle_info.append({
                'id': f'veh_{i}',
                'arrived': arrived,
                'traveled': np.random.uniform(100, 1000),
                'total': np.random.uniform(1000, 2000),
                'speed': np.random.uniform(5, 25),
                'accel': np.random.uniform(-2, 2)
            })

        # 模拟干预
        accel_commands = np.random.randint(0, 5)
        lane_changes = np.random.randint(0, 2)
        num_controlled = np.random.randint(3, 10)

        # 更新
        reward = calculator.update(vehicle_info, accel_commands, lane_changes, num_controlled)

        if step == 0:
            print(f"   第1步奖励: {reward:.4f}")

    # 计算episode得分
    scores = calculator.compute_episode_score()

    print(f"✅ Episode统计:")
    print(f"   OCR: {scores['ocr']:.4f}")
    print(f"   S_efficiency: {scores['s_efficiency']:.2f}")
    print(f"   S_stability: {scores['s_stability']:.2f}")
    print(f"   C_int: {scores['c_int']:.4f}")
    print(f"   P_intervention: {scores['p_intervention']:.4f}")
    print(f"   S_total: {scores['s_total']:.2f}")


def test_sparse_controller():
    """测试4: 稀疏控制器"""
    print("\n" + "=" * 70)
    print("测试4: 稀疏控制器")
    print("=" * 70)

    # 测试基于规则的控制器
    controller = create_sparse_controller(
        controller_type='rule',
        decision_interval=10,
        top_k_ratio=0.05,
    )

    print(f"✅ 稀疏控制器创建成功")

    # 测试决策逻辑
    for step in [0, 5, 10, 15, 20]:
        should_decide = controller.should_make_decision(step)
        print(f"   Step {step}: {'决策' if should_decide else '跳过'}")

    # 测试车辆选择
    num_vehicles = 100
    vehicle_states = np.random.randn(num_vehicles, 9)
    vehicle_ids = [f'veh_{i}' for i in range(num_vehicles)]

    selected_ids = controller.select_critical_vehicles(vehicle_states, vehicle_ids)

    print(f"✅ 车辆选择成功:")
    print(f"   总车辆数: {num_vehicles}")
    print(f"   选中车辆数: {len(selected_ids)}")
    print(f"   选中车辆ID: {selected_ids[:5]}...")


def test_integration():
    """测试5: 端到端集成测试"""
    print("\n" + "=" * 70)
    print("测试5: 端到端集成测试")
    print("=" * 70)

    # 创建组件
    config = {
        'model': {
            'gnn': {
                'hidden_dim': 64,
                'output_dim': 64,
                'num_layers': 3,
                'dropout': 0.1,
                'top_k_ratio': 0.05
            }
        }
    }

    policy = create_lightweight_policy_v5(321, 2, config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    policy = policy.to(device)
    policy.eval()

    calculator = create_ocr_reward_calculator()
    controller = create_sparse_controller('learned', policy_model=policy)

    print(f"✅ 所有组件创建成功")

    # 模拟一个episode
    num_steps = 10
    obs_dim = 321

    for step in range(num_steps):
        # 模拟观测
        obs = np.random.randn(obs_dim)

        # 决策是否控制
        if not controller.should_make_decision(step):
            continue

        # 选择车辆
        selected_ids, info = controller.select_critical_vehicles(obs)

        # 计算动作
        actions_dict = controller.compute_actions(obs, selected_ids)

        # 模拟环境反馈
        num_vehicles = np.random.randint(20, 50)
        vehicle_info = [{
            'id': f'veh_{i}',
            'arrived': np.random.random() < 0.1,
            'traveled': np.random.uniform(100, 1000),
            'total': np.random.uniform(1000, 2000),
            'speed': np.random.uniform(5, 25),
            'accel': np.random.uniform(-2, 2)
        } for i in range(num_vehicles)]

        # 更新奖励
        accel_commands = len(actions_dict['accel'])
        lane_changes = sum(1 for p in actions_dict['lane_change'].values() if p > 0.5)
        reward = calculator.update(vehicle_info, accel_commands, lane_changes, len(selected_ids))

        if step == 0:
            print(f"   Step {step}: 控制车辆数={len(selected_ids)}, 加速指令={accel_commands}, 换道={lane_changes}")

    # 计算最终得分
    scores = calculator.compute_episode_score()

    print(f"✅ 端到端测试完成:")
    print(f"   最终得分: {scores['s_total']:.2f}")


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("轻量级架构 v5.0 测试")
    print("=" * 70)

    try:
        # 测试1: 模型创建
        policy = test_model_creation()

        # 测试2: Top-K选择
        test_top_k_selection(policy)

        # 测试3: OCR奖励
        test_ocr_reward_calculator()

        # 测试4: 稀疏控制器
        test_sparse_controller()

        # 测试5: 端到端集成
        test_integration()

        print("\n" + "=" * 70)
        print("✅ 所有测试通过！")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
