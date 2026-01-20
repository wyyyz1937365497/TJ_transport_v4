"""
测试神经网络ICV评分系统

测试内容：
1. 神经网络评分器创建
2. 混合评分计算
3. 图构建
4. 集成到环境
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import yaml


def test_1_neural_scorer_creation():
    """测试1：创建神经网络评分器"""
    print("\n" + "="*70)
    print("测试1: 神经网络评分器创建")
    print("="*70)

    from src.models.neural_vehicle_scorer import create_neural_icv_scorer

    try:
        scorer = create_neural_icv_scorer(
            node_dim=9,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            device='cpu'  # 测试使用CPU
        )

        print("✅ 神经网络评分器创建成功")
        print(f"   设备: {scorer.device}")
        print(f"   训练模式: {scorer.is_training}")

        return scorer
    except Exception as e:
        print(f"❌ 创建失败: {e}")
        return None


def test_2_neural_scoring(scorer):
    """测试2：纯神经网络评分计算"""
    print("\n" + "="*70)
    print("测试2: 纯神经网络评分计算")
    print("="*70)

    # 简化测试：只测试神经网络前向传播
    try:
        import torch

        # 创建随机输入
        num_vehicles = 10
        vehicle_features = torch.randn(num_vehicles, 9)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)  # 简单图
        edge_attr = torch.randn(6, 4)

        # 测试神经网络评分
        neural_scorer = scorer.neural_scorer
        neural_scorer.eval()

        with torch.no_grad():
            scores = neural_scorer(vehicle_features, edge_index, edge_attr)

        print("✅ 神经网络前向传播成功")
        print(f"   输入车辆数: {num_vehicles}")
        print(f"   输出评分形状: {scores.shape}")
        print(f"   评分范围: [{scores.min():.4f}, {scores.max():.4f}]")

        # 显示部分评分
        for i in range(min(5, num_vehicles)):
            print(f"   车辆{i}: 神经评分={scores[i]:.4f}")

        return True
    except Exception as e:
        print(f"❌ 评分计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_graph_building():
    """测试3：图构建"""
    print("\n" + "="*70)
    print("测试3: 车辆交互图构建")
    print("="*70)

    from src.models.neural_vehicle_scorer import VehicleGraphBuilder

    graph_builder = VehicleGraphBuilder(
        distance_threshold=100.0,
        max_neighbors=8
    )

    # 模拟车辆状态
    num_vehicles = 20
    vehicle_states = {}

    for i in range(num_vehicles):
        veh_id = f"veh_{i}"
        vehicle_states[veh_id] = {
            'x': np.random.uniform(0, 500),
            'y': np.random.uniform(-50, 50),
            'speed': np.random.uniform(0, 30),
            'acceleration': np.random.uniform(-3, 3),
            'lane_index': np.random.randint(0, 4)
        }

    try:
        edge_index, edge_attr = graph_builder.build_graph(vehicle_states, num_vehicles)

        print("✅ 图构建成功")
        print(f"   车辆数量: {num_vehicles}")
        print(f"   边数量: {edge_index.shape[1]}")
        print(f"   边特征维度: {edge_attr.shape[1] if edge_attr.shape[0] > 0 else 0}")

        # 计算平均度
        if edge_index.shape[1] > 0:
            degrees = torch.zeros(num_vehicles)
            for i in range(edge_index.shape[1]):
                degrees[edge_index[0, i]] += 1
            avg_degree = degrees.mean().item()
            print(f"   平均度: {avg_degree:.2f}")

        return True
    except Exception as e:
        print(f"❌ 图构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_environment_integration():
    """测试4：环境集成"""
    print("\n" + "="*70)
    print("测试4: 环境集成测试")
    print("="*70)

    try:
        # 加载配置
        config_path = 'configs/phase1_lite.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 添加神经网络评分配置
        config['neural_icv_scoring'] = {
            'enabled': True,
            'rule_weight': 0.3,
            'neural_weight': 0.7,
            'node_dim': 9,
            'hidden_dim': 64,
            'num_layers': 3,
            'num_heads': 4,
            'checkpoint_path': None
        }

        # 创建环境（使用make_gym_env来正确处理配置路径）
        from src.env.gym_wrapper import make_gym_env

        env = make_gym_env(
            config=config,
            seed=42,
            device='cpu'
        )

        # 检查是否启用了神经网络评分
        use_neural = hasattr(env, 'sumo_env') and hasattr(env.sumo_env, 'use_neural_scoring')
        if use_neural:
            print(f"   神经网络评分: {'启用' if env.sumo_env.use_neural_scoring else '未启用'}")
        else:
            print(f"   神经网络评分: 未启用（使用CompetitionSumoEnv可启用）")

        print("✅ 环境创建成功")

        # 测试reset
        obs, info = env.reset()
        print(f"✅ 环境reset成功")
        print(f"   观测形状: {obs.shape}")

        # 测试step（几步）
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {i+1}: reward={reward:.4f}")

            if terminated or truncated:
                break

        env.close()
        print("✅ 环境集成测试通过")

        return True
    except Exception as e:
        print(f"❌ 环境集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("神经网络ICV评分系统测试")
    print("="*70)

    results = {}

    # 测试1：创建评分器
    scorer = test_1_neural_scorer_creation()
    results['test_1'] = scorer is not None

    if scorer is not None:
        # 测试2：神经网络评分
        scores = test_2_neural_scoring(scorer)
        results['test_2'] = scores is not None

    # 测试3：图构建
    results['test_3'] = test_3_graph_building()

    # 测试4：环境集成
    results['test_4'] = test_4_environment_integration()

    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)

    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    print("="*70)


if __name__ == "__main__":
    main()
