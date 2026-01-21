"""
ICV评分器集成测试脚本

测试统一评分接口的基本功能
"""

import sys
from pathlib import Path
import numpy as np
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.env.rule_based_scorer import RuleBasedVehicleScorer
from src.models.icv_gnn_scorer import ICVGraphScorer, ICVGNNScorer, create_icv_gnn_scorer
from src.env.vehicle_scoring import UnifiedVehicleScorer, create_vehicle_scorer_from_config


def test_rule_scorer():
    """测试规则评分器"""
    print("\n" + "="*60)
    print("测试 1: 规则评分器")
    print("="*60)

    config = {
        'heuristic_selector': {
            'bottleneck_s_min': 1200.0,
            'bottleneck_s_max': 2200.0,
            'desired_speed': 30.0,
            'max_accel': 2.0,
            'max_decel': -4.5,
            'safe_time_gap': 1.5
        },
        'sparse_controller': {
            'ttc_threshold': 3.0
        }
    }

    scorer = RuleBasedVehicleScorer(config=config, frenet_system=None)

    # 模拟车辆状态
    vehicle_states = {
        'veh_0': {
            'id': 'veh_0',
            's': 1400.0,  # 在瓶颈区域
            'd': 0.0,
            'vs': 5.0,    # 低速
            'vd': 0.0,
            'speed': 5.0,
            'acceleration': -1.0,
            'lane_id': 'E2_0',
            'lane_index': 0,
            'angle': 0.0,
            'in_bottleneck': True
        },
        'veh_1': {
            'id': 'veh_1',
            's': 500.0,   # 不在瓶颈区域
            'd': 0.0,
            'vs': 25.0,   # 高速
            'vd': 0.0,
            'speed': 25.0,
            'acceleration': 0.0,
            'lane_id': 'E1_1',
            'lane_index': 1,
            'angle': 0.0,
            'in_bottleneck': False
        }
    }

    context = {
        'traci_lib': None,  # 没有TraCI环境
        'all_vehicle_ids': ['veh_0', 'veh_1']
    }

    scores = scorer.compute_scores(vehicle_states, context)

    print(f"车辆 veh_0 评分: {scores['veh_0']:.4f} (预期: 较高，因为在瓶颈区域且速度低)")
    print(f"车辆 veh_1 评分: {scores['veh_1']:.4f} (预期: 较低，因为不在瓶颈区域且速度高)")

    # 验证评分合理性
    assert scores['veh_0'] > scores['veh_1'], "瓶颈区域的车辆应该有更高的评分"
    print("✓ 规则评分器测试通过")


def test_gnn_model():
    """测试GNN模型"""
    print("\n" + "="*60)
    print("测试 2: GNN模型")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 创建模型
    model = ICVGraphScorer(
        node_dim=9,
        hidden_dim=64,
        num_layers=3,
        dropout=0.1,
        interaction_radius=0.15
    ).to(device)

    # 模拟输入
    batch_size = 2
    num_vehicles = 10
    vehicle_states = torch.randn(batch_size, num_vehicles, 9).to(device)

    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(vehicle_states, return_debug_info=True)

    scores = outputs['scores']  # [B, N, 1]
    embeddings = outputs['node_embeddings']  # [B, N, hidden_dim]
    adj = outputs['adjacency_matrix']  # [B, N, N]

    print(f"评分形状: {scores.shape}")
    print(f"评分范围: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"嵌入形状: {embeddings.shape}")
    print(f"邻接矩阵形状: {adj.shape}")
    print(f"边数（平均）: {adj.sum().item() / batch_size:.2f}")

    # 验证输出
    assert scores.shape == (batch_size, num_vehicles, 1), "评分形状不正确"
    assert embeddings.shape == (batch_size, num_vehicles, 64), "嵌入形状不正确"
    assert torch.all((scores >= 0) & (scores <= 1)), "评分应该在[0, 1]范围内"

    print("✓ GNN模型测试通过")


def test_icv_gnn_scorer():
    """测试ICV GNN评分器包装类"""
    print("\n" + "="*60)
    print("测试 3: ICV GNN评分器包装类")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = {
        'neural_icv_scoring': {
            'node_dim': 9,
            'hidden_dim': 64,
            'num_layers': 3,
            'dropout': 0.1,
            'interaction_radius': 0.15,
            'checkpoint_path': None
        }
    }

    scorer = create_icv_gnn_scorer(config=config, device=device)

    # 模拟车辆状态
    vehicle_states = {
        'veh_0': {
            's': 1400.0,
            'd': 0.0,
            'vs': 5.0,
            'vd': 0.0,
            'speed': 5.0,
            'acceleration': -1.0,
            'lane_index': 0,
            'angle': 0.0,
            'in_bottleneck': True
        },
        'veh_1': {
            's': 500.0,
            'd': 0.0,
            'vs': 25.0,
            'vd': 0.0,
            'speed': 25.0,
            'acceleration': 0.0,
            'lane_index': 1,
            'angle': 0.0,
            'in_bottleneck': False
        }
    }

    context = {
        'traci_lib': None,
        'all_vehicle_ids': ['veh_0', 'veh_1']
    }

    scores = scorer.compute_scores(vehicle_states, context)

    print(f"车辆 veh_0 评分: {scores['veh_0']:.4f}")
    print(f"车辆 veh_1 评分: {scores['veh_1']:.4f}")

    # 验证评分范围
    for veh_id, score in scores.items():
        assert 0.0 <= score <= 1.0, f"车辆 {veh_id} 的评分 {score} 超出[0, 1]范围"

    print("✓ ICV GNN评分器包装类测试通过")


def test_unified_scorer():
    """测试统一评分器"""
    print("\n" + "="*60)
    print("测试 4: 统一评分器")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = {
        'neural_icv_scoring': {
            'enabled': True,
            'node_dim': 9,
            'hidden_dim': 64,
            'num_layers': 3,
            'dropout': 0.1,
            'interaction_radius': 0.15,
            'checkpoint_path': None,
            'fallback_on_error': True
        },
        'heuristic_selector': {
            'bottleneck_s_min': 1200.0,
            'bottleneck_s_max': 2200.0,
            'desired_speed': 30.0,
            'max_accel': 2.0,
            'max_decel': -4.5,
            'safe_time_gap': 1.5
        },
        'sparse_controller': {
            'ttc_threshold': 3.0
        }
    }

    scorer = create_vehicle_scorer_from_config(config=config, device=device)

    # 模拟车辆状态
    vehicle_states = {
        'veh_0': {
            's': 1400.0,
            'd': 0.0,
            'vs': 5.0,
            'vd': 0.0,
            'speed': 5.0,
            'acceleration': -1.0,
            'lane_index': 0,
            'angle': 0.0,
            'in_bottleneck': True
        },
        'veh_1': {
            's': 500.0,
            'd': 0.0,
            'vs': 25.0,
            'vd': 0.0,
            'speed': 25.0,
            'acceleration': 0.0,
            'lane_index': 1,
            'angle': 0.0,
            'in_bottleneck': False
        }
    }

    context = {
        'traci_lib': None,
        'all_vehicle_ids': ['veh_0', 'veh_1']
    }

    # 测试神经网络模式
    print("\n[神经网络模式]")
    scores = scorer.compute_scores(vehicle_states, context)
    print(f"车辆 veh_0 评分: {scores['veh_0']:.4f}")
    print(f"车辆 veh_1 评分: {scores['veh_1']:.4f}")

    stats = scorer.get_statistics()
    print(f"统计: {stats}")

    # 测试Top-K选择
    print("\n[Top-K选择]")
    top_k = scorer.get_top_k_vehicles(vehicle_states, context, k=1)
    print(f"Top-1 车辆: {top_k}")
    assert len(top_k) == 1, "Top-K选择应该返回K个车辆"

    # 切换到规则模式
    print("\n[规则模式]")
    scorer.switch_to_rule()
    scores_rule = scorer.compute_scores(vehicle_states, context)
    print(f"车辆 veh_0 评分: {scores_rule['veh_0']:.4f}")
    print(f"车辆 veh_1 评分: {scores_rule['veh_1']:.4f}")

    # 切换回神经网络模式
    print("\n[切换回神经网络模式]")
    scorer.switch_to_neural()
    scores_neural = scorer.compute_scores(vehicle_states, context)
    print(f"车辆 veh_0 评分: {scores_neural['veh_0']:.4f}")
    print(f"车辆 veh_1 评分: {scores_neural['veh_1']:.4f}")

    stats = scorer.get_statistics()
    print(f"统计: {stats}")

    print("✓ 统一评分器测试通过")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("ICV评分器集成测试")
    print("="*60)

    try:
        test_rule_scorer()
        test_gnn_model()
        test_icv_gnn_scorer()
        test_unified_scorer()

        print("\n" + "="*60)
        print("所有测试通过! ✓")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
