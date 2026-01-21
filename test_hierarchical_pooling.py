"""
测试HierarchicalPooling实现

验证点：
1. 输出维度正确性
2. 梯度回传
3. 数值稳定性
"""

import torch
import torch.nn as nn
import sys
sys.path.append('.')

from src.models.joint_icv_policy import JointICVPolicy, HierarchicalPooling

def test_hierarchical_pooling():
    """测试HierarchicalPooling模块"""
    print("=" * 80)
    print("测试 HierarchicalPooling 模块")
    print("=" * 80)

    # 创建模块
    hidden_dim = 64
    num_lanes = 4
    num_sections = 5
    batch_size = 4
    num_vehicles = 32

    pooling = HierarchicalPooling(
        hidden_dim=hidden_dim,
        num_lanes=num_lanes,
        num_sections=num_sections
    ).cuda()

    # 创建测试数据
    embeddings = torch.randn(batch_size, num_vehicles, hidden_dim).cuda()
    vehicle_states = torch.randn(batch_size, num_vehicles, 9).cuda()

    # 设置合理的车辆状态
    vehicle_states[:, :, 0] = torch.rand(batch_size, num_vehicles).cuda() * 3000  # 纵向位置 [0, 3000]
    vehicle_states[:, :, 6] = torch.randint(0, num_lanes, (batch_size, num_vehicles)).float().cuda()  # 车道索引 [0, 3]

    print(f"\n输入:")
    print(f"  embeddings: {embeddings.shape}")
    print(f"  vehicle_states: {vehicle_states.shape}")

    # 前向传播
    print("\n执行前向传播...")
    pooled_features = pooling(embeddings, vehicle_states)

    # 验证输出维度
    print("\n输出:")
    assert 'vehicle' in pooled_features, "缺少 'vehicle' 输出"
    assert 'lane' in pooled_features, "缺少 'lane' 输出"
    assert 'section' in pooled_features, "缺少 'section' 输出"
    assert 'global' in pooled_features, "缺少 'global' 输出"

    vehicle_feat = pooled_features['vehicle']
    lane_feat = pooled_features['lane']
    section_feat = pooled_features['section']
    global_feat = pooled_features['global']

    print(f"  vehicle: {vehicle_feat.shape} (期望: [{batch_size}, {num_vehicles}, {hidden_dim}])")
    print(f"  lane: {lane_feat.shape} (期望: [{batch_size}, {num_lanes}, {hidden_dim}])")
    print(f"  section: {section_feat.shape} (期望: [{batch_size}, {num_sections}, {hidden_dim}])")
    print(f"  global: {global_feat.shape} (期望: [{batch_size}, {hidden_dim}])")

    assert vehicle_feat.shape == (batch_size, num_vehicles, hidden_dim), f"vehicle维度错误: {vehicle_feat.shape}"
    assert lane_feat.shape == (batch_size, num_lanes, hidden_dim), f"lane维度错误: {lane_feat.shape}"
    assert section_feat.shape == (batch_size, num_sections, hidden_dim), f"section维度错误: {section_feat.shape}"
    assert global_feat.shape == (batch_size, hidden_dim), f"global维度错误: {global_feat.shape}"

    print("\n✅ 维度检查通过!")

    # 测试梯度回传
    print("\n测试梯度回传...")
    loss = (pooled_features['global'].sum() +
            pooled_features['lane'].sum() +
            pooled_features['section'].sum() +
            pooled_features['vehicle'].sum())
    loss.backward()

    # 检查梯度是否存在
    assert pooling.lane_attention[0].weight.grad is not None, "lane_attention梯度不存在"
    assert pooling.section_attention[0].weight.grad is not None, "section_attention梯度不存在"
    assert pooling.global_attention[0].weight.grad is not None, "global_attention梯度不存在"

    print("✅ 梯度回传检查通过!")

    # 测试数值稳定性
    print("\n测试数值稳定性...")
    assert not torch.isnan(vehicle_feat).any(), "vehicle_feat包含NaN"
    assert not torch.isnan(lane_feat).any(), "lane_feat包含NaN"
    assert not torch.isnan(section_feat).any(), "section_feat包含NaN"
    assert not torch.isnan(global_feat).any(), "global_feat包含NaN"

    assert not torch.isinf(vehicle_feat).any(), "vehicle_feat包含Inf"
    assert not torch.isinf(lane_feat).any(), "lane_feat包含Inf"
    assert not torch.isinf(section_feat).any(), "section_feat包含Inf"
    assert not torch.isinf(global_feat).any(), "global_feat包含Inf"

    print("✅ 数值稳定性检查通过!")

    print("\n" + "=" * 80)
    print("✅ HierarchicalPooling 所有测试通过!")
    print("=" * 80)


def test_joint_icv_policy_with_pooling():
    """测试JointICVPolicy集成"""
    print("\n" + "=" * 80)
    print("测试 JointICVPolicy 集成 HierarchicalPooling")
    print("=" * 80)

    # 创建策略
    policy = JointICVPolicy(
        obs_dim=321,
        node_dim=9,
        hidden_dim=64,
        num_layers=3,
        initial_k_ratio=0.10,
        device='cuda'
    ).cuda()

    print(f"\n策略参数量: {sum(p.numel() for p in policy.parameters()):,}")

    # 创建测试观测
    batch_size = 4
    max_vehicles = 32
    obs_dim = 321

    # 创建结构化的观测
    obs_list = []
    for _ in range(batch_size):
        vehicle_states = torch.randn(max_vehicles, 9)

        # 设置合理的车辆状态
        vehicle_states[:, 0] = torch.rand(max_vehicles) * 3000  # x [0, 3000]
        vehicle_states[:, 1] = torch.rand(max_vehicles) * 10 - 5  # y [-5, 5]
        vehicle_states[:, 2] = torch.rand(max_vehicles) * 30  # vx [0, 30]
        vehicle_states[:, 3] = torch.rand(max_vehicles) * 6 - 3  # vy [-3, 3]
        vehicle_states[:, 4] = torch.rand(max_vehicles) * 4.5 - 1.5  # ax [-1.5, 3]
        vehicle_states[:, 5] = torch.rand(max_vehicles) * 2 - 1  # ay [-1, 1]
        vehicle_states[:, 6] = torch.randint(0, 4, (max_vehicles,)).float()  # lane [0, 3]
        vehicle_states[:, 7] = torch.rand(max_vehicles)  # length
        vehicle_states[:, 8] = torch.rand(max_vehicles)  # width

        global_stats = torch.randn(32)
        num_vehicles = torch.tensor([max_vehicles])

        obs_single = torch.cat([
            vehicle_states.flatten(),
            global_stats,
            num_vehicles
        ])
        obs_list.append(obs_single)

    obs = torch.stack(obs_list).cuda()  # [batch_size, 321]

    print(f"\n输入:")
    print(f"  obs: {obs.shape}")

    # 前向传播
    print("\n执行前向传播...")
    outputs = policy(obs, deterministic=False)

    # 验证输出
    print("\n输出:")
    assert 'actions' in outputs, "缺少 'actions' 输出"
    assert 'value' in outputs, "缺少 'value' 输出"
    assert 'log_prob' in outputs, "缺少 'log_prob' 输出"
    assert 'importance' in outputs, "缺少 'importance' 输出"
    assert 'selection' in outputs, "缺少 'selection' 输出"
    assert 'k' in outputs, "缺少 'k' 输出"
    assert 'embeddings' in outputs, "缺少 'embeddings' 输出"
    assert 'pooled_features' in outputs, "缺少 'pooled_features' 输出"

    print(f"  actions: {outputs['actions'].shape}")
    print(f"  value: {outputs['value'].shape}")
    print(f"  log_prob: {outputs['log_prob'].shape}")
    print(f"  importance: {outputs['importance'].shape}")
    print(f"  selection: {outputs['selection'].shape}")
    print(f"  k: {outputs['k']}")
    print(f"  embeddings: {outputs['embeddings'].shape}")

    pooled = outputs['pooled_features']
    print(f"  pooled_features:")
    print(f"    vehicle: {pooled['vehicle'].shape}")
    print(f"    lane: {pooled['lane'].shape}")
    print(f"    section: {pooled['section'].shape}")
    print(f"    global: {pooled['global'].shape}")

    # 验证维度
    assert outputs['actions'].shape == (batch_size, 64), f"actions维度错误: {outputs['actions'].shape}"
    assert outputs['value'].shape == (batch_size, 1), f"value维度错误: {outputs['value'].shape}"
    assert outputs['log_prob'].shape == (batch_size,), f"log_prob维度错误: {outputs['log_prob'].shape}"
    assert pooled['vehicle'].shape == (batch_size, 32, 64), f"pooled['vehicle']维度错误"
    assert pooled['lane'].shape == (batch_size, 4, 64), f"pooled['lane']维度错误"
    assert pooled['section'].shape == (batch_size, 5, 64), f"pooled['section']维度错误"
    assert pooled['global'].shape == (batch_size, 64), f"pooled['global']维度错误"

    print("\n✅ 维度检查通过!")

    # 测试梯度回传
    print("\n测试梯度回传...")
    loss = (outputs['value'].sum() +
            outputs['log_prob'].sum() +
            outputs['pooled_features']['lane'].sum() +
            outputs['pooled_features']['section'].sum() +
            outputs['pooled_features']['global'].sum())
    loss.backward()

    # 检查关键组件的梯度
    assert policy.hierarchical_pooling.lane_attention[0].weight.grad is not None, "lane_attention梯度不存在"
    assert policy.hierarchical_pooling.section_attention[0].weight.grad is not None, "section_attention梯度不存在"
    assert policy.gnn_encoder.gnn_layers[0].linear.weight.grad is not None, "GNN梯度不存在"
    assert policy.importance_predictor.mlp[0].weight.grad is not None, "importance_predictor梯度不存在"

    print("✅ 梯度回传检查通过!")

    # 测试确定性模式
    print("\n测试确定性模式...")
    policy.eval()  # 设置为评估模式（禁用dropout）
    outputs_det1 = policy(obs, deterministic=True)
    outputs_det2 = policy(obs, deterministic=True)

    # 确定性模式下，两次前向传播应该相同
    assert torch.allclose(outputs_det1['actions'], outputs_det2['actions'], atol=1e-6), "确定性模式失败"

    print("✅ 确定性模式检查通过!")
    policy.train()  # 恢复训练模式

    # 测试数值稳定性
    print("\n测试数值稳定性...")
    assert not torch.isnan(outputs['actions']).any(), "actions包含NaN"
    assert not torch.isnan(outputs['value']).any(), "value包含NaN"
    assert not torch.isnan(pooled['global']).any(), "pooled['global']包含NaN"

    print("✅ 数值稳定性检查通过!")

    print("\n" + "=" * 80)
    print("✅ JointICVPolicy 集成测试通过!")
    print("=" * 80)


if __name__ == "__main__":
    print("\n开始测试 HierarchicalPooling 实现...\n")

    try:
        test_hierarchical_pooling()
        test_joint_icv_policy_with_pooling()

        print("\n" + "=" * 80)
        print("🎉 所有测试通过!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
