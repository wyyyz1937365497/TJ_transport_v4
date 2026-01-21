"""
v5.0完整架构端到端测试

测试内容：
1. 完整策略前向传播（所有组件）
2. 各组件独立启用测试
3. 梯度回传验证
4. 性能基准测试
5. 与环境交互测试
"""

import torch
import torch.nn as nn
import sys
import time
sys.path.append('.')

from src.models.joint_icv_policy import JointICVPolicy, create_joint_icv_policy


def create_test_obs(batch_size=4, num_vehicles=32):
    """创建测试观测"""
    obs_list = []
    for _ in range(batch_size):
        vehicle_states = torch.randn(num_vehicles, 9)

        # 设置合理的车辆状态
        vehicle_states[:, 0] = torch.rand(num_vehicles) * 3000  # x [0, 3000]
        vehicle_states[:, 1] = torch.rand(num_vehicles) * 10 - 5  # y [-5, 5]
        vehicle_states[:, 2] = torch.rand(num_vehicles) * 30  # vx [0, 30]
        vehicle_states[:, 3] = torch.rand(num_vehicles) * 6 - 3  # vy [-3, 3]
        vehicle_states[:, 4] = torch.rand(num_vehicles) * 4.5 - 1.5  # ax [-1.5, 3]
        vehicle_states[:, 5] = torch.rand(num_vehicles) * 2 - 1  # ay [-1, 1]
        vehicle_states[:, 6] = torch.randint(0, 4, (num_vehicles,)).float()  # lane [0, 3]
        vehicle_states[:, 7] = torch.rand(num_vehicles)  # length
        vehicle_states[:, 8] = torch.rand(num_vehicles)  # width

        global_stats = torch.randn(32)
        num_vehicles_tensor = torch.tensor([num_vehicles])

        obs_single = torch.cat([
            vehicle_states.flatten(),
            global_stats,
            num_vehicles_tensor
        ])
        obs_list.append(obs_single)

    return torch.stack(obs_list).cuda()


def test_baseline_policy():
    """测试基线策略（仅核心组件）"""
    print("=" * 80)
    print("测试 1: 基线策略（仅核心组件）")
    print("=" * 80)

    # 创建策略
    policy = create_joint_icv_policy(device='cuda')

    print(f"\n策略参数量: {sum(p.numel() for p in policy.parameters()):,}")

    # 创建测试观测
    obs = create_test_obs(batch_size=4, num_vehicles=32)

    # 前向传播
    print("\n执行前向传播...")
    start_time = time.time()
    outputs = policy(obs, deterministic=False)
    elapsed_time = (time.time() - start_time) * 1000

    # 验证输出
    print(f"\n前向传播耗时: {elapsed_time:.2f}ms")
    print(f"\n输出:")
    print(f"  actions: {outputs['actions'].shape}")
    print(f"  value: {outputs['value'].shape}")
    print(f"  log_prob: {outputs['log_prob'].shape}")
    print(f"  importance: {outputs['importance'].shape}")
    print(f"  selection: {outputs['selection'].shape}")
    print(f"  k: {outputs['k']}")
    print(f"  pooled_features: {outputs['pooled_features']['vehicle'].shape}, {outputs['pooled_features']['global'].shape}")
    print(f"  safety_reward: {outputs['safety_reward'].shape}")

    # 测试梯度回传
    print("\n测试梯度回传...")

    # 清零梯度
    policy.zero_grad()

    # 包含所有输出以确保梯度回传到所有组件
    loss = (outputs['value'].sum() +
            outputs['log_prob'].sum() +
            outputs['pooled_features']['lane'].sum() +
            outputs['pooled_features']['section'].sum() +
            outputs['pooled_features']['global'].sum())
    loss.backward()

    # 检查梯度
    assert policy.gnn_encoder.gnn_layers[0].linear.weight.grad is not None, "GNN梯度不存在"
    assert policy.hierarchical_pooling.lane_attention[0].weight.grad is not None, "HierarchicalPooling梯度不存在"

    print("✅ 基线策略测试通过!")

    return elapsed_time


def test_policy_with_world_model():
    """测试启用WorldModel的策略"""
    print("\n" + "=" * 80)
    print("测试 2: 启用WorldModel")
    print("=" * 80)

    # 创建策略
    policy = create_joint_icv_policy(device='cuda')

    # 启用WorldModel
    policy.enable_world_model(num_vehicles=32, latent_dim=64)

    print(f"\n策略参数量: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"WorldModel参数量: {sum(p.numel() for p in policy.world_model.parameters()):,}")

    # 创建测试观测
    obs = create_test_obs(batch_size=4, num_vehicles=32)

    # 前向传播
    print("\n执行前向传播...")
    start_time = time.time()
    outputs = policy(obs, deterministic=False)
    elapsed_time = (time.time() - start_time) * 1000

    print(f"\n前向传播耗时: {elapsed_time:.2f}ms")

    # 验证WorldModel输出
    assert 'world_model' in outputs, "缺少world_model输出"
    world_outputs = outputs['world_model']
    print(f"\nWorldModel输出:")
    print(f"  z_flow: {world_outputs['z_flow'].shape}")
    print(f"  z_risk: {world_outputs['z_risk'].shape}")
    print(f"  pred_next_states: {world_outputs['pred_next_states'].shape}")
    print(f"  risk_prob: {world_outputs['risk_prob'].shape}")

    # 测试梯度回传
    print("\n测试梯度回传...")

    # 清零梯度
    policy.zero_grad()

    # 包含WorldModel输出
    loss = (outputs['value'].sum() +
            outputs['log_prob'].sum() +
            outputs['world_model']['z_flow'].sum() +
            outputs['world_model']['z_risk'].sum())
    loss.backward()

    assert policy.world_model.encoder.gru.weight_hh_l0.grad is not None, "WorldModel梯度不存在"

    print("✅ WorldModel集成测试通过!")

    return elapsed_time


def test_policy_with_cost_critic():
    """测试启用CostCritic的策略"""
    print("\n" + "=" * 80)
    print("测试 3: 启用CostCritic")
    print("=" * 80)

    # 创建策略
    policy = create_joint_icv_policy(device='cuda')

    # 启用CostCritic
    policy.enable_cost_critic(global_dim=64)

    print(f"\n策略参数量: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"CostCritic参数量: {sum(p.numel() for p in policy.cost_critic.parameters()):,}")

    # 创建测试观测
    obs = create_test_obs(batch_size=4, num_vehicles=32)

    # 前向传播
    print("\n执行前向传播...")
    start_time = time.time()
    outputs = policy(obs, deterministic=False)
    elapsed_time = (time.time() - start_time) * 1000

    print(f"\n前向传播耗时: {elapsed_time:.2f}ms")

    # 验证CostCritic输出
    assert 'cost_value' in outputs, "缺少cost_value输出"
    assert 'cost_cost' in outputs, "缺少cost_cost输出"
    print(f"\nCostCritic输出:")
    print(f"  cost_value: {outputs['cost_value'].shape}")
    print(f"  cost_cost: {outputs['cost_cost'].shape}")

    # 测试梯度回传
    print("\n测试梯度回传...")

    # 清零梯度
    policy.zero_grad()

    loss = outputs['value'].sum() + outputs['cost_value'].sum() + outputs['cost_cost'].sum()
    loss.backward()

    assert policy.cost_critic.value_head[-1].weight.grad is not None, "CostCritic梯度不存在"

    print("✅ CostCritic集成测试通过!")

    return elapsed_time


def test_policy_with_dynamic_gate():
    """测试启用DynamicWeightGate的策略"""
    print("\n" + "=" * 80)
    print("测试 4: 启用DynamicWeightGate")
    print("=" * 80)

    # 创建策略
    policy = create_joint_icv_policy(device='cuda')

    # 启用DynamicWeightGate
    policy.enable_dynamic_gate(global_dim=64)

    print(f"\n策略参数量: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"DynamicWeightGate参数量: {sum(p.numel() for p in policy.dynamic_gate.parameters()):,}")

    # 创建测试观测
    obs = create_test_obs(batch_size=4, num_vehicles=32)

    # 前向传播
    print("\n执行前向传播...")
    start_time = time.time()
    outputs = policy(obs, deterministic=False)
    elapsed_time = (time.time() - start_time) * 1000

    print(f"\n前向传播耗时: {elapsed_time:.2f}ms")

    # 验证DynamicWeightGate输出
    assert 'reward_weights' in outputs, "缺少reward_weights输出"
    print(f"\nDynamicWeightGate输出:")
    print(f"  reward_weights: {outputs['reward_weights'].shape}")
    print(f"  权重示例: {outputs['reward_weights'][0].detach().cpu().numpy()}")

    # 验证权重归一化
    weight_sums = outputs['reward_weights'].sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), "权重和不为1"

    # 测试梯度回传
    print("\n测试梯度回传...")

    # 清零梯度
    policy.zero_grad()

    loss = outputs['value'].sum() + outputs['reward_weights'].sum()
    loss.backward()

    assert policy.dynamic_gate.weight_head.weight.grad is not None, "DynamicWeightGate梯度不存在"

    print("✅ DynamicWeightGate集成测试通过!")

    return elapsed_time


def test_policy_with_safety_shield():
    """测试启用SafetyShield的策略"""
    print("\n" + "=" * 80)
    print("测试 5: 启用SafetyShield（推理模式）")
    print("=" * 80)

    # 创建策略
    policy = create_joint_icv_policy(device='cuda')

    # 启用SafetyShield
    policy.enable_safety_shield()

    policy.eval()  # 设置为评估模式

    print(f"\nSafetyShield已启用（仅推理模式）")

    # 创建测试观测
    obs = create_test_obs(batch_size=4, num_vehicles=32)

    # 前向传播（推理模式）
    print("\n执行前向传播（推理模式）...")
    start_time = time.time()
    outputs = policy(obs, deterministic=True)
    elapsed_time = (time.time() - start_time) * 1000

    print(f"\n前向传播耗时: {elapsed_time:.2f}ms")

    # 验证安全奖励
    print(f"\nSafetyShield输出:")
    print(f"  safety_reward: {outputs['safety_reward'].shape}")
    print(f"  安全奖励示例: {outputs['safety_reward'].cpu().numpy()}")

    print("✅ SafetyShield集成测试通过!")

    policy.train()  # 恢复训练模式

    return elapsed_time


def test_full_v5_architecture():
    """测试完整v5.0架构（所有组件）"""
    print("\n" + "=" * 80)
    print("测试 6: 完整v5.0架构（所有组件启用）")
    print("=" * 80)

    # 创建策略
    policy = create_joint_icv_policy(device='cuda')

    # 启用所有v5.0组件
    policy.enable_world_model(num_vehicles=32, latent_dim=64)
    policy.enable_cost_critic(global_dim=64)
    policy.enable_dynamic_gate(global_dim=64)
    policy.enable_safety_shield()

    print(f"\n完整v5.0架构参数量: {sum(p.numel() for p in policy.parameters()):,}")

    # 创建测试观测
    obs = create_test_obs(batch_size=4, num_vehicles=32)

    # 训练模式前向传播
    print("\n执行前向传播（训练模式）...")
    policy.train()
    start_time = time.time()
    outputs_train = policy(obs, deterministic=False)
    train_time = (time.time() - start_time) * 1000

    print(f"训练模式耗时: {train_time:.2f}ms")

    # 推理模式前向传播
    print("\n执行前向传播（推理模式）...")
    policy.eval()
    start_time = time.time()
    outputs_eval = policy(obs, deterministic=True)
    eval_time = (time.time() - start_time) * 1000

    print(f"推理模式耗时: {eval_time:.2f}ms")

    # 验证所有输出
    print(f"\n完整输出:")
    print(f"  actions: {outputs_train['actions'].shape}")
    print(f"  value: {outputs_train['value'].shape}")
    print(f"  cost_value: {outputs_train['cost_value'].shape}")
    print(f"  reward_weights: {outputs_train['reward_weights'].shape}")
    print(f"  world_model: keys={outputs_train['world_model'].keys()}")
    print(f"  safety_reward: {outputs_train['safety_reward'].shape}")

    # 测试梯度回传
    print("\n测试梯度回传...")
    policy.train()
    policy.zero_grad()

    # 包含所有输出
    loss = (outputs_train['value'].sum() +
            outputs_train['cost_value'].sum() +
            outputs_train['cost_cost'].sum() +
            outputs_train['log_prob'].sum() +
            outputs_train['world_model']['z_flow'].sum() +
            outputs_train['reward_weights'].sum())
    loss.backward()

    # 验证所有组件的梯度
    assert policy.world_model.encoder.gru.weight_hh_l0.grad is not None, "WorldModel梯度不存在"
    assert policy.cost_critic.value_head[-1].weight.grad is not None, "CostCritic梯度不存在"
    assert policy.dynamic_gate.weight_head.weight.grad is not None, "DynamicWeightGate梯度不存在"
    assert policy.gnn_encoder.gnn_layers[0].linear.weight.grad is not None, "GNN梯度不存在"

    print("✅ 完整v5.0架构测试通过!")

    return train_time, eval_time


def test_performance_benchmark():
    """性能基准测试"""
    print("\n" + "=" * 80)
    print("测试 7: 性能基准测试")
    print("=" * 80)

    # 测试不同batch size的性能
    batch_sizes = [1, 4, 8, 16]
    num_iterations = 100

    print(f"\n测试配置: {num_iterations}次迭代")

    for batch_size in batch_sizes:
        # 创建策略
        policy = create_joint_icv_policy(device='cuda')
        policy.enable_world_model()
        policy.enable_cost_critic()
        policy.enable_dynamic_gate()

        obs = create_test_obs(batch_size=batch_size, num_vehicles=32)

        # 预热
        for _ in range(10):
            _ = policy(obs, deterministic=False)

        # 计时
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(num_iterations):
            _ = policy(obs, deterministic=False)

        torch.cuda.synchronize()
        total_time = time.time() - start_time

        avg_time = (total_time / num_iterations) * 1000  # ms
        throughput = (batch_size / (total_time / num_iterations))  # samples/s

        print(f"  Batch Size={batch_size:2d}: {avg_time:6.2f}ms, {throughput:6.1f} samples/s")

    print("✅ 性能基准测试完成!")


if __name__ == "__main__":
    print("\n开始v5.0完整架构端到端测试...\n")

    try:
        # 测试1: 基线策略
        baseline_time = test_baseline_policy()

        # 测试2-5: 各组件独立测试
        world_model_time = test_policy_with_world_model()
        cost_critic_time = test_policy_with_cost_critic()
        dynamic_gate_time = test_policy_with_dynamic_gate()
        safety_shield_time = test_policy_with_safety_shield()

        # 测试6: 完整架构
        train_time, eval_time = test_full_v5_architecture()

        # 测试7: 性能基准
        test_performance_benchmark()

        # 总结
        print("\n" + "=" * 80)
        print("🎉 所有测试通过!")
        print("=" * 80)

        print("\n性能总结:")
        print(f"  基线策略: {baseline_time:.2f}ms")
        print(f"  +WorldModel: {world_model_time:.2f}ms (+{world_model_time-baseline_time:.2f}ms)")
        print(f"  +CostCritic: {cost_critic_time:.2f}ms (+{cost_critic_time-baseline_time:.2f}ms)")
        print(f"  +DynamicGate: {dynamic_gate_time:.2f}ms (+{dynamic_gate_time-baseline_time:.2f}ms)")
        print(f"  +SafetyShield: {safety_shield_time:.2f}ms (推理)")
        print(f"  完整架构训练: {train_time:.2f}ms")
        print(f"  完整架构推理: {eval_time:.2f}ms")

        print("\n预期目标:")
        print(f"  训练速度: <20ms ✓" if train_time < 20 else f"  训练速度: <20ms ✗ ({train_time:.2f}ms)")
        print(f"  推理速度: <15ms ✓" if eval_time < 15 else f"  推理速度: <15ms ✗ ({eval_time:.2f}ms)")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
