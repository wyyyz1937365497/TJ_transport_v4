"""
测试CostCritic实现

验证点：
1. 输出维度正确性
2. 梯度回传
3. 拉格朗日损失计算
4. λ自适应更新
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('.')

from src.models.cost_critic import (
    CostCritic,
    LagrangianOptimizer,
    ConstrainedPPOOptimizer,
    create_cost_critic,
    create_lagrangian_optimizer
)

def test_cost_critic_forward():
    """测试CostCritic前向传播"""
    print("=" * 80)
    print("测试 CostCritic 前向传播")
    print("=" * 80)

    # 创建模型
    model = create_cost_critic(
        hidden_dim=64,
        global_dim=64,
        dropout=0.1,
        device='cuda'
    )

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建测试数据
    batch_size = 4
    num_vehicles = 32
    hidden_dim = 64
    global_dim = 64

    embeddings = torch.randn(batch_size, num_vehicles, hidden_dim).cuda()
    global_state = torch.randn(batch_size, global_dim).cuda()

    print(f"\n输入:")
    print(f"  embeddings: {embeddings.shape}")
    print(f"  global_state: {global_state.shape}")

    # 前向传播
    print("\n执行前向传播...")
    outputs = model(embeddings, global_state)

    # 验证输出
    print("\n输出:")
    assert 'value' in outputs, "缺少 'value' 输出"
    assert 'cost' in outputs, "缺少 'cost' 输出"

    print(f"  value: {outputs['value'].shape} (期望: [{batch_size}, 1])")
    print(f"  cost: {outputs['cost'].shape} (期望: [{batch_size}, 1])")

    assert outputs['value'].shape == (batch_size, 1), f"value维度错误"
    assert outputs['cost'].shape == (batch_size, 1), f"cost维度错误"

    print("\n✅ 维度检查通过!")

    # 测试数值稳定性
    print("\n测试数值稳定性...")
    assert not torch.isnan(outputs['value']).any(), "value包含NaN"
    assert not torch.isnan(outputs['cost']).any(), "cost包含NaN"

    # Cost应该为正数
    assert (outputs['cost'] >= 0).all(), "cost包含负值"

    print("✅ 数值稳定性检查通过!")

    print("\n" + "=" * 80)
    print("✅ CostCritic 前向传播测试通过!")
    print("=" * 80)


def test_cost_critic_loss():
    """测试CostCritic损失计算"""
    print("\n" + "=" * 80)
    print("测试 CostCritic 损失计算")
    print("=" * 80)

    # 创建模型
    model = create_cost_critic(device='cuda')

    # 创建测试数据
    batch_size = 4

    embeddings = torch.randn(batch_size, 32, 64).cuda()
    global_state = torch.randn(batch_size, 64).cuda()
    target_costs = torch.rand(batch_size, 1).cuda()  # [0, 1]范围

    # 前向传播
    outputs = model(embeddings, global_state)

    # 计算损失
    print("\n计算损失...")
    cost_loss = model.compute_cost_loss(outputs['cost'], target_costs)
    value_loss = F.mse_loss(outputs['value'], torch.randn_like(outputs['value']))
    loss = cost_loss + value_loss

    print(f"\n损失:")
    print(f"  total_loss: {loss.item():.4f}")
    print(f"  cost_loss: {cost_loss.item():.4f}")
    print(f"  value_loss: {value_loss.item():.4f}")

    assert not torch.isnan(loss), "cost_loss是NaN"

    print("\n✅ 损失计算检查通过!")

    # 测试梯度回传
    print("\n测试梯度回传...")
    loss.backward()

    # 检查关键组件的梯度
    assert model.value_head[-1].weight.grad is not None, "value_head梯度不存在"
    # cost_head的最后一层是Softplus，前一层是Linear
    assert model.cost_head[-2].weight.grad is not None, "cost_head梯度不存在"

    print("✅ 梯度回传检查通过!")

    print("\n" + "=" * 80)
    print("✅ CostCritic 损失计算测试通过!")
    print("=" * 80)


def test_lagrangian_optimizer():
    """测试拉格朗日优化器"""
    print("\n" + "=" * 80)
    print("测试 LagrangianOptimizer")
    print("=" * 80)

    # 创建优化器
    lagrangian = create_lagrangian_optimizer(
        initial_lambda=0.1,
        lambda_lr=0.01,
        cost_threshold=0.5
    )

    print(f"\n初始λ: {lagrangian.get_lambda():.4f}")

    # 测试拉格朗日损失计算
    batch_size = 4
    rewards = torch.randn(batch_size).cuda()
    costs = torch.rand(batch_size).cuda() * 0.8  # [0, 0.8]范围

    print(f"\n测试数据:")
    print(f"  rewards: {rewards}")
    print(f"  costs: {costs}")
    print(f"  cost_threshold: {lagrangian.cost_threshold}")

    # 计算拉格朗日损失
    print("\n计算拉格朗日损失...")
    lagrangian_loss = lagrangian.compute_lagrangian_loss(rewards, costs)

    print(f"\n拉格朗日损失: {lagrangian_loss}")

    assert lagrangian_loss.shape == (batch_size,), f"lagrangian_loss维度错误"
    assert not torch.isnan(lagrangian_loss).any(), "lagrangian_loss包含NaN"

    print("✅ 拉格朗日损失计算检查通过!")

    # 测试λ更新
    print("\n测试λ更新...")

    # 场景1：成本低于阈值
    print("\n场景1：成本低于阈值 (cost=0.3)")
    update_info = lagrangian.update_lambda(mean_cost=0.3)
    print(f"  更新前: {update_info['old_lambda']:.4f}")
    print(f"  更新后: {update_info['new_lambda']:.4f}")
    print(f"  约束违反: {update_info['constraint_violation']:.4f}")
    assert update_info['new_lambda'] <= update_info['old_lambda'], "成本低于阈值时，λ应该减小或保持"

    # 场景2：成本高于阈值
    print("\n场景2：成本高于阈值 (cost=0.7)")
    update_info = lagrangian.update_lambda(mean_cost=0.7)
    print(f"  更新前: {update_info['old_lambda']:.4f}")
    print(f"  更新后: {update_info['new_lambda']:.4f}")
    print(f"  约束违反: {update_info['constraint_violation']:.4f}")
    assert update_info['new_lambda'] >= update_info['old_lambda'], "成本高于阈值时，λ应该增大"

    print("\n✅ λ自适应更新检查通过!")

    print("\n" + "=" * 80)
    print("✅ LagrangianOptimizer 测试通过!")
    print("=" * 80)


def test_constrained_ppo_optimizer():
    """测试约束PPO优化器"""
    print("\n" + "=" * 80)
    print("测试 ConstrainedPPOOptimizer")
    print("=" * 80)

    # 创建组件
    cost_critic = create_cost_critic(device='cuda')
    lagrangian = create_lagrangian_optimizer()

    constrained_optimizer = ConstrainedPPOOptimizer(
        cost_critic=cost_critic,
        lagrangian=lagrangian,
        clip_param=0.2,
        value_loss_coef=0.5,
        cost_loss_coef=0.5,
        entropy_coef=0.01
    )

    # 创建测试数据
    batch_size = 4
    obs_dim = 321
    action_dim = 64

    obs = torch.randn(batch_size, obs_dim).cuda()
    actions = torch.randn(batch_size, action_dim).cuda()
    old_log_probs = torch.randn(batch_size).cuda()
    advantages = torch.randn(batch_size).cuda()
    returns = torch.randn(batch_size).cuda()
    costs = torch.rand(batch_size).cuda() * 0.8

    # 定义mock policy函数
    def mock_policy_fn(obs):
        return {
            'log_prob': torch.randn(batch_size).cuda(),
            'entropy': torch.randn(batch_size).cuda(),
            'embeddings': torch.randn(batch_size, 32, 64).cuda()
        }

    # 计算约束损失
    print("\n计算约束PPO损失...")
    losses = constrained_optimizer.compute_constrained_loss(
        obs=obs,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=advantages,
        returns=returns,
        costs=costs,
        policy_fn=mock_policy_fn
    )

    print(f"\n损失:")
    print(f"  total: {losses['total'].item():.4f}")
    print(f"  policy: {losses['policy'].item():.4f}")
    print(f"  value: {losses['value'].item():.4f}")
    print(f"  cost: {losses['cost'].item():.4f}")
    print(f"  entropy: {losses['entropy'].item():.4f}")

    # 验证损失不为NaN
    for key, loss in losses.items():
        assert not torch.isnan(loss), f"{key}_loss是NaN"

    print("\n✅ 约束PPO损失计算检查通过!")

    # 测试梯度回传
    print("\n测试梯度回传...")
    total_loss = losses['total']
    total_loss.backward()

    assert cost_critic.value_head[-1].weight.grad is not None, "value_head梯度不存在"
    # cost_head的最后一层是Softplus，前一层是Linear
    assert cost_critic.cost_head[-2].weight.grad is not None, "cost_head梯度不存在"

    print("✅ 梯度回传检查通过!")

    print("\n" + "=" * 80)
    print("✅ ConstrainedPPOOptimizer 测试通过!")
    print("=" * 80)


if __name__ == "__main__":
    print("\n开始测试 CostCritic 实现...\n")

    try:
        test_cost_critic_forward()
        test_cost_critic_loss()
        test_lagrangian_optimizer()
        test_constrained_ppo_optimizer()

        print("\n" + "=" * 80)
        print("🎉 所有测试通过!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
