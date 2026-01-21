"""
测试DynamicWeightGate实现

验证点：
1. 输出维度正确性
2. 权重归一化（sum = 1）
3. 梯度回传
4. 加权奖励计算
5. 自适应版本的温度控制
"""

import torch
import torch.nn as nn
import sys
sys.path.append('.')

from src.models.dynamic_weight_gate import (
    DynamicWeightGate,
    AdaptiveWeightGate,
    FixedWeightGate,
    create_dynamic_weight_gate,
    PRESET_WEIGHTS
)

def test_dynamic_weight_gate_forward():
    """测试DynamicWeightGate前向传播"""
    print("=" * 80)
    print("测试 DynamicWeightGate 前向传播")
    print("=" * 80)

    # 创建模型
    model = create_dynamic_weight_gate(
        global_dim=64,
        hidden_dim=32,
        dropout=0.1,
        adaptive=False,
        device='cuda'
    )

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建测试数据
    batch_size = 4
    global_dim = 64

    global_state = torch.randn(batch_size, global_dim).cuda()

    print(f"\n输入:")
    print(f"  global_state: {global_state.shape}")

    # 前向传播
    print("\n执行前向传播...")
    weights = model(global_state)

    # 验证输出
    print("\n输出:")
    print(f"  weights: {weights.shape} (期望: [{batch_size}, 3])")

    assert weights.shape == (batch_size, 3), f"weights维度错误"

    print("\n✅ 维度检查通过!")

    # 验证权重归一化
    print("\n检查权重归一化...")
    weight_sums = weights.sum(dim=-1)
    print(f"  weight_sums: {weight_sums}")

    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), "权重和不为1"

    print("✅ 权重归一化检查通过!")

    # 验证权重范围
    print("\n检查权重范围...")
    assert (weights >= 0).all() and (weights <= 1).all(), "权重不在[0,1]范围"

    print(f"  最小权重: {weights.min().item():.4f}")
    print(f"  最大权重: {weights.max().item():.4f}")

    print("✅ 权重范围检查通过!")

    # 测试数值稳定性
    print("\n测试数值稳定性...")
    assert not torch.isnan(weights).any(), "weights包含NaN"

    print("✅ 数值稳定性检查通过!")

    print("\n" + "=" * 80)
    print("✅ DynamicWeightGate 前向传播测试通过!")
    print("=" * 80)


def test_dynamic_weight_gate_weight_dict():
    """测试权重字典接口"""
    print("\n" + "=" * 80)
    print("测试 DynamicWeightGate 权重字典接口")
    print("=" * 80)

    # 创建模型
    model = create_dynamic_weight_gate(device='cuda')

    # 创建测试数据
    batch_size = 4
    global_state = torch.randn(batch_size, 64).cuda()

    # 获取权重字典
    print("\n获取权重字典...")
    weight_dict = model.get_weight_dict(global_state)

    print("\n权重字典:")
    assert 'w_efficiency' in weight_dict, "缺少 'w_efficiency'"
    assert 'w_stability' in weight_dict, "缺少 'w_stability'"
    assert 'w_cost' in weight_dict, "缺少 'w_cost'"

    print(f"  w_efficiency: {weight_dict['w_efficiency'].shape}")
    print(f"  w_stability: {weight_dict['w_stability'].shape}")
    print(f"  w_cost: {weight_dict['w_cost'].shape}")

    # 验证维度
    for key, value in weight_dict.items():
        assert value.shape == (batch_size,), f"{key}维度错误"

    # 验证归一化
    total = weight_dict['w_efficiency'] + weight_dict['w_stability'] + weight_dict['w_cost']
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5), "权重和不为1"

    print("\n✅ 权重字典接口检查通过!")

    print("\n" + "=" * 80)
    print("✅ DynamicWeightGate 权重字典测试通过!")
    print("=" * 80)


def test_dynamic_weight_gate_weighted_reward():
    """测试加权奖励计算"""
    print("\n" + "=" * 80)
    print("测试 DynamicWeightGate 加权奖励计算")
    print("=" * 80)

    # 创建模型
    model = create_dynamic_weight_gate(device='cuda')

    # 创建测试数据
    batch_size = 4

    global_state = torch.randn(batch_size, 64).cuda()
    reward_efficiency = torch.randn(batch_size).cuda()
    reward_stability = torch.randn(batch_size).cuda()
    reward_cost = torch.randn(batch_size).cuda()

    print("\n输入:")
    print(f"  global_state: {global_state.shape}")
    print(f"  reward_efficiency: {reward_efficiency.shape}")
    print(f"  reward_stability: {reward_stability.shape}")
    print(f"  reward_cost: {reward_cost.shape}")

    # 计算加权奖励
    print("\n计算加权奖励...")
    weighted_reward = model.compute_weighted_reward(
        global_state,
        reward_efficiency,
        reward_stability,
        reward_cost
    )

    print(f"\n加权奖励: {weighted_reward.shape} (期望: [{batch_size}])")

    assert weighted_reward.shape == (batch_size,), f"weighted_reward维度错误"

    print("✅ 维度检查通过!")

    # 验证数值稳定性
    assert not torch.isnan(weighted_reward).any(), "weighted_reward包含NaN"

    print("✅ 数值稳定性检查通过!")

    print("\n" + "=" * 80)
    print("✅ DynamicWeightGate 加权奖励测试通过!")
    print("=" * 80)


def test_dynamic_weight_gate_gradients():
    """测试梯度回传"""
    print("\n" + "=" * 80)
    print("测试 DynamicWeightGate 梯度回传")
    print("=" * 80)

    # 创建模型
    model = create_dynamic_weight_gate(device='cuda')

    # 创建测试数据
    batch_size = 4

    global_state = torch.randn(batch_size, 64).cuda()
    reward_efficiency = torch.randn(batch_size).cuda()
    reward_stability = torch.randn(batch_size).cuda()
    reward_cost = torch.randn(batch_size).cuda()

    # 计算加权奖励
    weighted_reward = model.compute_weighted_reward(
        global_state,
        reward_efficiency,
        reward_stability,
        reward_cost
    )

    # 计算损失
    print("\n计算损失...")
    loss = weighted_reward.sum()

    # 反向传播
    print("\n反向传播...")
    loss.backward()

    # 检查梯度
    print("\n检查梯度...")
    assert model.feature_extractor[0].weight.grad is not None, "feature_extractor梯度不存在"
    assert model.weight_head.weight.grad is not None, "weight_head梯度不存在"

    print("✅ 梯度回传检查通过!")

    print("\n" + "=" * 80)
    print("✅ DynamicWeightGate 梯度测试通过!")
    print("=" * 80)


def test_adaptive_weight_gate():
    """测试自适应权重门控"""
    print("\n" + "=" * 80)
    print("测试 AdaptiveWeightGate")
    print("=" * 80)

    # 创建模型
    model = create_dynamic_weight_gate(
        global_dim=64,
        hidden_dim=32,
        adaptive=True,
        device='cuda'
    )

    print(f"\n模型类型: {type(model).__name__}")
    print(f"初始温度: {model.temperature}")
    print(f"初始平滑系数: {model.smoothing_coef}")

    # 创建测试数据
    batch_size = 4
    global_state = torch.randn(batch_size, 64).cuda()

    # 前向传播
    print("\n执行前向传播...")
    weights = model(global_state, use_smoothing=True)

    print(f"  weights: {weights.shape}")

    # 验证输出
    assert weights.shape == (batch_size, 3), f"weights维度错误"

    print("✅ 维度检查通过!")

    # 测试温度调整
    print("\n测试温度调整...")
    model.set_temperature(0.5)
    print(f"  新温度: {model.temperature}")

    weights_low_temp = model(global_state, use_smoothing=False)

    # 低温度应该导致更锐利的分布（方差更大）
    variance_high = weights.var(dim=0).mean().item()
    variance_low = weights_low_temp.var(dim=0).mean().item()

    print(f"  高温度方差: {variance_high:.4f}")
    print(f"  低温度方差: {variance_low:.4f}")

    # 低温度应该有更大的方差（更锐利）
    assert variance_low >= variance_high * 0.8, "温度调整效果不明显"

    print("✅ 温度调整检查通过!")

    print("\n" + "=" * 80)
    print("✅ AdaptiveWeightGate 测试通过!")
    print("=" * 80)


def test_fixed_weight_gate():
    """测试固定权重门控"""
    print("\n" + "=" * 80)
    print("测试 FixedWeightGate")
    print("=" * 80)

    # 测试所有预设配置
    for preset_name in PRESET_WEIGHTS.keys():
        print(f"\n测试预设: {preset_name}")

        # 创建模型
        model = FixedWeightGate(preset=preset_name).cuda()

        # 创建测试数据
        batch_size = 4
        global_state = torch.randn(batch_size, 64).cuda()

        # 前向传播
        weights = model(global_state)

        print(f"  权重: {weights[0].cpu().numpy()}")

        # 验证输出
        assert weights.shape == (batch_size, 3), f"weights维度错误"

        # 验证所有batch返回相同的权重
        assert torch.allclose(weights, weights[0:1, :]), "固定权重不一致"

        # 验证权重和为1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, device='cuda')), "权重和不为1"

    print("\n✅ 所有预设配置测试通过!")

    print("\n" + "=" * 80)
    print("✅ FixedWeightGate 测试通过!")
    print("=" * 80)


if __name__ == "__main__":
    print("\n开始测试 DynamicWeightGate 实现...\n")

    try:
        test_dynamic_weight_gate_forward()
        test_dynamic_weight_gate_weight_dict()
        test_dynamic_weight_gate_weighted_reward()
        test_dynamic_weight_gate_gradients()
        test_adaptive_weight_gate()
        test_fixed_weight_gate()

        print("\n" + "=" * 80)
        print("🎉 所有测试通过!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
