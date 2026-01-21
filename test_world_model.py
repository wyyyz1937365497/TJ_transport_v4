"""
测试WorldModel实现

验证点：
1. 输出维度正确性
2. 梯度回传
3. 损失计算
4. 多步预测
"""

import torch
import torch.nn as nn
import sys
sys.path.append('.')

from src.models.world_model import WorldModel, create_world_model

def test_world_model_forward():
    """测试WorldModel前向传播"""
    print("=" * 80)
    print("测试 WorldModel 前向传播")
    print("=" * 80)

    # 创建模型
    model = create_world_model(
        hidden_dim=64,
        latent_dim=64,
        num_layers=2,
        num_vehicles=32,
        device='cuda'
    )

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建测试数据
    batch_size = 4
    num_vehicles = 32
    hidden_dim = 64

    embeddings = torch.randn(batch_size, num_vehicles, hidden_dim).cuda()

    print(f"\n输入:")
    print(f"  embeddings: {embeddings.shape}")

    # 前向传播（无隐状态）
    print("\n执行前向传播（无隐状态）...")
    outputs = model(embeddings)

    # 验证输出
    print("\n输出:")
    assert 'z_flow' in outputs, "缺少 'z_flow' 输出"
    assert 'z_risk' in outputs, "缺少 'z_risk' 输出"
    assert 'pred_next_states' in outputs, "缺少 'pred_next_states' 输出"
    assert 'risk_prob' in outputs, "缺少 'risk_prob' 输出"
    assert 'hidden' in outputs, "缺少 'hidden' 输出"

    print(f"  z_flow: {outputs['z_flow'].shape} (期望: [{batch_size}, 64])")
    print(f"  z_risk: {outputs['z_risk'].shape} (期望: [{batch_size}, 64])")
    print(f"  pred_next_states: {outputs['pred_next_states'].shape} (期望: [{batch_size}, 32, 9])")
    print(f"  risk_prob: {outputs['risk_prob'].shape} (期望: [{batch_size}, 32])")
    print(f"  hidden: {outputs['hidden'].shape} (期望: [2, {batch_size}, 64])")

    assert outputs['z_flow'].shape == (batch_size, 64), f"z_flow维度错误"
    assert outputs['z_risk'].shape == (batch_size, 64), f"z_risk维度错误"
    assert outputs['pred_next_states'].shape == (batch_size, 32, 9), f"pred_next_states维度错误"
    assert outputs['risk_prob'].shape == (batch_size, 32), f"risk_prob维度错误"
    assert outputs['hidden'].shape == (2, batch_size, 64), f"hidden维度错误"

    print("\n✅ 维度检查通过!")

    # 测试有隐状态的前向传播
    print("\n执行前向传播（有隐状态）...")
    prev_hidden = outputs['hidden']
    outputs2 = model(embeddings, prev_hidden)

    assert outputs2['hidden'].shape == (2, batch_size, 64), "hidden维度错误"

    print("✅ 有隐状态的前向传播通过!")

    # 测试数值稳定性
    print("\n测试数值稳定性...")
    assert not torch.isnan(outputs['z_flow']).any(), "z_flow包含NaN"
    assert not torch.isnan(outputs['z_risk']).any(), "z_risk包含NaN"
    assert not torch.isnan(outputs['pred_next_states']).any(), "pred_next_states包含NaN"
    assert not torch.isnan(outputs['risk_prob']).any(), "risk_prob包含NaN"

    assert torch.all(outputs['risk_prob'] >= 0) & torch.all(outputs['risk_prob'] <= 1), "risk_prob不在[0,1]范围"

    print("✅ 数值稳定性检查通过!")

    print("\n" + "=" * 80)
    print("✅ WorldModel 前向传播测试通过!")
    print("=" * 80)


def test_world_model_loss():
    """测试WorldModel损失计算"""
    print("\n" + "=" * 80)
    print("测试 WorldModel 损失计算")
    print("=" * 80)

    # 创建模型
    model = create_world_model(device='cuda')

    # 创建测试数据
    batch_size = 4
    num_vehicles = 32

    embeddings = torch.randn(batch_size, num_vehicles, 64).cuda()
    true_next_states = torch.randn(batch_size, num_vehicles, 9).cuda()
    true_risk_labels = torch.randint(0, 2, (batch_size, num_vehicles)).float().cuda()

    # 前向传播
    outputs = model(embeddings)

    # 计算损失
    print("\n计算损失...")
    total_loss, loss_dict = model.compute_loss(
        pred_next_states=outputs['pred_next_states'],
        true_next_states=true_next_states,
        pred_risk_prob=outputs['risk_prob'],
        true_risk_labels=true_risk_labels
    )

    print(f"\n损失:")
    print(f"  total_loss: {total_loss.item():.4f}")
    print(f"  flow_loss: {loss_dict['flow'].item():.4f}")
    print(f"  risk_loss: {loss_dict['risk'].item():.4f}")

    assert not torch.isnan(total_loss), "total_loss是NaN"
    assert not torch.isnan(loss_dict['flow']), "flow_loss是NaN"
    assert not torch.isnan(loss_dict['risk']), "risk_loss是NaN"

    print("\n✅ 损失计算检查通过!")

    # 测试梯度回传
    print("\n测试梯度回传...")
    total_loss.backward()

    # 检查关键组件的梯度
    assert model.encoder.gru.weight_hh_l0.grad is not None, "GRU梯度不存在"
    assert model.flow_decoder.decoder[-1].weight.grad is not None, "flow_decoder梯度不存在"
    assert model.risk_decoder.decoder[-1].weight.grad is not None, "risk_decoder梯度不存在"

    print("✅ 梯度回传检查通过!")

    print("\n" + "=" * 80)
    print("✅ WorldModel 损失计算测试通过!")
    print("=" * 80)


def test_world_model_trajectory():
    """测试WorldModel多步预测"""
    print("\n" + "=" * 80)
    print("测试 WorldModel 多步预测")
    print("=" * 80)

    # 创建模型
    model = create_world_model(device='cuda')

    # 创建测试数据
    T = 10  # 时间步
    batch_size = 4
    num_vehicles = 32

    embeddings_sequence = torch.randn(T, batch_size, num_vehicles, 64).cuda()

    print(f"\n输入:")
    print(f"  embeddings_sequence: {embeddings_sequence.shape} (T={T}, B={batch_size}, N={num_vehicles}, D=64)")

    # 多步预测
    print("\n执行多步预测...")
    trajectory_outputs = model.get_trajectory_predictions(embeddings_sequence)

    # 验证输出
    print("\n输出:")
    assert 'pred_states' in trajectory_outputs, "缺少 'pred_states' 输出"
    assert 'risk_probs' in trajectory_outputs, "缺少 'risk_probs' 输出"

    print(f"  pred_states: {trajectory_outputs['pred_states'].shape} (期望: [{T}, {batch_size}, {num_vehicles}, 9])")
    print(f"  risk_probs: {trajectory_outputs['risk_probs'].shape} (期望: [{T}, {batch_size}, {num_vehicles}])")

    assert trajectory_outputs['pred_states'].shape == (T, batch_size, num_vehicles, 9), f"pred_states维度错误"
    assert trajectory_outputs['risk_probs'].shape == (T, batch_size, num_vehicles), f"risk_probs维度错误"

    print("\n✅ 多步预测维度检查通过!")

    # 测试数值稳定性
    print("\n测试数值稳定性...")
    assert not torch.isnan(trajectory_outputs['pred_states']).any(), "pred_states包含NaN"
    assert not torch.isnan(trajectory_outputs['risk_probs']).any(), "risk_probs包含NaN"
    assert torch.all(trajectory_outputs['risk_probs'] >= 0) & torch.all(trajectory_outputs['risk_probs'] <= 1), "risk_probs不在[0,1]范围"

    print("✅ 数值稳定性检查通过!")

    print("\n" + "=" * 80)
    print("✅ WorldModel 多步预测测试通过!")
    print("=" * 80)


if __name__ == "__main__":
    print("\n开始测试 WorldModel 实现...\n")

    try:
        test_world_model_forward()
        test_world_model_loss()
        test_world_model_trajectory()

        print("\n" + "=" * 80)
        print("🎉 所有测试通过!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
