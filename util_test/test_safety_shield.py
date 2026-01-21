"""
测试SafetyShield实现

验证点：
1. Level 1物理限制裁剪
2. Level 2 TTC检查和制动
3. 完整过滤流程
4. 统计信息
"""

import torch
import sys
sys.path.append('.')

from src.models.safety_shield import SafetyShield, create_safety_shield

def test_level1_physical_clipping():
    """测试Level 1物理限制裁剪"""
    print("=" * 80)
    print("测试 SafetyShield Level 1: 物理限制裁剪")
    print("=" * 80)

    # 创建安全屏障
    shield = create_safety_shield()

    # 创建测试数据
    batch_size = 2
    num_vehicles = 4

    # 创建超出限制的加速度
    actions = torch.tensor([
        [[5.0, 0.5], [3.0, -0.3], [-5.0, 0.0], [1.0, 0.2]],  # Batch 1
        [[-6.0, 0.1], [0.5, 0.0], [4.5, -0.5], [-3.0, 0.3]]   # Batch 2
    ])  # [2, 4, 2]

    vehicle_states = torch.randn(batch_size, num_vehicles, 9)

    print(f"\n输入动作（加速度超出限制）:")
    print(f"  min_accel限制: {shield.min_accel} m/s²")
    print(f"  max_accel限制: {shield.max_accel} m/s²")
    print(f"  actions accel: {actions[:,:,0]}")

    # Level 1裁剪
    print("\n执行Level 1裁剪...")
    clipped_actions, clip_mask = shield.level1_physical_clipping(actions, vehicle_states)

    print(f"\n裁剪后动作:")
    print(f"  clipped_actions: {clipped_actions[:,:,0]}")
    print(f"  clip_mask: {clip_mask}")

    # 验证裁剪
    assert (clipped_actions[:,:,0] >= shield.min_accel).all(), "存在小于min_accel的加速度"
    assert (clipped_actions[:,:,0] <= shield.max_accel).all(), "存在大于max_accel的加速度"

    # 验证裁剪掩码
    expected_clips = torch.tensor([
        [True, True, True, False],  # Batch 1: 5.0, 3.0, -5.0被裁剪
        [True, False, True, False]   # Batch 2: -6.0, 4.5被裁剪
    ])
    assert torch.equal(clip_mask, expected_clips), "裁剪掩码不正确"

    print("\n✅ Level 1 物理限制裁剪测试通过!")

    print("\n" + "=" * 80)


def test_level2_ttc_check():
    """测试Level 2 TTC检查"""
    print("\n" + "=" * 80)
    print("测试 SafetyShield Level 2: TTC检查")
    print("=" * 80)

    # 创建安全屏障
    shield = create_safety_shield(ttc_threshold=2.0)

    # 创建测试数据
    batch_size = 1
    num_vehicles = 3

    # 构造场景：3辆车在同一车道，车辆1接近车辆2
    vehicle_states = torch.zeros(batch_size, num_vehicles, 9)
    # 车辆0: x=100m, vx=20 m/s
    vehicle_states[0, 0, 0] = 100.0  # x
    vehicle_states[0, 0, 2] = 20.0   # vx
    vehicle_states[0, 0, 6] = 0.0    # lane

    # 车辆1: x=140m, vx=15 m/s (前车，较慢)
    vehicle_states[0, 1, 0] = 140.0  # x
    vehicle_states[0, 1, 2] = 15.0   # vx
    vehicle_states[0, 1, 6] = 0.0    # lane

    # 车辆2: x=50m, vx=25 m/s (后方，更快)
    vehicle_states[0, 2, 0] = 50.0   # x
    vehicle_states[0, 2, 2] = 25.0   # vx
    vehicle_states[0, 2, 6] = 0.0    # lane

    actions = torch.zeros(batch_size, num_vehicles, 2)
    actions[0, :, 0] = 1.0  # 所有车辆加速

    print(f"\n场景: 3辆车同车道")
    print(f"  TTC阈值: {shield.ttc_threshold} s")
    print(f"  车辆0: x=100m, vx=20m/s")
    print(f"  车辆1: x=140m, vx=15m/s (前车)")
    print(f"  车辆2: x=50m, vx=25m/s")

    # Level 2 TTC检查
    print("\n执行Level 2 TTC检查...")
    safe_actions, brake_mask, ttc_info = shield.level2_ttc_check(actions, vehicle_states)

    print(f"\nTTC信息:")
    print(f"  min_ttc: {ttc_info['min_ttc']}")

    # 计算期望的TTC
    # 车辆0→车辆1: TTC = (140-100) / (20-15) = 40/5 = 8.0s (>阈值，不制动)
    # 车辆1→车辆2: 车辆2在后面，不考虑
    # 车辆2→车辆0: TTC = (100-50) / (25-20) = 50/5 = 10.0s (>阈值，不制动)

    print(f"\n制动掩码: {brake_mask}")

    # 验证TTC计算
    ttc_0_to_1 = (140.0 - 100.0) / (20.0 - 15.0)  # = 8.0s
    print(f"  车辆0→车辆1 TTC: {ttc_0_to_1:.2f}s (> {shield.ttc_threshold}s, 不制动)")

    # 在这个场景下，所有TTC都大于阈值，不应该触发制动
    assert not brake_mask.any(), "不应该触发制动"

    print("\n✅ Level 2 TTC检查测试通过!")

    # 测试高风险场景
    print("\n测试高风险场景...")
    vehicle_states[0, 1, 0] = 110.0  # 车辆1移动到110m
    vehicle_states[0, 1, 2] = 5.0    # 车辆1减速到5 m/s

    print(f"  新场景: 车辆1: x=110m, vx=5m/s (极慢前车)")

    safe_actions, brake_mask, ttc_info = shield.level2_ttc_check(actions, vehicle_states)

    print(f"  min_ttc: {ttc_info['min_ttc']}")
    print(f"  制动掩码: {brake_mask}")

    # TTC = (110-100) / (20-5) = 10/15 = 0.67s (< 阈值，应该制动)
    ttc_0_to_1_new = (110.0 - 100.0) / (20.0 - 5.0)  # ≈ 0.67s
    print(f"  车辆0→车辆1 TTC: {ttc_0_to_1_new:.2f}s (< {shield.ttc_threshold}s, 应该制动)")

    # 验证制动
    assert brake_mask[0, 0], "车辆0应该触发制动"
    assert safe_actions[0, 0, 0] == shield.min_decel, "车辆0应该紧急制动"

    print("\n✅ Level 2 高风险场景测试通过!")

    print("\n" + "=" * 80)


def test_full_filtering():
    """测试完整过滤流程"""
    print("\n" + "=" * 80)
    print("测试 SafetyShield 完整过滤流程")
    print("=" * 80)

    # 创建安全屏障
    shield = create_safety_shield()

    # 创建测试数据
    batch_size = 2
    num_vehicles = 4

    # 动作：包含物理违规
    actions = torch.randn(batch_size, num_vehicles, 2) * 5  # 放大随机范围

    vehicle_states = torch.zeros(batch_size, num_vehicles, 9)
    vehicle_states[:, :, 0] = torch.rand(batch_size, num_vehicles) * 100  # x
    vehicle_states[:, :, 2] = torch.rand(batch_size, num_vehicles) * 20 + 5  # vx
    vehicle_states[:, :, 6] = torch.randint(0, 4, (batch_size, num_vehicles)).float()  # lane

    print(f"\n输入:")
    print(f"  actions: {actions.shape}")
    print(f"  vehicle_states: {vehicle_states.shape}")

    # 完整过滤
    print("\n执行完整过滤（返回详细信息）...")
    result = shield.filter_actions(actions, vehicle_states, return_details=True)

    # 验证输出
    print(f"\n输出:")
    assert 'safe_actions' in result, "缺少 'safe_actions'"
    assert 'safety_reward' in result, "缺少 'safety_reward'"
    assert 'level1_mask' in result, "缺少 'level1_mask'"
    assert 'level2_mask' in result, "缺少 'level2_mask'"
    assert 'ttc_info' in result, "缺少 'ttc_info'"

    print(f"  safe_actions: {result['safe_actions'].shape}")
    print(f"  safety_reward: {result['safety_reward'].shape}")
    print(f"  level1_mask: {result['level1_mask'].shape}")
    print(f"  level2_mask: {result['level2_mask'].shape}")
    print(f"  ttc_info keys: {result['ttc_info'].keys()}")

    # 验证安全动作
    safe_actions = result['safe_actions']
    assert (safe_actions[:,:,0] >= shield.min_accel).all(), "存在小于min_accel的加速度"
    assert (safe_actions[:,:,0] <= shield.max_accel).all(), "存在大于max_accel的加速度"

    # 验证奖励范围
    safety_reward = result['safety_reward']
    assert safety_reward.shape == (batch_size,), "safety_reward维度错误"

    print(f"\n奖励统计:")
    print(f"  safety_reward: {safety_reward}")
    print(f"  最小奖励: {safety_reward.min().item():.4f}")
    print(f"  最大奖励: {safety_reward.max().item():.4f}")

    print("\n✅ 完整过滤流程测试通过!")

    print("\n" + "=" * 80)


def test_statistics():
    """测试统计信息"""
    print("\n" + "=" * 80)
    print("测试 SafetyShield 统计信息")
    print("=" * 80)

    # 创建安全屏障
    shield = create_safety_shield()

    # 重置统计
    shield.reset_stats()

    # 执行多次过滤
    for _ in range(10):
        actions = torch.randn(2, 4, 2) * 5
        vehicle_states = torch.zeros(2, 4, 9)
        vehicle_states[:, :, 0] = torch.rand(2, 4) * 100
        vehicle_states[:, :, 2] = torch.rand(2, 4) * 20 + 5
        vehicle_states[:, :, 6] = torch.randint(0, 4, (2, 4)).float()

        shield.filter_actions(actions, vehicle_states)

    # 获取统计
    stats = shield.get_stats()

    print(f"\n统计信息:")
    print(f"  total_checks: {stats['total_checks']}")
    print(f"  level1_clips: {stats['level1_clips']}")
    print(f"  level2_brakes: {stats['level2_brakes']}")
    print(f"  level1_rate: {stats['level1_rate']:.2%}")
    print(f"  level2_rate: {stats['level2_rate']:.2%}")

    assert stats['total_checks'] == 20, "total_checks错误"  # 10次 * 2个batch
    assert stats['level1_rate'] >= 0, "level1_rate错误"
    assert stats['level2_rate'] >= 0, "level2_rate错误"

    print("\n✅ 统计信息测试通过!")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("\n开始测试 SafetyShield 实现...\n")

    try:
        test_level1_physical_clipping()
        test_level2_ttc_check()
        test_full_filtering()
        test_statistics()

        print("\n" + "=" * 80)
        print("🎉 所有测试通过!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
