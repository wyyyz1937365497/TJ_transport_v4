#!/usr/bin/env python3
"""
智能ICV管理器测试脚本

测试场景：
1. 正常情况：Top-K=5，控制少量车辆
2. 拥堵风险：Top-K=10，扩大控制范围
3. 紧急事件：Top-K=15，全面介入
4. 事件触发：TTC<2.0s立即触发
5. 安全屏障：动作裁剪和紧急制动

作者: TJ Transport Team v4.0
日期: 2026-01-19
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.env.smart_icv_manager import (
    SmartICVManager,
    VehicleRiskMetrics,
    InterventionLevel
)


def create_mock_vehicles(num_vehicles: int, avg_speed: float = 10.0, safe=True) -> list:
    """创建模拟车辆数据

    Args:
        num_vehicles: 车辆数量
        avg_speed: 平均速度
        safe: 是否生成安全车辆（避免高危事件）
    """
    vehicles = []
    np.random.seed(42)  # 固定随机种子，确保可重复

    for i in range(num_vehicles):
        if safe:
            # 安全模式：确保不会有高危车辆
            speed = avg_speed + np.random.randn() * 1.0  # 减小速度波动
            leader_speed = speed + np.random.randn() * 0.5  # 前车速度相近
            leader_distance = 30.0 + np.random.rand() * 20.0  # 足够的车距
        else:
            # 可能产生高危车辆
            speed = avg_speed + np.random.randn() * 2.0
            leader_speed = avg_speed + np.random.randn() * 2.0
            leader_distance = 20.0 + np.random.randn() * 5.0

        vehicles.append({
            'veh_id': f'veh_{i}',
            'position': (100.0 + i * 10.0, 50.0),
            'speed': speed,
            'lane_index': i % 3,
            'edge_id': 'edge_main',
            'acceleration': np.random.randn() * 0.5,
            'leader_speed': leader_speed,
            'leader_distance': leader_distance,
            'in_bottleneck': i < 5,  # 前5辆车在瓶颈区域
            'bottleneck_proximity': 1.0 if i < 5 else 0.0,
        })
    return vehicles


def test_scenario_1_normal():
    """测试场景1：正常情况（平峰期）"""
    print("\n" + "=" * 80)
    print("场景1：正常情况（平峰期）")
    print("=" * 80)

    # 初始化管理器
    manager = SmartICVManager(
        max_vehicles=50,
        default_top_k=5,
        emergency_top_k=15,
        elevated_top_k=10,
    )

    # 创建正常车流（平均速度15m/s，安全模式）
    vehicles = create_mock_vehicles(50, avg_speed=15.0, safe=True)

    # 模拟10步
    for step in range(10):
        selected_icvs, level = manager.update_icv_selection(
            current_step=step * 10,
            all_vehicles=vehicles,
            avg_speed=15.0,
            congestion_detected=False,
        )

        if step == 0:  # 第一次选择
            print(f"✅ Step {step * 10}: Level={level.value}, Top-K={len(selected_icvs)}")
            print(f"   选中的ICV: {list(selected_icvs)[:3]}...")

            # 验证
            assert level == InterventionLevel.NORMAL, "正常情况应为NORMAL级别"
            assert len(selected_icvs) == 5, "正常情况应选择5辆"
            print("   ✓ 测试通过")

    print("\n📊 统计信息:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")


def test_scenario_2_congestion():
    """测试场景2：拥堵风险"""
    print("\n" + "=" * 80)
    print("场景2：拥堵风险（平均速度<5m/s）")
    print("=" * 80)

    # 初始化管理器
    manager = SmartICVManager(
        max_vehicles=50,
        default_top_k=5,
        emergency_top_k=15,
        elevated_top_k=10,
    )

    # 创建拥堵车流（平均速度3m/s，安全模式）
    vehicles = create_mock_vehicles(50, avg_speed=3.0, safe=True)

    # 触发ICV选择
    selected_icvs, level = manager.update_icv_selection(
        current_step=0,
        all_vehicles=vehicles,
        avg_speed=3.0,
        congestion_detected=True,
    )

    print(f"✅ Step 0: Level={level.value}, Top-K={len(selected_icvs)}")
    print(f"   选中的ICV: {list(selected_icvs)[:3]}...")

    # 验证
    assert level == InterventionLevel.ELEVATED, "拥堵情况应为ELEVATED级别"
    assert len(selected_icvs) == 10, "拥堵情况应选择10辆"
    print("   ✓ 测试通过")


def test_scenario_3_emergency():
    """测试场景3：紧急事件（TTC<2.0s）"""
    print("\n" + "=" * 80)
    print("场景3：紧急事件（检测到极高危车辆）")
    print("=" * 80)

    # 初始化管理器
    manager = SmartICVManager(
        max_vehicles=50,
        default_top_k=5,
        emergency_top_k=15,
        elevated_top_k=10,
    )

    # 创建车流（包含极高危车辆）
    vehicles = create_mock_vehicles(50, avg_speed=10.0, safe=True)

    # 人为制造极高危车辆（TTC<2.0s）
    vehicles[0]['leader_distance'] = 5.0  # 距离前车5米
    vehicles[0]['speed'] = 20.0  # 速度20m/s
    vehicles[0]['leader_speed'] = 10.0  # 前车10m/s
    # TTC = 5 / (20-10) = 0.5s < 2.0s

    # 再制造一个高危车辆
    vehicles[1]['leader_distance'] = 8.0
    vehicles[1]['speed'] = 18.0
    vehicles[1]['leader_speed'] = 10.0
    # TTC = 8 / (18-10) = 1.0s < 2.0s

    # 触发ICV选择
    selected_icvs, level = manager.update_icv_selection(
        current_step=0,
        all_vehicles=vehicles,
        avg_speed=10.0,
        congestion_detected=False,
    )

    print(f"✅ Step 0: Level={level.value}, Top-K={len(selected_icvs)}")
    print(f"   选中的ICV: {list(selected_icvs)[:3]}...")

    # 验证
    assert level == InterventionLevel.EMERGENCY, "极高危情况应为EMERGENCY级别"
    assert len(selected_icvs) == 15, "紧急情况应选择15辆"
    print("   ✓ 测试通过")


def test_scenario_4_event_triggered():
    """测试场景4：事件触发机制"""
    print("\n" + "=" * 80)
    print("场景4：事件触发（检测到高危立即触发）")
    print("=" * 80)

    # 初始化管理器
    manager = SmartICVManager(
        max_vehicles=50,
        default_top_k=5,
        emergency_top_k=15,
        elevated_top_k=10,
        decision_interval=10,  # 10步决策周期
    )

    # 创建正常车流（安全模式）
    vehicles = create_mock_vehicles(50, avg_speed=15.0, safe=True)

    # Step 0-9：正常情况，不更新ICV选择
    for step in range(10):
        selected_icvs, level = manager.update_icv_selection(
            current_step=step,
            all_vehicles=vehicles,
            avg_speed=15.0,
            congestion_detected=False,
        )

        if step == 0:
            print(f"✅ Step {step}: Level={level.value}, Top-K={len(selected_icvs)} (首次选择)")
        else:
            assert len(selected_icvs) == 0, f"Step {step}: 不应该更新ICV选择"
            if step == 5:
                print(f"✅ Step {step}: 未触发更新（在决策周期内）✓")

    # Step 10：定时触发（到达决策周期）
    selected_icvs, level = manager.update_icv_selection(
        current_step=10,
        all_vehicles=vehicles,
        avg_speed=15.0,
        congestion_detected=False,
    )
    print(f"✅ Step 10: Level={level.value}, Top-K={len(selected_icvs)} (定时触发)")
    assert len(selected_icvs) > 0, "Step 10应该触发更新"
    print("   ✓ 定时触发正常")

    # Step 11：人为制造高危事件（立即触发）
    vehicles[0]['leader_distance'] = 3.0
    vehicles[0]['speed'] = 25.0
    vehicles[0]['leader_speed'] = 10.0

    # 由于决策周期是10步，Step 11本不应该更新
    # 但如果实现了事件触发机制，应该能立即检测到高危
    # 这需要在update_icv_selection中额外检测emergency_events
    # 当前实现中，事件触发通过should_update_icv_selection判断
    # 如果要实现立即触发，需要在step()方法中每步检测风险

    print("\n   ⚠️  注意：完整的事件触发机制需要在环境step()中实现")
    print("   当前版本仅支持定时触发（每10步）")


def test_scenario_5_safety_barrier():
    """测试场景5：两级安全屏障"""
    print("\n" + "=" * 80)
    print("场景5：两级安全屏障")
    print("=" * 80)

    # 初始化管理器
    manager = SmartICVManager(
        max_vehicles=50,
        ttc_threshold=2.0,
    )

    # 测试Level 1：规则卫士
    print("\n--- Level 1: 规则卫士（动作裁剪）---")
    rl_action = np.array([5.0, 1.5])  # 超出范围的动作 [accel, lane_change]
    risk_metrics = VehicleRiskMetrics(ttc=10.0, thw=2.0, drac=0.5, risk_level=0.1)

    corrected_action = manager.get_safety_barrier_action(
        veh_id='test_veh',
        rl_action=rl_action,
        risk_metrics=risk_metrics,
    )

    print(f"原始动作: {rl_action}")
    print(f"修正动作: {corrected_action}")
    assert corrected_action[0] == 2.0, "加速度应裁剪到2.0"
    assert corrected_action[1] == 1.0, "变道应裁剪到1.0"
    print("   ✓ Level 1测试通过")

    # 测试Level 2：紧急避险
    print("\n--- Level 2: 紧急避险（TTC<2.0s）---")
    rl_action = np.array([1.0, 0.0])  # 正常加速
    risk_metrics = VehicleRiskMetrics(ttc=1.5, thw=1.0, drac=5.0, risk_level=0.8)

    corrected_action = manager.get_safety_barrier_action(
        veh_id='test_veh',
        rl_action=rl_action,
        risk_metrics=risk_metrics,
    )

    print(f"原始动作: {rl_action}")
    print(f"修正动作: {corrected_action}")
    assert corrected_action[0] == -4.0, "应强制最大制动"
    assert corrected_action[1] == 0.0, "应禁止变道"
    assert corrected_action[2] == -1.0, "应标记为紧急制动"
    print("   ✓ Level 2测试通过")


def test_risk_metrics():
    """测试风险指标计算"""
    print("\n" + "=" * 80)
    print("测试：风险指标计算")
    print("=" * 80)

    manager = SmartICVManager(max_vehicles=50)

    # 测试案例1：极高危（TTC<2.0s）
    risk1 = manager.compute_risk_metrics(
        veh_id='veh1',
        speed=20.0,
        leader_speed=10.0,
        leader_distance=10.0,  # TTC = 10/(20-10) = 1.0s
        acceleration=0.0,
    )
    print(f"\n案例1（极高危）:")
    print(f"  TTC: {risk1.ttc:.2f}s")
    print(f"  THW: {risk1.thw:.2f}s")
    print(f"  DRAC: {risk1.drac:.2f}m/s²")
    print(f"  风险级别: {risk1.risk_level:.2f}")
    assert risk1.ttc < 2.0, "TTC应小于2.0s"
    assert risk1.risk_level > 0.5, "风险级别应较高"
    print("   ✓ 极高危检测正常")

    # 测试案例2：安全（TTC>10.0s）
    risk2 = manager.compute_risk_metrics(
        veh_id='veh2',
        speed=10.0,
        leader_speed=10.0,
        leader_distance=100.0,
        acceleration=0.0,
    )
    print(f"\n案例2（安全）:")
    print(f"  TTC: {risk2.ttc:.2f}s")
    print(f"  THW: {risk2.thw:.2f}s")
    print(f"  DRAC: {risk2.drac:.2f}m/s²")
    print(f"  风险级别: {risk2.risk_level:.2f}")
    assert risk2.ttc == float('inf'), "TTC应为无穷大"
    assert risk2.risk_level < 0.2, "风险级别应较低"
    print("   ✓ 安全检测正常")


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("智能ICV管理器测试套件")
    print("=" * 80)

    try:
        # 运行所有测试
        test_risk_metrics()
        test_scenario_1_normal()
        test_scenario_2_congestion()
        test_scenario_3_emergency()
        test_scenario_4_event_triggered()
        test_scenario_5_safety_barrier()

        # 总结
        print("\n" + "=" * 80)
        print("✅ 所有测试通过！")
        print("=" * 80)
        print("\n📝 测试覆盖:")
        print("   ✓ 风险指标计算（TTC/THW/DRAC）")
        print("   ✓ 正常情况：Top-K=5")
        print("   ✓ 拥堵风险：Top-K=10")
        print("   ✓ 紧急事件：Top-K=15")
        print("   ✓ 定时触发机制")
        print("   ✓ 两级安全屏障（规则卫士+紧急避险）")

        print("\n🚀 下一步:")
        print("   1. 在真实环境中测试智能ICV管理器")
        print("   2. 对比Top-K vs 固定25%的性能差异")
        print("   3. 验证事件触发机制的实时性")
        print("   4. 评估干预成本（P_int指标）的提升")

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
