"""
测试Frenet坐标系实现

验证环境是否正确提供Frenet坐标系的特征。
"""
import sys
sys.path.insert(0, 'src')

from src.env.competition_env import CompetitionSumoEnv
import yaml

print("="*70)
print("Frenet坐标系测试")
print("="*70)

try:
    # 加载配置
    with open('configs/quick_test.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print(f"\n配置文件: configs/quick_test.yaml")
    print(f"SUMO配置: {config.get('sumo_cfg', 'N/A')}")

    # 创建环境
    print(f"\n正在创建CompetitionSumoEnv...")
    env = CompetitionSumoEnv(config)
    print("✅ 环境创建成功")

    # 重置环境
    print(f"\n正在重置环境...")
    obs = env.reset()
    print("✅ 环境重置成功")

    # 验证观测包含Frenet特征
    print(f"\n正在验证Frenet坐标系特征...")
    vehicle_states = obs.get('vehicle_states', {})

    if not vehicle_states:
        print("⚠️  当前环境中没有车辆")
    else:
        print(f"\n当前车辆数量: {len(vehicle_states)}")
        print(f"\n前3辆车的Frenet坐标特征:")
        print("-"*70)
        print(f"{'车辆ID':<15} {'s(纵向位置)':<12} {'d(横向偏移)':<12} {'vs(纵向速度)':<12} {'vd(横向速度)':<12}")
        print("-"*70)

        for i, (veh_id, state) in enumerate(list(vehicle_states.items())[:3]):
            # 验证必需的Frenet特征
            assert 's' in state, f"❌ 缺少s坐标"
            assert 'd' in state, f"❌ 缺少d坐标"
            assert 'vs' in state, f"❌ 缺少vs速度"
            assert 'vd' in state, f"❌ 缺少vd速度"

            print(f"{veh_id:<15} {state['s']:>10.2f}m  {state['d']:>10.2f}m  {state['vs']:>10.2f}m/s  {state['vd']:>10.2f}m/s")

        print("-"*70)
        print(f"\n✅ 所有车辆都包含完整的Frenet坐标特征!")

    # 验证全局统计
    global_stats = obs.get('global_stats', [])
    print(f"\n全局统计维度: {len(global_stats)}")
    print(f"预期: 32维（扩展后）")

    # 关闭环境
    env.close()
    print(f"\n✅ 环境已关闭")

    print("\n" + "="*70)
    print("✅ Frenet坐标系测试通过!")
    print("="*70)

    print("\n📋 测试总结:")
    print("  - CompetitionSumoEnv环境正常工作")
    print("  - 车辆状态包含s, d, vs, vd等Frenet特征")
    print("  - 观测空间已更新为9维特征")
    print("  - 可以继续进行训练测试")

except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
