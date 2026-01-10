"""
测试并行数据收集
验证多进程SUMO数据收集是否正常工作
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.env import collect_parallel_data_optimized

def test_parallel_collection():
    """测试并行数据收集"""
    print("="*70)
    print("🧪 测试并行数据收集")
    print("="*70)

    # 配置
    config = {
        "sumo_cfg": "仿真环境_初赛_1.0/仿真环境-初赛/sumo.sumocfg",
        "step_length": 0.1,
        "max_steps": 500,  # 较短的测试
        "control_ratio": 0.25,
        "seed": 42
    }

    print("\n配置:")
    print(f"  - SUMO配置: {config['sumo_cfg']}")
    print(f"  - 最大步数: {config['max_steps']}")
    print(f"  - 超时时间: 120s")

    # 收集数据
    print("\n开始收集...")
    trajectories, stats = collect_parallel_data_optimized(
        config=config,
        num_episodes=3,  # 测试3个episodes
        max_steps=500,
        timeout=120,  # 2分钟超时
        num_workers=None,  # 自动检测
        output_dir="data"
    )

    # 打印结果
    print("\n" + "="*70)
    print("📊 收集结果")
    print("="*70)
    print(f"总车辆数: {stats['total_vehicles']}")
    print(f"总步数: {stats['total_steps']}")
    print(f"成功率: {stats['successful_episodes']}/{stats['total_episodes']}")
    print(f"总耗时: {stats['collection_time']:.1f}s")
    print(f"吞吐量: {stats['throughput']:.2f} episodes/s")

    # 检查数据
    if len(trajectories) > 0:
        print(f"\n✅ 成功收集 {len(trajectories)} 辆车的数据")

        # 显示前几个轨迹
        sample_vehicles = list(trajectories.keys())[:3]
        for veh_id in sample_vehicles:
            traj = trajectories[veh_id]
            print(f"\n车辆 {veh_id}:")
            print(f"  - 记录点数: {len(traj['timestamps'])}")
            print(f"  - 时间范围: {traj['timestamps'][0]:.1f}s - {traj['timestamps'][-1]:.1f}s")
            print(f"  - 平均速度: {sum(traj['speeds'])/len(traj['speeds']):.2f} m/s")
    else:
        print("\n❌ 没有收集到任何数据")
        return False

    print("\n✅ 测试通过!")
    return True


if __name__ == "__main__":
    try:
        success = test_parallel_collection()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
