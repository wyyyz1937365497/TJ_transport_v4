"""
简单测试脚本 - 检查SUMO是否正常工作
"""

import sys
import time
import yaml
import numpy as np
from src.env.sumo_env import SumoEnvironment

try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False


def load_config(config_path: str):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def test_sumo_simulation():
    """测试SUMO仿真是否能正常运行"""

    print("=" * 60)
    print("🧪 SUMO仿真测试")
    print("=" * 60)

    # 加载配置
    print("\n📋 加载配置...")
    config = load_config("wsl/config/base.yaml")

    # 获取环境配置
    env_config = config.get('environment', {})

    # 简化配置用于测试
    test_config = env_config.copy()
    test_config['max_steps'] = 100
    test_config['use_gui'] = False
    test_config['port'] = 8899

    print(f"   - SUMO配置: {test_config.get('sumo_cfg', 'N/A')}")
    print(f"   - 最大步数: {test_config['max_steps']}")
    print(f"   - GUI: {test_config['use_gui']}")

    # 创建环境
    print("\n🔧 创建SUMO环境...")
    try:
        env = SumoEnvironment(
            config=test_config,
            use_gui=False,
            port=8899,
            disable_port_retry=True
        )
        print("✅ 环境创建成功")
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 重置环境
    print("\n🔄 重置环境...")
    try:
        obs = env.reset()
        print(f"✅ 环境重置成功")
        print(f"   - 车辆数量: {len(obs.get('vehicle_states', {}))}")
        print(f"   - ICV数量: {len(obs.get('icv_ids', set()))}")
        print(f"   - 全局统计形状: {obs.get('global_stats', np.array([])).shape}")
    except Exception as e:
        print(f"❌ 环境重置失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 运行几步测试
    print("\n🏃 运行仿真测试...")
    step_times = []

    for step in range(10):
        start_time = time.time()

        # 生成随机动作
        vehicle_ids = list(obs.get('icv_ids', set()))
        actions = {}
        for veh_id in vehicle_ids:
            actions[veh_id] = np.array([
                np.random.uniform(-2.0, 2.0),  # 加速度
                np.random.choice([0.0, 1.0])    # 换道
            ])

        # 执行一步
        try:
            obs, reward, done, info = env.step(actions)
            step_time = time.time() - start_time
            step_times.append(step_time)

            if step % 2 == 0:
                print(f"   步骤 {step+1}: 车辆数={len(obs.get('vehicle_states', {}))}, "
                      f"耗时={step_time:.3f}s, 奖励={reward:.4f}, done={done}")

        except traci.exceptions.FatalTraCIError as e:
            if "Connection already closed" in str(e):
                print(f"   ⚠️  SUMO连接已关闭 (步骤 {step+1})")
                print(f"   这通常表示仿真已结束或车辆全部离开")
                break
            else:
                print(f"❌ 步骤 {step+1} 失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        except Exception as e:
            print(f"❌ 步骤 {step+1} 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        if done:
            print(f"   仿真在步骤 {step+1} 结束 (done=True)")
            break

    # 统计结果
    print("\n📊 测试结果统计:")
    print(f"   - 成功运行步数: {len(step_times)}")
    print(f"   - 平均每步耗时: {np.mean(step_times):.4f}s")
    print(f"   - 最快步: {np.min(step_times):.4f}s")
    print(f"   - 最慢步: {np.max(step_times):.4f}s")
    print(f"   - CPU使用率: {'正常' if np.mean(step_times) < 1.0 else '可能有问题'}")

    # 关闭环境
    print("\n🔚 关闭环境...")
    env.close()
    print("✅ 测试完成")

    return True


if __name__ == "__main__":
    try:
        success = test_sumo_simulation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
