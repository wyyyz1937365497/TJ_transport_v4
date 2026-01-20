"""
测试智能车辆选择训练流程

测试内容：
1. 环境初始化
2. 神经网络评分器
3. 启发式选择器
4. OCR奖励计算
5. Stage 1训练（少量数据）
6. Stage 2训练（少量episodes）
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

import yaml
import torch
import numpy as np

from src.env.gym_wrapper import make_gym_env
from src.models.neural_vehicle_scorer import create_neural_icv_scorer
from train_smart_selector import (
    HeuristicVehicleSelector,
    HeuristicController,
    Stage1Trainer,
    Stage2Trainer
)

def test_environment():
    """测试环境初始化"""
    print("\n" + "="*70)
    print("测试1: 环境初始化")
    print("="*70)
    
    with open('configs/phase1_lite.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        env = make_gym_env(config=config, seed=42, device='cuda')
        print("✅ 环境创建成功")
        
        obs, info = env.reset()
        print(f"✅ 环境重置成功")
        print(f"   观测键: {obs.keys()}")
        print(f"   车辆数: {len(obs.get('vehicle_ids', []))}")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        return False


def test_neural_scorer():
    """测试神经网络评分器"""
    print("\n" + "="*70)
    print("测试2: 神经网络评分器")
    print("="*70)
    
    try:
        model = create_neural_icv_scorer(
            node_dim=9,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            device='cuda'
        )
        print("✅ 神经网络评分器创建成功")
        
        # 创建模拟车辆状态
        vehicle_states = {}
        for i in range(10):
            vehicle_states[f'icv_{i}'] = {
                's': 1000.0 + i * 10,
                'd': 0.0,
                'speed': 20.0,
                'acceleration': 0.0,
                'lane_index': 0,
                'angle': 0.0,
                'vs': 20.0,
                'vd': 0.0,
                'is_icv': True
            }
        
        scores = model.compute_scores(vehicle_states)
        print(f"✅ 评分计算成功")
        print(f"   评分数: {len(scores)}")
        print(f"   评分范围: {min(scores.values()):.2f} - {max(scores.values()):.2f}")
        
        return True
    except Exception as e:
        print(f"❌ 神经网络评分器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_heuristic_selector():
    """测试启发式选择器"""
    print("\n" + "="*70)
    print("测试3: 启发式选择器和控制器")
    print("="*70)
    
    with open('configs/phase1_lite.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        selector = HeuristicVehicleSelector(config)
        controller = HeuristicController(config)
        print("✅ 启发式模块创建成功")
        
        # 创建模拟车辆状态
        vehicle_states = {}
        icv_ids = []
        for i in range(20):
            veh_id = f'icv_{i}'
            vehicle_states[veh_id] = {
                's': 1000.0 + i * 50,  # 有些在瓶颈区域
                'd': 0.0,
                'speed': 15.0 + i * 0.5,
                'acceleration': 0.0,
                'lane_index': i % 3,
                'lead_distance': 30.0 + i * 5,
                'max_speed': 33.33
            }
            icv_ids.append(veh_id)
        
        # 测试选择
        selected = selector.select_vehicles(vehicle_states, icv_ids, top_k=5)
        print(f"✅ 车辆选择成功")
        print(f"   选中车辆: {len(selected)}/{len(icv_ids)}")
        print(f"   选中ID: {selected[:3]}...")
        
        # 测试控制
        actions = controller.compute_actions(vehicle_states, selected)
        print(f"✅ 动作生成成功")
        print(f"   动作数: {len(actions)}")
        if len(actions) > 0:
            first_action = list(actions.values())[0]
            print(f"   示例动作: accel={first_action[0]:.2f}, lc={first_action[1]}")
        
        return True
    except Exception as e:
        print(f"❌ 启发式模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stage1_training():
    """测试Stage 1训练（少量数据）"""
    print("\n" + "="*70)
    print("测试4: Stage 1训练（少量数据）")
    print("="*70)
    
    with open('configs/phase1_lite.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改为少量数据
    config['training']['stage1']['num_episodes'] = 2
    config['training']['stage1']['epochs'] = 2
    
    try:
        env = make_gym_env(config=config, seed=42, device='cuda')
        model = create_neural_icv_scorer(
            node_dim=9,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            device='cuda'
        )
        
        trainer = Stage1Trainer(model, config, 'cuda')
        print("✅ Stage 1训练器创建成功")
        
        # 只收集少量数据
        num_samples = trainer.collect_data(env)
        print(f"✅ 数据收集完成: {num_samples} 样本")
        
        # 训练一个epoch
        if num_samples > 0:
            loss = trainer.train_epoch(0)
            print(f"✅ 训练完成: loss={loss:.4f}")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Stage 1训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stage2_training():
    """测试Stage 2训练（少量episodes）"""
    print("\n" + "="*70)
    print("测试5: Stage 2训练（少量episodes）")
    print("="*70)
    
    with open('configs/phase1_lite.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改为少量episodes
    config['training']['stage2']['num_episodes'] = 2
    
    try:
        env = make_gym_env(config=config, seed=42, device='cuda')
        model = create_neural_icv_scorer(
            node_dim=9,
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            device='cuda'
        )
        
        trainer = Stage2Trainer(model, config, 'cuda')
        print("✅ Stage 2训练器创建成功")
        
        # 收集一个episode
        episode_result = trainer.collect_episode(env)
        print(f"✅ Episode收集完成")
        print(f"   奖励: {episode_result['episode_reward']:.4f}")
        print(f"   得分: {episode_result['final_score']:.4f}")
        print(f"   步数: {episode_result['num_steps']}")
        
        # 测试策略更新
        loss = trainer.update_policy([episode_result])
        print(f"✅ 策略更新完成: loss={loss:.4f}")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Stage 2训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("智能车辆选择训练流程测试")
    print("="*70)
    
    tests = [
        ("环境初始化", test_environment),
        ("神经网络评分器", test_neural_scorer),
        ("启发式选择器", test_heuristic_selector),
        ("Stage 1训练", test_stage1_training),
        ("Stage 2训练", test_stage2_training),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ 测试 '{name}' 出现异常: {e}")
            results[name] = False
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:30s}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统准备就绪。")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")


if __name__ == '__main__':
    main()
