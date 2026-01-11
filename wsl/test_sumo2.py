#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
from src.utils import load_config
from src.env.sumo_env import SumoEnvironment

with open('test_output.txt', 'w') as f:
    f.write('✅ 测试SUMO环境...\n\n')
    
    config = load_config('config/base.yaml')
    env_config = config.environment
    
    env = SumoEnvironment(env_config, use_gui=False, port=8999)
    obs = env.reset()
    
    total_vehicles = set()
    max_steps = 100
    
    f.write(f'开始仿真（最多{max_steps}步）...\n')
    f.write('注意：早期车辆可能还没出发，需要等待...\n\n')
    f.flush()
    
    for step in range(max_steps):
        result = env.step()
        obs = result.observation
        done = result.done
        
        current_vehicles = set(obs.vehicle_ids)
        total_vehicles.update(current_vehicles)
        
        if step % 10 == 0 or len(current_vehicles) > 0 or done:
            msg = f'步数 {step}: 车辆 {len(current_vehicles)}, 累计 {len(total_vehicles)}, done={done}\n'
            f.write(msg)
            f.flush()
        
        if done:
            f.write(f'\n仿真在步数 {step} 结束\n')
            break
    
    env.close()
    
    f.write(f'\n✅ 测试成功!\n')
    f.write(f'   总车辆数: {len(total_vehicles)}\n')
    f.write(f'   总步数: {step + 1}\n')
    f.write(f'\n🎉 SUMO环境已就绪!\n')

print('Test completed, check test_output.txt')
