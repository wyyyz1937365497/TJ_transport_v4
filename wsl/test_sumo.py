#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
from src.utils import load_config
from src.env.sumo_env import SumoEnvironment

print('✅ 测试SUMO环境...\n')

config = load_config('config/base.yaml')
env_config = config.environment

env = SumoEnvironment(env_config, use_gui=False, port=8999)
obs = env.reset()

total_vehicles = set()
max_steps = 100  # 增加到100步

print(f'开始仿真（最多{max_steps}步）...')
print('注意：早期车辆可能还没出发，需要等待...\n')

# 等待车辆出现
for step in range(max_steps):
    result = env.step()
    obs = result.observation
    done = result.done

    current_vehicles = set(obs.vehicle_ids)
    total_vehicles.update(current_vehicles)

    # 每10步打印一次，或者有车辆时
    if step % 10 == 0 or len(current_vehicles) > 0 or done:
        print(f'步数 {step}: 车辆 {len(current_vehicles)}, 累计 {len(total_vehicles)}, done={done}')

    if done:
        print(f'\n仿真在步数 {step} 结束')
        break

env.close()

print(f'\n✅ 测试成功!')
print(f'   总车辆数: {len(total_vehicles)}')
print(f'   总步数: {step + 1}')
print(f'\n🎉 SUMO环境已就绪!')
