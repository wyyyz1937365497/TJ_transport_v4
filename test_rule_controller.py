#!/usr/bin/env python3
"""
测试规则控制器性能（600步快速测试）
"""
import sys
sys.path.append('/home/wyyyz/TJ_transport_v4')

import yaml
import numpy as np
from src.env.competition_env import CompetitionSumoEnv
sys.path.append('/home/wyyyz/TJ_transport_v4/src')
from controllers.simple_rule_controller import AdaptiveSpeedController
import logging

logging.basicConfig(level=logging.ERROR)

# Load config
with open('configs/mpc.yaml', 'r') as f:
    config = yaml.safe_load(f)

env_config = config['environment'].copy()
env_config['max_steps'] = 600

# Test 1: Baseline
print("=" * 80)
print("600步对比测试")
print("=" * 80)

print("\n【测试1】Baseline (无控制)")
print("-" * 80)
env_baseline = CompetitionSumoEnv(config=env_config, use_gui=False, device='cpu')
obs = env_baseline.reset()

for step in range(600):
    obs, _, done, info = env_baseline.step({})
    if step % 200 == 0:
        ocr = info.get('arrived_count', 0) / info.get('departed_count', 1) if info.get('departed_count', 0) > 0 else 0
        print(f"  Step {step:3d}: Arrived={info.get('arrived_count', 0):2d}, "
              f"Departed={info.get('departed_count', 0):3d}, OCR={ocr:.3f}")
    if done:
        break

baseline_ocr = info.get('arrived_count', 0) / info.get('departed_count', 1) if info.get('departed_count', 0) > 0 else 0.0
print(f"\n  Final OCR: {baseline_ocr:.4f} ({baseline_ocr*100:.2f}%)")
env_baseline.close()

# Test 2: Rule-based controller
print("\n【测试2】自适应速度规则控制")
print("-" * 80)
env_rule = CompetitionSumoEnv(config=env_config, use_gui=False, device='cpu')

controller_config = {
    'bottleneck_s_min': 1200.0,
    'bottleneck_s_max': 2200.0
}
controller = AdaptiveSpeedController(controller_config)

obs = env_rule.reset()
control_count = 0

for step in range(600):
    vehicle_ids = obs.get('vehicle_ids', [])
    icv_ids = list(obs.get('icv_ids', set()))

    actions_dict = controller.compute_actions(obs, vehicle_ids, icv_ids)
    if len(actions_dict) > 0:
        control_count += 1

    obs, _, done, info = env_rule.step(actions_dict)

    if step % 200 == 0:
        ocr = info.get('arrived_count', 0) / info.get('departed_count', 1) if info.get('departed_count', 0) > 0 else 0
        print(f"  Step {step:3d}: Arrived={info.get('arrived_count', 0):2d}, "
              f"Departed={info.get('departed_count', 0):3d}, OCR={ocr:.3f}, "
              f"Controlled={len(actions_dict):2d}")
    if done:
        break

rule_ocr = info.get('arrived_count', 0) / info.get('departed_count', 1) if info.get('departed_count', 0) > 0 else 0.0
print(f"\n  Final OCR: {rule_ocr:.4f} ({rule_ocr*100:.2f}%)")
print(f"  控制步数: {control_count}/600 ({control_count/600*100:.1f}%)")
env_rule.close()

# Comparison
print("\n" + "=" * 80)
print("性能对比")
print("=" * 80)
print(f"  Baseline:  OCR={baseline_ocr:.4f} ({baseline_ocr*100:.2f}%)")
print(f"  Rule-based: OCR={rule_ocr:.4f} ({rule_ocr*100:.2f}%)")

if rule_ocr > baseline_ocr:
    improvement = (rule_ocr - baseline_ocr) / baseline_ocr * 100
    print(f"\n  ✅ 规则控制改善了性能！提升: {improvement:.1f}%")
elif rule_ocr >= baseline_ocr * 0.98:
    print(f"\n  ⚠️  规则控制略低于baseline，但在可接受范围内（<2%下降）")
else:
    decline = (baseline_ocr - rule_ocr) / baseline_ocr * 100
    print(f"\n  ❌ 规则控制损害了性能！下降: {decline:.1f}%")
    print("  建议：调整控制参数或使用baseline")

print("\n" + "=" * 80)
