#!/usr/bin/env python3
"""
快速Baseline评估 - 无ICV控制

极简版本，不依赖torch和项目代码
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

# SUMO配置
SUMO_HOME = os.environ.get("SUMO_HOME", "/home/wyyyz/build/sumo")
if SUMO_HOME + "/tools" not in sys.path:
    sys.path.append(SUMO_HOME + "/tools")

import traci
from sumolib import checkBinary


def run_baseline(sumo_cfg, max_steps=3600):
    """运行baseline仿真（无控制）"""
    print("=" * 70)
    print("Baseline评估 - 无ICV控制")
    print("=" * 70)

    # 启动SUMO
    sumo_binary = checkBinary('sumo')
    traci.start([
        sumo_binary,
        "-c", sumo_cfg,
        "--no-step-log", "true",
        "--no-warnings", "true"
    ])

    print(f"✓ SUMO已启动")

    # 统计
    all_departed = set()
    all_arrived = set()
    step = 0

    try:
        while step < max_steps and traci.simulation.getMinExpectedNumber() > 0:
            # 收集统计
            departed = traci.simulation.getDepartedIDList()
            arrived = traci.simulation.getArrivedIDList()

            all_departed.update(departed)
            all_arrived.update(arrived)

            # 每1000步输出
            if step % 1000 == 0 and step > 0:
                active = len(traci.vehicle.getIDList())
                print(f"[步骤 {step}] 活跃: {active}, "
                      f"累计出发: {len(all_departed)}, "
                      f"累计到达: {len(all_arrived)}")

            traci.simulationStep()
            step += 1

    except KeyboardInterrupt:
        print("\n⚠️  仿真被中断")

    finally:
        traci.close()
        print(f"\n✓ SUMO已关闭 (运行{step}步)")

    # 计算结果
    departed_count = len(all_departed)
    arrived_count = len(all_arrived)
    ocr = arrived_count / departed_count if departed_count > 0 else 0.0

    results = {
        'ocr': ocr,
        'departed_count': departed_count,
        'arrived_count': arrived_count,
        'total_steps': step
    }

    return results


def main():
    """主函数"""
    sumo_cfg = "仿真环境_初赛_1.0/仿真环境-初赛/sumo_train.sumocfg"
    max_steps = 3600

    results = run_baseline(sumo_cfg, max_steps)

    # 打印结果
    print("\n" + "=" * 70)
    print("Baseline评估结果")
    print("=" * 70)
    print(f"\n📊 核心指标:")
    print(f"  OCR: {results['ocr']:.4f} ({results['arrived_count']}/{results['departed_count']})")
    print(f"  累计出发: {results['departed_count']}")
    print(f"  累计到达: {results['arrived_count']}")
    print(f"  运行步数: {results['total_steps']}")

    # 保存到JSON
    output_file = "competition_results/baseline_results.json"
    os.makedirs("competition_results", exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ 结果已保存到: {output_file}")
    print(f"\n📌 Baseline OCR: {results['ocr']:.4f}")
    print(f"   用法: python scripts/evaluate_ocr_max.py --baseline_ocr {results['ocr']:.4f}\n")


if __name__ == '__main__':
    main()
