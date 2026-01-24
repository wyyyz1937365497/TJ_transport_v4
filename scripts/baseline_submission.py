#!/usr/bin/env python3
"""
Baseline评估脚本 - 无ICV控制

用途：测量没有AI干预时的OCR，作为比赛baseline
基于submit_solution.py的框架，但禁用所有ICV控制

运行方式：
    python scripts/baseline_submission.py
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill

# SUMO配置
SUMO_HOME = os.environ.get("SUMO_HOME", "/home/wyyyz/build/sumo")
if SUMO_HOME + "/tools" not in sys.path:
    sys.path.append(SUMO_HOME + "/tools")

import traci
from sumolib import checkBinary

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.frenet_utils import get_frenet_system


class BaselineEvaluator:
    """
    Baseline评估器 - 无ICV控制

    模拟SUMO默认驾驶行为（无AI干预）
    """

    def __init__(self, sumo_cfg, max_steps=3600, use_gui=False):
        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.use_gui = use_gui

        # 数据存储
        self.step_data = []
        self.vehicle_data = []
        self.route_data = {}

        # 统计
        self.cumulative_departed = 0
        self.cumulative_arrived = 0
        self.all_departed_vehicles = set()
        self.all_arrived_vehicles = set()

        # Frenet坐标系
        self.frenet_system = None

        print("=" * 70)
        print("Baseline评估器 - 无ICV控制")
        print("=" * 70)
        print(f"SUMO配置: {sumo_cfg}")
        print(f"最大步数: {max_steps}")
        print(f"ICV控制: 禁用（baseline）")
        print("=" * 70)

    def initialize_environment(self):
        """初始化SUMO环境"""
        print("\n[初始化] 正在初始化SUMO环境...")

        # 解析配置文件
        cfg_dir = os.path.dirname(self.sumo_cfg)
        import xml.etree.ElementTree as ET
        tree = ET.parse(self.sumo_cfg)
        root = tree.getroot()

        net_file = None
        route_files = []

        for input_elem in root.findall('.//input'):
            net_elem = input_elem.find('net-file')
            if net_elem is not None:
                net_file = os.path.join(cfg_dir, net_elem.get('value'))

            route_elem = input_elem.find('route-files')
            if route_elem is not None:
                route_files.append(os.path.join(cfg_dir, route_elem.get('value')))

        # 初始化Frenet坐标系
        if net_file and os.path.exists(net_file):
            self.frenet_system = get_frenet_system(net_file)
            print(f"✓ Frenet坐标系初始化成功")

        print(f"✓ 环境初始化完成")

    def start_simulation(self):
        """启动SUMO仿真"""
        sumo_binary = checkBinary('sumo-gui' if self.use_gui else 'sumo')

        traci.start([
            sumo_binary,
            "-c", self.sumo_cfg,
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--collision.action", "warn",
            "--collision.check-junctions", "true"
        ])

        print(f"✓ SUMO已启动")

    def collect_step_data(self, step):
        """收集每步数据"""
        vehicle_ids = traci.vehicle.getIDList()

        # 统计
        departed = traci.simulation.getDepartedNumber()
        arrived = traci.simulation.getArrivedNumber()

        self.cumulative_departed += departed
        self.cumulative_arrived += arrived

        # 记录当前步的车辆
        current_departed = traci.simulation.getDepartedIDList()
        current_arrived = traci.simulation.getArrivedIDList()

        for veh_id in current_departed:
            self.all_departed_vehicles.add(veh_id)

        for veh_id in current_arrived:
            self.all_arrived_vehicles.add(veh_id)

        # 收集车辆数据
        for veh_id in vehicle_ids:
            try:
                lane_id = traci.vehicle.getLaneID(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                position = traci.vehicle.getPosition(veh_id)
                angle = traci.vehicle.getAngle(veh_id)
                route = traci.vehicle.getRouteID(veh_id)
                road_id = traci.vehicle.getRoadID(veh_id)

                # 获取OD信息（首次遇到时记录）
                if veh_id not in self.vehicle_od_data:
                    self.vehicle_od_data[veh_id] = {
                        'origin': road_id,
                        'route': route,
                        'depart_time': step
                    }

                # 计算Frenet坐标（如果可用）
                s, d = 0.0, 0.0
                if self.frenet_system and lane_id in self.frenet_system.lanes:
                    try:
                        x, y = position
                        s, d = self.frenet_system.cartesian_to_frenet(x, y, road_id, lane_id)
                    except:
                        pass

                # 记录车辆数据
                self.vehicle_data.append({
                    'step': step,
                    'vehicle_id': veh_id,
                    'lane_id': lane_id,
                    'speed': speed,
                    'position_x': position[0],
                    'position_y': position[1],
                    'angle': angle,
                    's_coord': s,
                    'd_coord': d,
                    'road_id': road_id,
                    'route': route
                })

            except Exception as e:
                continue

        # 记录步数据
        self.step_data.append({
            'step': step,
            'active_vehicles': len(vehicle_ids),
            'departed_this_step': departed,
            'arrived_this_step': arrived,
            'cumulative_departed': self.cumulative_departed,
            'cumulative_arrived': self.cumulative_arrived
        })

        # 每1000步输出一次
        if step % 1000 == 0 and step > 0:
            print(f"[步骤 {step}] 活跃: {len(vehicle_ids)}, "
                  f"累计出发: {self.cumulative_departed}, "
                  f"累计到达: {self.cumulative_arrived}")

    def apply_control_algorithm(self, step):
        """
        控制算法 - BASELINE版本（无控制）

        关键：不应用任何控制，使用SUMO默认的IDM模型
        """
        # Baseline：什么都不做，SUMO会自动使用IDM模型
        pass

    def run(self):
        """运行baseline仿真"""
        print("\n[运行] 开始baseline仿真...")

        self.initialize_environment()
        self.start_simulation()

        step = 0
        try:
            while step < self.max_steps and traci.simulation.getMinExpectedNumber() > 0:
                self.apply_control_algorithm(step)
                self.collect_step_data(step)
                traci.simulationStep()
                step += 1

        except KeyboardInterrupt:
            print("\n⚠️  仿真被中断")

        finally:
            traci.close()
            print(f"\n✓ SUMO已关闭 (运行{step}步)")

        return self.compute_results()

    def compute_results(self):
        """计算结果"""
        departed_count = len(self.all_departed_vehicles)
        arrived_count = len(self.all_arrived_vehicles)

        ocr = arrived_count / departed_count if departed_count > 0 else 0.0

        # 计算速度统计
        speeds = [v['speed'] for v in self.vehicle_data]
        mean_speed = np.mean(speeds) if speeds else 0.0
        std_speed = np.std(speeds) if speeds else 0.0

        results = {
            'ocr': ocr,
            'departed_count': departed_count,
            'arrived_count': arrived_count,
            'mean_speed': mean_speed,
            'std_speed': std_speed,
            'total_steps': len(self.step_data),
            'final_active_vehicles': self.step_data[-1]['active_vehicles'] if self.step_data else 0
        }

        return results

    def save_results(self, results, output_dir="competition_results"):
        """保存结果到Excel"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "baseline_results.xlsx"

        print(f"\n[保存] 正在保存结果到 {output_file}...")

        wb = Workbook()

        # 删除默认sheet
        wb.remove(wb.active)

        # Sheet 1: 数据汇总
        ws_summary = wb.create_sheet("数据汇总")
        ws_summary.append(["指标", "值"])
        ws_summary.append(["理论总需求", results['departed_count']])
        ws_summary.append(["实际累计出发", results['departed_count']])
        ws_summary.append(["实际累计到达", results['arrived_count']])
        ws_summary.append(["OD完成率(OCR)", results['ocr']])
        ws_summary.append(["唯一车辆数", results['departed_count']])
        ws_summary.append(["平均速度(m/s)", f"{results['mean_speed']:.2f}"])
        ws_summary.append(["速度标准差", f"{results['std_speed']:.2f}"])

        # Sheet 2: 仿真参数
        ws_params = wb.create_sheet("仿真参数")
        ws_params.append(["参数", "值"])
        ws_params.append(["SUMO配置", self.sumo_cfg])
        ws_params.append(["最大步数", self.max_steps])
        ws_params.append(["ICV比例", "0% (baseline)"])
        ws_params.append(["使用GUI", self.use_gui])

        # Sheet 3: 时间步数据
        ws_steps = wb.create_sheet("时间步数据")
        if self.step_data:
            headers = list(self.step_data[0].keys())
            ws_steps.append(headers)
            for row in self.step_data[::10]:  # 每10步保存一次
                ws_steps.append([row[h] for h in headers])

        # Sheet 4: 车辆数据
        ws_vehicles = wb.create_sheet("车辆数据")
        if self.vehicle_data:
            headers = list(self.vehicle_data[0].keys())
            ws_vehicles.append(headers)
            for row in self.vehicle_data[::100]:  # 每100步保存一次
                ws_vehicles.append([row[h] for h in headers])

        wb.save(output_file)

        print(f"✓ 结果已保存到: {output_file}")

        return output_file


def main():
    """主函数"""
    # 配置
    sumo_cfg = "仿真环境_初赛_1.0/仿真环境-初赛/sumo_train.sumocfg"
    max_steps = 3600
    use_gui = False
    output_dir = "competition_results"

    print("\n" + "=" * 70)
    print("Baseline评估 - 无ICV控制")
    print("=" * 70)

    # 创建评估器
    evaluator = BaselineEvaluator(
        sumo_cfg=sumo_cfg,
        max_steps=max_steps,
        use_gui=use_gui
    )

    # 运行仿真
    results = evaluator.run()

    # 打印结果
    print("\n" + "=" * 70)
    print("Baseline评估结果")
    print("=" * 70)
    print(f"\n📊 核心指标:")
    print(f"  OCR: {results['ocr']:.4f} ({results['arrived_count']}/{results['departed_count']})")
    print(f"  累计出发: {results['departed_count']}")
    print(f"  累计到达: {results['arrived_count']}")
    print(f"  运行步数: {results['total_steps']}")
    print(f"  最终活跃车辆: {results['final_active_vehicles']}")
    print(f"\n🚗 车辆性能:")
    print(f"  平均速度: {results['mean_speed']:.2f} ± {results['std_speed']:.2f} m/s")

    # 保存结果
    evaluator.save_results(results, output_dir)

    # 输出baseline OCR（方便复制到评估脚本）
    print(f"\n📌 Baseline OCR: {results['ocr']:.4f}")
    print(f"   用法: python scripts/evaluate_ocr_max.py --baseline_ocr {results['ocr']:.4f}\n")

    # 保存到JSON
    json_file = Path(output_dir) / "baseline_results.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ JSON结果已保存到: {json_file}\n")

    return results


if __name__ == '__main__':
    main()
