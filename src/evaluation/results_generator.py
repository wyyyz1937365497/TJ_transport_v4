"""
XLSX结果生成器
功能：生成符合竞赛要求的结果文件
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    print("⚠️  警告: openpyxl未安装，XLSX功能将不可用")


class XLSXResultGenerator:
    """
    XLSX结果文件生成器

    按照竞赛要求生成结果文件：
    - 车辆ID
    - 时间步
    - 控制指令（加速度、换道）
    """

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if not OPENPYXL_AVAILABLE:
            raise RuntimeError("openpyxl未安装，无法生成XLSX文件")

    def generate_results(
        self,
        model,
        config: Dict[str, Any],
        num_episodes: int = 5,
        filename: Optional[str] = None
    ) -> str:
        """
        生成XLSX结果文件

        Args:
            model: 训练好的模型
            config: 配置
            num_episodes: 仿真episodes数量
            filename: 输出文件名（可选）

        Returns:
            filepath: 生成的文件路径
        """
        print("\n" + "="*70)
        print("📊 生成XLSX结果文件")
        print("="*70)

        from ..env import SumoEnvironment

        # 创建环境
        env_config = config.get('environment', {})
        env = SumoEnvironment(env_config, use_gui=False)

        # 收集所有控制指令
        all_control_actions = []

        for episode in range(num_episodes):
            print(f"\n🔄 运行 Episode {episode + 1}/{num_episodes}")

            # 重置环境
            observation = env.reset()
            done = False
            step = 0

            while not done:
                # 准备batch
                batch = self._prepare_batch(observation, env, config)

                # 模型推理
                try:
                    output = model(batch)

                    # 提取控制指令
                    if output['selected_vehicle_ids']:
                        safe_actions = output['safe_actions'].cpu().numpy()

                        for i, veh_id in enumerate(output['selected_vehicle_ids']):
                            # 映射动作到实际控制
                            acceleration = self._map_acceleration(safe_actions[i, 0])
                            lane_change = int(safe_actions[i, 1] > 0.5)

                            # 记录
                            all_control_actions.append({
                                'vehicle_id': veh_id,
                                'time': step * config.get('step_length', 0.1),
                                'acceleration': acceleration,
                                'lane_change': lane_change
                            })
                except Exception as e:
                    print(f"   ⚠️  警告: 推理失败 - {e}")

                # 推进仿真
                observation, reward, done, info = env.step(actions=None)
                step += 1

                # 进度报告
                if step % 100 == 0:
                    print(f"   Step {step} | 车辆数: {len(observation['vehicle_states'])}")

            print(f"✅ Episode {episode + 1} 完成 | 步数: {step}")

        env.close()

        # 生成XLSX
        if all_control_actions:
            filepath = self._create_xlsx(all_control_actions, filename)
            print(f"\n✅ XLSX文件已生成: {filepath}")
            print(f"   - 总控制指令数: {len(all_control_actions)}")
            return filepath
        else:
            print("\n⚠️  警告: 没有生成任何控制指令")
            return ""

    def _prepare_batch(
        self,
        observation: Dict,
        env,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """准备推理batch"""
        vehicle_states = observation['vehicle_states']
        vehicle_ids = observation['vehicle_ids']
        icv_ids = observation['icv_ids']
        global_stats = observation['global_stats']

        device = torch.device(config.get('device', 'cpu'))

        batch = {
            'vehicle_states': vehicle_states,
            'vehicle_ids': vehicle_ids,
            'icv_ids': icv_ids,
            'global_metrics': torch.tensor(global_stats, dtype=torch.float32).unsqueeze(0).to(device),
            'is_icv': torch.tensor(
                [1 if vid in icv_ids else 0 for vid in vehicle_ids],
                dtype=torch.float32
            ).to(device) if vehicle_ids else torch.tensor([], dtype=torch.float32).to(device)
        }

        return batch

    def _map_acceleration(self, normalized_accel: float) -> float:
        """
        将归一化加速度映射到实际范围

        Args:
            normalized_accel: [-1, 1]

        Returns:
            acceleration: [-3, 2] m/s²
        """
        # Tanh输出[-1,1]映射到物理加速度[-3,2]
        min_accel = -3.0
        max_accel = 2.0
        accel_range = max_accel - min_accel

        physical_accel = min_accel + (normalized_accel + 1) / 2 * accel_range

        return float(physical_accel)

    def _create_xlsx(
        self,
        control_actions: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """
        创建XLSX文件

        Args:
            control_actions: 控制指令列表
            filename: 文件名（可选）

        Returns:
            filepath: 生成的文件路径
        """
        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"traffic_control_results_{timestamp}.xlsx"

        filepath = os.path.join(self.output_dir, filename)

        # 创建DataFrame
        df = pd.DataFrame(control_actions)

        # 按时间排序
        df = df.sort_values(['time', 'vehicle_id'])

        # 创建Excel工作簿
        wb = Workbook()
        ws = wb.active
        ws.title = "控制指令"

        # 写入标题
        headers = ['车辆ID', '时间(s)', '加速度(m/s²)', '换道']
        ws.append(headers)

        # 标题样式
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_alignment = Alignment(horizontal="center", vertical="center")

        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment

        # 写入数据
        for _, row in df.iterrows():
            ws.append([
                row['vehicle_id'],
                row['time'],
                round(row['acceleration'], 3),
                row['lane_change']
            ])

        # 调整列宽
        ws.column_dimensions['A'].width = 15
        ws.column_dimensions['B'].width = 12
        ws.column_dimensions['C'].width = 15
        ws.column_dimensions['D'].width = 10

        # 添加统计工作表
        self._add_summary_sheet(wb, df)

        # 保存文件
        wb.save(filepath)

        return filepath

    def _add_summary_sheet(self, wb: Workbook, df: pd.DataFrame):
        """添加统计摘要工作表"""
        ws = wb.create_sheet("统计摘要")

        # 计算统计信息
        total_vehicles = df['vehicle_id'].nunique()
        total_actions = len(df)
        total_time = df['time'].max() if len(df) > 0 else 0

        accel_stats = df['acceleration'].describe()
        lane_changes = (df['lane_change'] == 1).sum()

        # 写入统计
        summary_data = [
            ["统计项目", "数值"],
            ["控制车辆总数", total_vehicles],
            ["控制指令总数", total_actions],
            ["仿真时长", f"{total_time:.1f}"],
            ["", ""],
            ["加速度统计", ""],
            ["平均值", f"{accel_stats['mean']:.3f}"],
            ["标准差", f"{accel_stats['std']:.3f}"],
            ["最小值", f"{accel_stats['min']:.3f}"],
            ["最大值", f"{accel_stats['max']:.3f}"],
            ["", ""],
            ["换道次数", lane_changes],
            ["换道比例", f"{lane_changes/total_actions*100:.2f}%" if total_actions > 0 else "0%"]
        ]

        for row in summary_data:
            ws.append(row)

        # 样式
        ws['A1'].font = Font(bold=True)
        ws['B1'].font = Font(bold=True)

        # 调整列宽
        ws.column_dimensions['A'].width = 20
        ws.column_dimensions['B'].width = 15

    def generate_simple_xlsx(
        self,
        trajectories: Dict[str, Dict],
        filename: Optional[str] = None
    ) -> str:
        """
        生成简化版XLSX（用于测试）

        Args:
            trajectories: 轨迹数据
            filename: 文件名（可选）

        Returns:
            filepath: 生成的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simple_results_{timestamp}.xlsx"

        filepath = os.path.join(self.output_dir, filename)

        # 创建Excel工作簿
        wb = Workbook()
        ws = wb.active
        ws.title = "轨迹数据"

        # 写入标题
        headers = ['车辆ID', '时间(s)', '位置', '速度(m/s)', '加速度(m/s²)', '车道ID']
        ws.append(headers)

        # 标题样式
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")

        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # 写入数据（采样）
        sample_count = 0
        max_samples = 10000  # 限制样本数避免文件过大

        for veh_id, traj in trajectories.items():
            for i in range(len(traj['timestamps'])):
                if sample_count >= max_samples:
                    break

                ws.append([
                    veh_id,
                    round(traj['timestamps'][i], 2),
                    round(traj['positions'][i], 2),
                    round(traj['speeds'][i], 2),
                    round(traj['accelerations'][i], 2),
                    traj['lane_ids'][i]
                ])

                sample_count += 1

        # 调整列宽
        for col in ['A', 'B', 'C', 'D', 'E', 'F']:
            ws.column_dimensions[col].width = 15

        # 保存
        wb.save(filepath)

        return filepath


def generate_evaluation_report(
    model,
    config: Dict[str, Any],
    output_dir: str = "results"
) -> Dict[str, Any]:
    """
    生成评估报告

    Args:
        model: 训练好的模型
        config: 配置
        output_dir: 输出目录

    Returns:
        评估指标
    """
    print("\n" + "="*70)
    print("📊 生成评估报告")
    print("="*70)

    from ..env import SumoEnvironment

    # 创建环境
    env_config = config.get('environment', {})
    env = SumoEnvironment(env_config, use_gui=False)

    # 运行评估
    all_rewards = []
    all_costs = []
    all_speeds = []
    all_collisions = []

    num_episodes = 5

    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        episode_reward = 0
        episode_cost = 0
        episode_speeds = []
        step = 0

        while not done:
            # 准备batch
            batch = {
                'vehicle_states': observation['vehicle_states'],
                'vehicle_ids': observation['vehicle_ids'],
                'icv_ids': observation['icv_ids'],
                'global_metrics': torch.tensor(
                    observation['global_stats'],
                    dtype=torch.float32
                ).unsqueeze(0).to(config.get('device', 'cpu')),
                'is_icv': torch.tensor(
                    [1 if vid in observation['icv_ids'] else 0 for vid in observation['vehicle_ids']],
                    dtype=torch.float32
                ).to(config.get('device', 'cpu')) if observation['vehicle_ids'] else torch.tensor([], dtype=torch.float32).to(config.get('device', 'cpu'))
            }

            # 模型推理
            try:
                output = model(batch)

                # 计算成本
                if output['selected_vehicle_ids']:
                    safe_actions = output['safe_actions']
                    for action in safe_actions:
                        accel_cost = abs(action[0].item())
                        lane_cost = 5.0 if action[1].item() > 0.5 else 0.0
                        episode_cost += (accel_cost + lane_cost)

            except:
                pass

            # 推进仿真
            observation, reward, done, info = env.step(actions=None)

            episode_reward += reward
            episode_speeds.extend([v['speed'] for v in observation['vehicle_states'].values()])

            step += 1

        all_rewards.append(episode_reward)
        all_costs.append(episode_cost)
        all_speeds.extend(episode_speeds)
        all_collisions.append(info.get('collisions', 0))

        print(f"   Episode {episode + 1} | Reward: {episode_reward:.3f} | Cost: {episode_cost:.3f}")

    env.close()

    # 计算指标
    metrics = {
        'avg_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'avg_cost': np.mean(all_costs),
        'std_cost': np.std(all_costs),
        'avg_speed': np.mean(all_speeds) if all_speeds else 0,
        'std_speed': np.std(all_speeds) if len(all_speeds) > 1 else 0,
        'total_collisions': sum(all_collisions),
        'num_episodes': num_episodes
    }

    print(f"\n✅ 评估完成!")
    print(f"   - 平均奖励: {metrics['avg_reward']:.3f} ± {metrics['std_reward']:.3f}")
    print(f"   - 平均成本: {metrics['avg_cost']:.3f} ± {metrics['std_cost']:.3f}")
    print(f"   - 平均速度: {metrics['avg_speed']:.2f} ± {metrics['std_speed']:.2f} m/s")
    print(f"   - 总碰撞: {metrics['total_collisions']}")

    return metrics
