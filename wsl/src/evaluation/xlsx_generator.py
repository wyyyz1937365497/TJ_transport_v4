"""
XLSX结果生成器 - 完整实现
生成竞赛提交格式的XLSX文件
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import (
    Font,
    Alignment,
    PatternFill,
    Border,
    Side,
)
from openpyxl.utils import get_column_letter

from ..models import TrafficController
from ..env import SumoEnvironment
from ..utils import Config, get_logger
from .evaluator import Evaluator

logger = get_logger()


class XLSXResultGenerator:
    """
    XLSX结果生成器 - 完整实现

    生成符合竞赛要求的XLSX结果文件

    格式要求：
    - Sheet1: 每个车辆每一步的控制指令
    - Sheet2: 统计摘要
    """

    def __init__(
        self,
        model: TrafficController,
        config: Config,
    ):
        self.model = model
        self.config = config

        # 设备
        self.device = config.device

        # 样式定义
        self._init_styles()

    def _init_styles(self):
        """初始化样式"""
        # 标题样式
        self.header_style = {
            "font": Font(bold=True, size=12, color="FFFFFF"),
            "fill": PatternFill(start_color="366092", end_color="366092", fill_type="solid"),
            "alignment": Alignment(horizontal="center", vertical="center"),
            "border": Border(
                left=Side(style="thin"),
                right=Side(style="thin"),
                top=Side(style="thin"),
                bottom=Side(style="thin"),
            ),
        }

        # 数据样式
        self.data_style = {
            "font": Font(size=10),
            "alignment": Alignment(horizontal="center", vertical="center"),
            "border": Border(
                left=Side(style="thin"),
                right=Side(style="thin"),
                top=Side(style="thin"),
                bottom=Side(style="thin"),
            ),
        }

        # 摘要标题样式
        self.summary_header_style = {
            "font": Font(bold=True, size=11, color="FFFFFF"),
            "fill": PatternFill(start_color="548235", end_color="548235", fill_type="solid"),
            "alignment": Alignment(horizontal="left", vertical="center"),
        }

    def generate(
        self,
        output_path: str = None,
        num_episodes: int = 5,
        verbose: bool = True,
    ) -> str:
        """
        生成XLSX结果文件

        Args:
            output_path: 输出文件路径
            num_episodes: 评估episodes数量
            verbose: 是否显示进度

        Returns:
            生成的文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.config.result_dir / f"results_{timestamp}.xlsx"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("生成XLSX结果文件")
        logger.info(f"输出路径: {output_path}")
        logger.info(f"Episodes: {num_episodes}")
        logger.info("=" * 70)

        # 评估模型并收集数据
        if verbose:
            logger.info("开始评估...")

        evaluator = Evaluator(self.model, self.config)
        summary_metrics = evaluator.evaluate(verbose=verbose)

        # 获取所有轨迹数据
        all_trajectories = evaluator.all_trajectories

        # 创建工作簿
        wb = Workbook()

        # 删除默认sheet
        wb.remove(wb.active)

        # Sheet1: 车辆控制指令
        self._create_control_sheet(wb, all_trajectories)

        # Sheet2: 统计摘要
        self._create_summary_sheet(wb, summary_metrics, all_trajectories)

        # Sheet3: 车辆轨迹
        self._create_trajectory_sheet(wb, all_trajectories)

        # 保存文件
        wb.save(output_path)

        logger.info("=" * 70)
        logger.info(f"XLSX文件已生成: {output_path}")
        logger.info("=" * 70)

        return str(output_path)

    def _create_control_sheet(self, wb: Workbook, all_trajectories: List[Dict]):
        """
        创建控制指令sheet

        格式：
        | 车辆ID | 时间步 | 加速度 | 换道 |
        """
        ws = wb.create_sheet("控制指令")

        # 写入标题
        headers = ["车辆ID", "时间步", "加速度 (m/s²)", "换道指令", "车道索引"]
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            self._apply_style(cell, self.header_style)

        # 写入数据
        row = 2
        total_rows = 0

        for episode_data in all_trajectories:
            trajectories = episode_data["trajectories"]

            for veh_id, traj in trajectories.items():
                timestamps = traj["timestamps"]
                actions = traj.get("actions", [])
                lane_indices = traj["lane_indices"]

                for i, (timestamp, action, lane_idx) in enumerate(
                    zip(timestamps, actions, lane_indices)
                ):
                    ws.cell(row=row, column=1, value=veh_id)
                    ws.cell(row=row, column=2, value=int(timestamp / 0.1))  # 转换为步数
                    ws.cell(row=row, column=3, value=float(action[0]))
                    ws.cell(row=row, column=4, value=float(action[1]))
                    ws.cell(row=row, column=5, value=int(lane_idx))

                    # 应用样式
                    for col in range(1, 6):
                        cell = ws.cell(row=row, column=col)
                        self._apply_style(cell, self.data_style)

                    row += 1
                    total_rows += 1

        # 自动调整列宽
        for col in range(1, 6):
            ws.column_dimensions[get_column_letter(col)].width = 18

        logger.info(f"  控制指令: {total_rows} 行")

    def _create_summary_sheet(
        self,
        wb: Workbook,
        metrics: Dict[str, Any],
        all_trajectories: List[Dict],
    ):
        """
        创建统计摘要sheet

        包含：
        1. 基本统计
        2. 性能指标
        3. 成本指标
        4. 车辆统计
        """
        ws = wb.create_sheet("统计摘要")

        row = 1

        # ========== 1. 基本信息 ==========
        row = self._add_section_header(ws, row, "基本信息")

        data = [
            ("生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("评估Episodes", len(all_trajectories)),
        ]

        for key, value in data:
            ws.cell(row=row, column=1, value=key)
            ws.cell(row=row, column=2, value=str(value))
            self._apply_style(ws.cell(row, 1), self.summary_header_style)
            row += 1

        row += 1

        # ========== 2. 性能指标 ==========
        row = self._add_section_header(ws, row, "性能指标")

        data = [
            ("OD完成率 (OCR)", f"{metrics.get('ocr_mean', 0):.2%}"),
            ("平均速度", f"{metrics.get('avg_speed_mean', 0):.2f} m/s"),
            ("速度标准差", f"{metrics.get('speed_std_mean', 0):.2f} m/s"),
            ("平均绝对加速度", f"{metrics.get('avg_abs_accel_mean', 0):.2f} m/s²"),
            ("急刹车比例", f"{metrics.get('emergency_braking_ratio_mean', 0):.2%}"),
        ]

        for key, value in data:
            ws.cell(row=row, column=1, value=key)
            ws.cell(row=row, column=2, value=value)
            self._apply_style(ws.cell(row, 1), self.summary_header_style)
            row += 1

        row += 1

        # ========== 3. 成本指标 ==========
        row = self._add_section_header(ws, row, "成本指标")

        data = [
            ("平均成本", f"{metrics.get('avg_cost_mean', 0):.4f}"),
            ("总成本", f"{metrics.get('total_cost_mean', 0):.2f}"),
            ("总动作数", f"{int(metrics.get('total_actions_mean', 0))}"),
            ("换道次数", f"{int(metrics.get('lane_changes_mean', 0))}"),
            ("加速度干预", f"{int(metrics.get('accel_interventions_mean', 0))}"),
        ]

        for key, value in data:
            ws.cell(row=row, column=1, value=key)
            ws.cell(row=row, column=2, value=value)
            self._apply_style(ws.cell(row, 1), self.summary_header_style)
            row += 1

        row += 1

        # ========== 4. 奖励指标 ==========
        row = self._add_section_header(ws, row, "奖励指标")

        data = [
            ("总奖励", f"{metrics.get('total_reward_mean', 0):.2f}"),
            ("平均奖励", f"{metrics.get('avg_reward_mean', 0):.2f}"),
        ]

        for key, value in data:
            ws.cell(row=row, column=1, value=key)
            ws.cell(row=row, column=2, value=value)
            self._apply_style(ws.cell(row, 1), self.summary_header_style)
            row += 1

        row += 1

        # ========== 5. 行程时间 ==========
        row = self._add_section_header(ws, row, "行程时间")

        data = [
            ("平均行程时间", f"{metrics.get('avg_travel_time_mean', 0):.1f} s"),
            ("最长行程时间", f"{metrics.get('max_travel_time_mean', 0):.1f} s"),
            ("最短行程时间", f"{metrics.get('min_travel_time_mean', 0):.1f} s"),
        ]

        for key, value in data:
            ws.cell(row=row, column=1, value=key)
            ws.cell(row=row, column=2, value=value)
            self._apply_style(ws.cell(row, 1), self.summary_header_style)
            row += 1

        row += 1

        # ========== 6. 安全性 ==========
        row = self._add_section_header(ws, row, "安全性")

        data = [
            ("碰撞次数", f"{metrics.get('collisions_mean', 0):.0f}"),
            ("到达车辆数", f"{int(metrics.get('arrived_count_mean', 0))}"),
            ("出发车辆数", f"{int(metrics.get('departed_count_mean', 0))}"),
        ]

        for key, value in data:
            ws.cell(row=row, column=1, value=key)
            ws.cell(row=row, column=2, value=value)
            self._apply_style(ws.cell(row, 1), self.summary_header_style)
            row += 1

        row += 1

        # ========== 7. 综合评分 ==========
        row = self._add_section_header(ws, row, "综合评分")

        data = [
            ("效率评分", f"{metrics.get('efficiency_score_mean', 0):.1f} / 100"),
            ("吞吐量", f"{metrics.get('throughput_mean', 0):.2f} veh/s"),
        ]

        for key, value in data:
            ws.cell(row=row, column=1, value=key)
            ws.cell(row=row, column=2, value=value)
            self._apply_style(ws.cell(row, 1), self.summary_header_style)
            row += 1

        # 调整列宽
        ws.column_dimensions["A"].width = 25
        ws.column_dimensions["B"].width = 20

        logger.info("  统计摘要: 已创建")

    def _create_trajectory_sheet(self, wb: Workbook, all_trajectories: List[Dict]):
        """
        创建车辆轨迹sheet

        格式：
        | Episode | 车辆ID | 时间戳 | 位置 | 速度 | 加速度 | 车道 |
        """
        ws = wb.create_sheet("车辆轨迹")

        # 写入标题
        headers = [
            "Episode",
            "车辆ID",
            "时间戳 (s)",
            "位置 (m)",
            "速度 (m/s)",
            "加速度 (m/s²)",
            "车道索引",
        ]
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            self._apply_style(cell, self.header_style)

        # 写入数据
        row = 2
        total_rows = 0

        for episode_idx, episode_data in enumerate(all_trajectories):
            trajectories = episode_data["trajectories"]

            for veh_id, traj in trajectories.items():
                timestamps = traj["timestamps"]
                positions = traj["positions"]
                speeds = traj["speeds"]
                accels = traj["accelerations"]
                lane_indices = traj["lane_indices"]

                for i in range(len(timestamps)):
                    ws.cell(row=row, column=1, value=episode_idx)
                    ws.cell(row=row, column=2, value=veh_id)
                    ws.cell(row=row, column=3, value=float(timestamps[i]))
                    ws.cell(row=row, column=4, value=float(positions[i]))
                    ws.cell(row=row, column=5, value=float(speeds[i]))
                    ws.cell(row=row, column=6, value=float(accels[i]))
                    ws.cell(row=row, column=7, value=int(lane_indices[i]))

                    # 应用样式
                    for col in range(1, 8):
                        cell = ws.cell(row=row, column=col)
                        self._apply_style(cell, self.data_style)

                    row += 1
                    total_rows += 1

        # 自动调整列宽
        for col in range(1, 8):
            ws.column_dimensions[get_column_letter(col)].width = 15

        logger.info(f"  车辆轨迹: {total_rows} 行")

    def _add_section_header(self, ws, row: int, title: str) -> int:
        """添加节标题"""
        cell = ws.cell(row=row, column=1, value=title)
        cell.font = Font(bold=True, size=12, color="FFFFFF")
        cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
        return row + 1

    def _apply_style(self, cell, style: Dict):
        """应用样式到单元格"""
        if "font" in style:
            cell.font = style["font"]
        if "fill" in style:
            cell.fill = style["fill"]
        if "alignment" in style:
            cell.alignment = style["alignment"]
        if "border" in style:
            cell.border = style["border"]


def generate_xlsx_report(
    model: TrafficController,
    config: Config,
    output_path: str = None,
    num_episodes: int = 5,
) -> str:
    """
    便捷函数：生成XLSX报告

    Args:
        model: 训练好的模型
        config: 配置
        output_path: 输出路径
        num_episodes: 评估episodes数量

    Returns:
        生成的文件路径
    """
    generator = XLSXResultGenerator(model, config)
    return generator.generate(output_path, num_episodes)
