"""
Frenet坐标系工具集 - 针对固定SUMO路网优化

利用比赛的固定路网配置,提供准确的Frenet坐标计算和车道中心线管理。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
from pathlib import Path


class LaneCenterline:
    """车道中心线 - 管理单个车道的几何信息"""

    def __init__(self, lane_id: str, shape: List[Tuple[float, float]], length: float):
        self.lane_id = lane_id
        self.shape = np.array(shape)  # [(x, y), ...]
        self.length = length

        # 预计算累积距离(用于快速查找最近点)
        self._compute_cumulative_distance()

    def _compute_cumulative_distance(self):
        """预计算沿车道的累积距离"""
        if len(self.shape) < 2:
            self.cumulative_dist = np.array([0.0])
            return

        # 计算相邻点间的距离
        diffs = np.diff(self.shape, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)

        # 累积距离
        self.cumulative_dist = np.concatenate([[0.0], np.cumsum(segment_lengths)])

    def get_position_at_s(self, s: float) -> Tuple[float, float]:
        """
        获取纵向位置s处的笛卡尔坐标

        Args:
            s: 沿车道距离(m)

        Returns:
            (x, y): 笛卡尔坐标
        """
        if len(self.shape) == 0:
            return (0.0, 0.0)

        # 边界处理
        s = np.clip(s, 0.0, self.length)

        # 找到s对应的线段
        idx = np.searchsorted(self.cumulative_dist, s) - 1
        idx = max(0, min(idx, len(self.shape) - 2))

        # 线性插值
        s0, s1 = self.cumulative_dist[idx], self.cumulative_dist[idx + 1]
        if s1 - s0 < 1e-6:
            return tuple(self.shape[idx])

        ratio = (s - s0) / (s1 - s0)
        pos = self.shape[idx] + ratio * (self.shape[idx + 1] - self.shape[idx])

        return tuple(pos)

    def get_heading_at_s(self, s: float) -> float:
        """
        获取纵向位置s处的航向角(弧度)

        Returns:
            heading: 航向角(弧度), 范围[-pi, pi]
        """
        if len(self.shape) < 2:
            return 0.0

        s = np.clip(s, 0.0, self.length)
        idx = np.searchsorted(self.cumulative_dist, s) - 1
        idx = max(0, min(idx, len(self.shape) - 2))

        # 计算切线方向
        tangent = self.shape[idx + 1] - self.shape[idx]
        heading = np.arctan2(tangent[1], tangent[0])

        return heading

    def get_curvature_at_s(self, s: float) -> float:
        """
        获取纵向位置s处的曲率

        Returns:
            curvature: 曲率(1/m), 正值表示左转
        """
        if len(self.shape) < 3:
            return 0.0

        s = np.clip(s, 0.0, self.length)
        idx = np.searchsorted(self.cumulative_dist, s) - 1
        idx = max(1, min(idx, len(self.shape) - 2))

        # 使用三点法估算曲率
        p0, p1, p2 = self.shape[idx - 1], self.shape[idx], self.shape[idx + 1]

        # 计算向量
        v1 = p1 - p0
        v2 = p2 - p1

        # 叉积和点积
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)

        # 曲率
        curvature = cross / (np.linalg.norm(v1) * np.linalg.norm(v2) * (np.linalg.norm(v1) + np.linalg.norm(v2)) + 1e-6)

        return curvature


class FrenetCoordinateSystem:
    """
    Frenet坐标系管理器 - 针对比赛路网优化

    功能:
    1. 解析net.xml提取车道几何
    2. 提供准确的Frenet坐标转换
    3. 识别瓶颈区域
    4. 预计算关键位置信息
    """

    def __init__(self, net_xml_path: str):
        self.net_xml_path = Path(net_xml_path)
        self.lanes: Dict[str, LaneCenterline] = {}
        self.edge_to_lanes: Dict[str, List[str]] = {}
        self.bottleneck_areas: List[Dict] = []

        # 解析路网
        self._parse_network()

        # 识别瓶颈
        self._identify_bottlenecks()

    def _parse_network(self):
        """解析SUMO net.xml文件"""
        tree = ET.parse(self.net_xml_path)
        root = tree.getroot()

        # 解析每条车道
        for edge in root.findall('edge'):
            edge_id = edge.get('id')

            # 跳过内部边(junction内部连接)
            if edge.get('function') == 'internal':
                continue

            if edge_id not in self.edge_to_lanes:
                self.edge_to_lanes[edge_id] = []

            for lane in edge.findall('lane'):
                lane_id = lane.get('id')
                length = float(lane.get('length'))
                shape_str = lane.get('shape', '')

                # 解析shape坐标
                if shape_str:
                    shape = [
                        tuple(map(float, point.split(',')))
                        for point in shape_str.split()
                    ]
                else:
                    # 如果没有shape,使用from/to junction坐标推断
                    shape = self._infer_lane_shape(edge, lane)

                # 创建车道中心线
                centerline = LaneCenterline(lane_id, shape, length)
                self.lanes[lane_id] = centerline
                self.edge_to_lanes[edge_id].append(lane_id)

    def _infer_lane_shape(self, edge, lane) -> List[Tuple[float, float]]:
        """推断车道shape(当net.xml中没有明确shape时)"""
        # 简化处理: 使用from和to junction的坐标
        from_junction = edge.get('from')
        to_junction = edge.get('to')

        # 这里需要解析junction坐标,暂时返回简化直线
        # 实际实现中需要遍历junction元素获取坐标
        return [(0.0, 0.0), (100.0, 0.0)]  # 占位符

    def _identify_bottlenecks(self):
        """识别瓶颈区域(基于路网拓扑)"""
        # 根据net.xml分析,关键瓶颈点:
        # 1. J14: E15汇入
        # 2. J15: E17汇入(最严重)
        # 3. J17: E19汇入

        self.bottleneck_areas = [
            {
                'junction_id': 'J14',
                'edges': ['E9', 'E15', 'E10'],
                'description': 'E15匝道汇入瓶颈',
                'critical_range': (1090, 1200)  # s坐标范围
            },
            {
                'junction_id': 'J15',
                'edges': ['E10', 'E17', 'E11'],
                'description': 'E17匝道汇入瓶颈(最严重)',
                'critical_range': (1280, 1320)
            },
            {
                'junction_id': 'J17',
                'edges': ['E12', 'E19', 'E13'],
                'description': 'E19匝道汇入瓶颈',
                'critical_range': (1720, 1760)
            }
        ]

    def cartesian_to_frenet(
        self,
        x: float, y: float,
        edge_id: str, lane_id: str
    ) -> Tuple[float, float]:
        """
        笛卡尔坐标转Frenet坐标

        Args:
            x, y: 笛卡尔坐标
            edge_id: 边ID
            lane_id: 车道ID

        Returns:
            (s, d): 纵向位置(m), 横向偏移(m)
        """
        if lane_id not in self.lanes:
            # 车道不存在,返回简化坐标
            return (0.0, 0.0)

        centerline = self.lanes[lane_id]

        # 找到最近点
        distances = np.linalg.norm(centerline.shape - np.array([x, y]), axis=1)
        nearest_idx = np.argmin(distances)

        # s坐标: 沿车道距离
        s = centerline.cumulative_dist[nearest_idx]

        # d坐标: 横向偏移(右为正,左为负)
        nearest_point = centerline.shape[nearest_idx]
        heading = centerline.get_heading_at_s(s)

        # 计算横向偏移
        dx = x - nearest_point[0]
        dy = y - nearest_point[1]

        # 旋转到Frenet坐标系
        d = dx * np.sin(heading) - dy * np.cos(heading)

        return (s, d)

    def frenet_to_cartesian(
        self,
        s: float, d: float,
        lane_id: str
    ) -> Tuple[float, float]:
        """
        Frenet坐标转笛卡尔坐标

        Args:
            s: 纵向位置(m)
            d: 横向偏移(m)
            lane_id: 车道ID

        Returns:
            (x, y): 笛卡尔坐标
        """
        if lane_id not in self.lanes:
            return (0.0, 0.0)

        centerline = self.lanes[lane_id]

        # 获取s位置的中心线点
        x0, y0 = centerline.get_position_at_s(s)

        # 获取航向角
        heading = centerline.get_heading_at_s(s)

        # 计算横向偏移对应的笛卡尔坐标
        # d为正表示右侧,需要沿着法线方向移动
        x = x0 - d * np.sin(heading)
        y = y0 + d * np.cos(heading)

        return (x, y)

    def is_in_bottleneck(self, s: float, edge_id: str) -> bool:
        """
        判断位置是否在瓶颈区域

        Args:
            s: 纵向位置
            edge_id: 边ID

        Returns:
            True if in bottleneck area
        """
        for area in self.bottleneck_areas:
            if edge_id in area['edges']:
                s_min, s_max = area['critical_range']
                if s_min <= s <= s_max:
                    return True
        return False

    def get_bottleneck_info(self, edge_id: str) -> Optional[Dict]:
        """获取指定边的瓶颈信息"""
        for area in self.bottleneck_areas:
            if edge_id in area['edges']:
                return area
        return None

    def get_lane_length(self, lane_id: str) -> float:
        """获取车道长度"""
        if lane_id in self.lanes:
            return self.lanes[lane_id].length
        return 0.0

    def get_lane_curvature(self, lane_id: str, s: float) -> float:
        """获取车道在位置s处的曲率"""
        if lane_id in self.lanes:
            return self.lanes[lane_id].get_curvature_at_s(s)
        return 0.0


# 全局单例
_frenet_system: Optional[FrenetCoordinateSystem] = None


def get_frenet_system(net_xml_path: Optional[str] = None) -> FrenetCoordinateSystem:
    """
    获取Frenet坐标系单例

    Args:
        net_xml_path: net.xml文件路径,首次调用时必须提供

    Returns:
        FrenetCoordinateSystem实例
    """
    global _frenet_system

    if _frenet_system is None:
        if net_xml_path is None:
            raise ValueError("首次调用必须提供net_xml_path")
        _frenet_system = FrenetCoordinateSystem(net_xml_path)

    return _frenet_system


def normalize_frenet_features(
    s: float, d: float,
    vs: float, vd: float,
    speed: float, acceleration: float,
    lane_index: float, angle: float,
    is_icv: float,
    edge_id: Optional[str] = None
) -> np.ndarray:
    """
    归一化Frenet特征(用于神经网络输入)

    Args:
        s: 纵向位置(m)
        d: 横向偏移(m)
        vs: 纵向速度(m/s)
        vd: 横向速度(m/s)
        speed: 总速度(m/s)
        acceleration: 加速度(m/s²)
        lane_index: 车道索引
        angle: 航向角(度)
        is_icv: 是否为ICV
        edge_id: 边ID(可选,用于瓶颈检测)

    Returns:
        9维归一化特征向量
    """
    features = np.array([
        s / 2000.0,              # 纵向位置(最大约2000m)
        d / 10.0,                # 横向偏移(约±5m)
        vs / 30.0,               # 纵向速度(最大30m/s)
        vd / 10.0,               # 横向速度(约±5m/s)
        speed / 30.0,            # 总速度
        acceleration / 3.0,      # 加速度
        lane_index / 10.0,       # 车道索引
        angle / 360.0,           # 航向角
        is_icv                   # 二值特征
    ], dtype=np.float32)

    return features


if __name__ == '__main__':
    # 测试代码
    print("=" * 70)
    print("测试Frenet坐标系工具")
    print("=" * 70)

    # 创建Frenet系统
    net_path = "仿真环境_初赛_1.0/仿真环境-初赛/net.xml"
    frenet = FrenetCoordinateSystem(net_path)

    print(f"\n[OK] 解析到 {len(frenet.lanes)} 条车道")
    print(f"[OK] 识别到 {len(frenet.bottleneck_areas)} 个瓶颈区域")

    # 显示瓶颈信息
    print("\n瓶颈区域:")
    for area in frenet.bottleneck_areas:
        print(f"  - {area['description']}: {area['critical_range']}m")

    # 测试坐标转换
    if 'E1_0' in frenet.lanes:
        lane = frenet.lanes['E1_0']
        print(f"\n车道 E1_0:")
        print(f"  长度: {lane.length:.2f}m")

        # 测试几个位置
        for s in [0, 100, 200, 300]:
            x, y = frenet.frenet_to_cartesian(s, 0.0, 'E1_0')
            s_back, d_back = frenet.cartesian_to_frenet(x, y, 'E1', 'E1_0')
            print(f"  s={s}m -> ({x:.2f}, {y:.2f}) -> s={s_back:.2f}, d={d_back:.2f}")

    print("\n" + "=" * 70)
    print("[OK] 测试完成")
    print("=" * 70)
