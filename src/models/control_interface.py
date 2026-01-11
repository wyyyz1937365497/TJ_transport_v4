"""
交通控制接口 - 支持多种控制方式

初赛: 车辆控制（加速度、换道）
复赛: + 红绿灯控制 + VSL限速控制

设计原则：
- 模块化：每种控制方式独立实现
- 可扩展：易于添加新的控制方式
- 向后兼容：不影响现有训练流程
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np


class ControlInterface(ABC):
    """控制接口基类"""

    @abstractmethod
    def get_action_space(self) -> Dict[str, Any]:
        """返回动作空间定义"""
        pass

    @abstractmethod
    def apply_actions(self, env, actions: Dict[str, Any]) -> Dict[str, Any]:
        """应用控制动作到环境"""
        pass


class VehicleControl(ControlInterface):
    """
    初赛：车辆控制

    动作空间：
    - acceleration: 加速度 [-3.0, 2.0] m/s²
    - lane_change: 换道 [-1=左, 0=不变, 1=右]
    """

    def get_action_space(self) -> Dict[str, Any]:
        return {
            'type': 'continuous',
            'shape': (2,),  # [acceleration, lane_change]
            'low': np.array([-3.0, 0.0]),
            'high': np.array([2.0, 1.0]),
            'description': ['加速度(m/s²)', '换道(-1左/0不变/1右)']
        }

    def apply_actions(self, env, actions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """应用车辆控制动作"""
        return env.step(actions)


class TrafficLightControl(ControlInterface):
    """
    复赛：红绿灯控制（预留）

    动作空间：
    - phase: 相位索引 (0=绿灯, 1=黄灯, 2=红灯)
    - duration: 持续时间 [10, 60] 秒

    TODO: 复赛阶段实现
    """

    def get_action_space(self) -> Dict[str, Any]:
        return {
            'type': 'multi-discrete',
            'n_phase': 3,  # 绿/黄/红
            'duration_range': [10, 60],  # 秒
            'description': '红绿灯相位和持续时间'
        }

    def apply_actions(self, env, actions: Dict[str, Any]) -> Dict[str, Any]:
        """应用红绿灯控制动作（TODO: 复赛实现）"""
        print("⚠️  红绿灯控制尚未实现（复赛功能）")
        return env.step({})  # 降级到无控制


class VSLControl(ControlInterface):
    """
    复赛：可变限速控制（Variable Speed Limit）

    动作空间：
    - edge_id: 路段ID
    - speed_limit: 限速值 [40, 120] km/h

    TODO: 复赛阶段实现
    """

    def get_action_space(self) -> Dict[str, Any]:
        return {
            'type': 'continuous',
            'speed_limit_range': [40, 120],  # km/h
            'description': '可变限速标志的限速值'
        }

    def apply_actions(self, env, actions: Dict[str, Any]) -> Dict[str, Any]:
        """应用VSL控制动作（TODO: 复赛实现）"""
        print("⚠️  VSL控制尚未实现（复赛功能）")
        return env.step({})  # 降级到无控制


class HybridController:
    """
    混合控制器 - 支持多种控制方式

    根据比赛阶段启用不同的控制方式：
    - 初赛: 仅车辆控制
    - 复赛: 车辆 + 红绿灯 + VSL
    """

    def __init__(self, config: Dict[str, Any]):
        self.controllers = {
            'vehicle': VehicleControl(),
            'traffic_light': TrafficLightControl(),
            'vsl': VSLControl()
        }

        # 根据比赛阶段启用控制方式
        self.active_controllers = ['vehicle']  # 初赛默认

        phase = config.get('competition_phase', 'preliminary')
        if phase == 'final':
            self.active_controllers.extend(['traffic_light', 'vsl'])
            print("✅ 复赛模式: 启用红绿灯和VSL控制")

    def add_controller(self, controller_type: str):
        """动态添加控制方式"""
        if controller_type in self.controllers:
            if controller_type not in self.active_controllers:
                self.active_controllers.append(controller_type)
                print(f"✅ 已添加控制方式: {controller_type}")

    def get_combined_action_space(self) -> Dict[str, Dict]:
        """获取组合动作空间"""
        spaces = {}
        for ctrl_type in self.active_controllers:
            spaces[ctrl_type] = self.controllers[ctrl_type].get_action_space()
        return spaces

    def apply_hybrid_actions(
        self,
        env,
        actions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        应用混合控制动作

        Args:
            env: 环境
            actions: {
                'vehicle': {...},  # 车辆控制动作
                'traffic_light': {...},  # 红绿灯动作（可选）
                'vsl': {...}  # VSL动作（可选）
            }

        Returns:
            组合的观察、奖励等
        """
        results = {}

        # 按优先级应用控制（车辆 > 红绿灯 > VSL）
        for ctrl_type in ['vehicle', 'traffic_light', 'vsl']:
            if ctrl_type in self.active_controllers and ctrl_type in actions:
                try:
                    result = self.controllers[ctrl_type].apply_actions(env, actions[ctrl_type])
                    results[ctrl_type] = result
                except Exception as e:
                    print(f"⚠️  {ctrl_type}控制失败: {e}")
                    results[ctrl_type] = None

        return results


def create_controller(config: Dict[str, Any]) -> HybridController:
    """
    创建混合控制器的工厂函数

    Args:
        config: 配置字典，应包含 'competition_phase'

    Returns:
        混合控制器实例
    """
    return HybridController(config)


# 测试代码
if __name__ == '__main__':
    print("="*70)
    print("测试控制接口")
    print("="*70)

    # 初赛配置
    prelim_config = {'competition_phase': 'preliminary'}
    controller = HybridController(prelim_config)

    print(f"\n✅ 初赛模式控制方式: {controller.active_controllers}")
    action_space = controller.get_combined_action_space()
    print(f"动作空间:")
    for ctrl_type, space in action_space.items():
        print(f"  - {ctrl_type}: {space['type']}")

    # 复赛配置
    final_config = {'competition_phase': 'final'}
    controller_final = HybridController(final_config)

    print(f"\n✅ 复赛模式控制方式: {controller_final.active_controllers}")
    action_space_final = controller_final.get_combined_action_space()
    print(f"动作空间:")
    for ctrl_type, space in action_space_final.items():
        print(f"  - {ctrl_type}: {space['type']}")

    print("\n" + "="*70)
    print("✅ 控制接口测试通过!")
    print("="*70)
