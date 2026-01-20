"""
智能交通协同控制系统
"""

__version__ = "4.0.0"
__author__ = "Traffic Control AI Team"

# 延迟导入，避免循环依赖
def __getattr__(name):
    if name == 'TrafficController':
        from src.controller.traffic_controller import TrafficController
        return TrafficController
    elif name == 'create_model_from_config':
        from src.models.factory import create_model_from_config
        return create_model_from_config
    elif name == 'SumoEnvironment':
        from src.env.sumo_env import SumoEnvironment
        return SumoEnvironment
    elif name == 'EfficientDataCollector':
        from src.training.data_collector import EfficientDataCollector
        return EfficientDataCollector
    elif name == 'Trainer':
        from src.training.trainer import Trainer
        return Trainer
    elif name == 'XLSXResultGenerator':
        from src.evaluation.result_generator import XLSXResultGenerator
        return XLSXResultGenerator
    elif name == 'generate_evaluation_report':
        from src.evaluation.report import generate_evaluation_report
        return generate_evaluation_report
    raise AttributeError(f"module {__name__} has no attribute {name}")
