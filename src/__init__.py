"""
智能交通协同控制系统（模仿学习 + MPC架构）

主要组件：
- 环境接口：CompetitionSumoEnv
- 策略网络：SimplifiedICVPolicy
- MPC控制器：MPCController
- 车辆评分：RuleBasedVehicleScorer
"""

__version__ = "5.0.0"
__author__ = "Traffic Control AI Team"
__date__ = "2025-01-24"

# 延迟导入，避免循环依赖
def __getattr__(name):
    if name == 'CompetitionSumoEnv':
        from src.env.competition_env import CompetitionSumoEnv
        return CompetitionSumoEnv
    elif name == 'SimplifiedICVPolicy':
        from src.models.simplified_icv_policy import SimplifiedICVPolicy
        return SimplifiedICVPolicy
    elif name == 'MPCController':
        from src.mpc.core.mpc_controller import MPCController
        return MPCController
    elif name == 'RuleBasedVehicleScorer':
        from src.env.rule_based_scorer import RuleBasedVehicleScorer
        return RuleBasedVehicleScorer
    raise AttributeError(f"module {__name__} has no attribute {name}")
