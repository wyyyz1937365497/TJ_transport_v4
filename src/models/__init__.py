"""
神经网络模型模块
"""

# 可选导入 - 避免在缺少依赖时失败
try:
    from .gnn import RiskSensitiveGNN, GraphBuilder
    _gnn_available = True
except ImportError:
    _gnn_available = False

try:
    from .world_model import ProgressiveWorldModel, WorldModelLoss
    _world_model_available = True
except ImportError:
    _world_model_available = False

try:
    from .controller import InfluenceDrivenController, ValueNetwork, CostNetwork
    _controller_available = True
except ImportError:
    _controller_available = False

try:
    from .safety import DualModeSafetyShield, SafetyMonitor
    _safety_available = True
except ImportError:
    _safety_available = False

try:
    from .traffic_controller import TrafficController, create_model_from_config
    _traffic_controller_available = True
except ImportError:
    _traffic_controller_available = False

try:
    from .sb3_policy import create_custom_policy
    _sb3_policy_available = True
except ImportError:
    _sb3_policy_available = False

try:
    from .sb3_full_policy import FullTrafficController, FullTrafficActorCriticPolicy, create_full_traffic_policy
    _sb3_full_policy_available = True
except ImportError:
    _sb3_full_policy_available = False

# 导出可用的模块
__all__ = []

if _gnn_available:
    __all__.extend(['RiskSensitiveGNN', 'GraphBuilder'])

if _world_model_available:
    __all__.extend(['ProgressiveWorldModel', 'WorldModelLoss'])

if _controller_available:
    __all__.extend(['InfluenceDrivenController', 'ValueNetwork', 'CostNetwork'])

if _safety_available:
    __all__.extend(['DualModeSafetyShield', 'SafetyMonitor'])

if _traffic_controller_available:
    __all__.extend(['TrafficController', 'create_model_from_config'])

if _sb3_policy_available:
    __all__.extend(['create_custom_policy'])

if _sb3_full_policy_available:
    __all__.extend(['FullTrafficController', 'FullTrafficActorCriticPolicy', 'create_full_traffic_policy'])
