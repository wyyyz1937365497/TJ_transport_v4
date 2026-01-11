"""
算法模块
"""

# 可选导入 - 避免在缺少依赖时失败
try:
    from .training import Trainer, train_full_pipeline
    _training_available = True
except ImportError:
    _training_available = False

__all__ = []

if _training_available:
    __all__.extend(['Trainer', 'train_full_pipeline'])
