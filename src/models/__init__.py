"""
神经网络模型模块（模仿学习 + MPC架构）

主要组件：
- SimplifiedICVPolicy: 简化的策略网络，用于模仿学习
- SafetyShield: 安全防护盾模块（可选）
"""

# 延迟导入，避免循环依赖
def __getattr__(name):
    if name in [
        'SimplifiedICVPolicy',
    ]:
        from src.models.simplified_icv_policy import SimplifiedICVPolicy
        return SimplifiedICVPolicy

    if name in [
        'SafetyShield',
    ]:
        from src.models.safety_shield import SafetyShield
        return SafetyShield

    raise AttributeError(f"module {__name__} has no attribute {name}")

# 导出所有模块
__all__ = [
    'SimplifiedICVPolicy',
    'SafetyShield',
]
