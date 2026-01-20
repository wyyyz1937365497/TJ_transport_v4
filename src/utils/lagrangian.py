import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class LagrangianMultiplier(nn.Module):
    """
    拉格朗日乘子控制器 - 用于处理带约束的优化问题 (Constrained RL)
    
    实现 PID-Lagrangian 更新逻辑:
    lambda_new = lambda_old + Kp * error + Ki * integral_error + Kd * derivative_error
    其中 error = cost - limit
    """
    
    def __init__(
        self,
        cost_limit: float,
        kp: float = 0.05,
        ki: float = 0.005,
        kd: float = 0.0,
        init_value: float = 1.0,
        min_value: float = 0.0,
        max_value: float = 100.0
    ):
        super().__init__()
        self.cost_limit = cost_limit
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_value = min_value
        self.max_value = max_value
        
        # 乘子参数 (使用Parameter使其可保存，但通常不通过梯度更新，而是手动更新)
        self.param = nn.Parameter(torch.tensor(init_value), requires_grad=False)
        
        # 状态追踪
        self.integral_error = 0.0
        self.last_error = 0.0
        
    def update(self, current_cost: float) -> float:
        """更新拉格朗日乘子"""
        error = current_cost - self.cost_limit
        
        # 积分项
        self.integral_error += error
        # 微分项
        derivative = error - self.last_error
        self.last_error = error
        
        # PID 更新
        delta = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative)
        
        # 更新值并截断
        new_val = self.param.item() + delta
        new_val = np.clip(new_val, self.min_value, self.max_value)
        
        self.param.data.fill_(new_val)
        
        return new_val
        
    def get_value(self) -> float:
        return self.param.item()

    def forward(self, loss_val: torch.Tensor) -> torch.Tensor:
        """应用惩罚: loss + lambda * cost"""
        # 注意：这里仅返回加权后的值，实际使用时通常是 loss + lambda * cost
        # 此方法主要用于方便调用
        return loss_val * self.param
