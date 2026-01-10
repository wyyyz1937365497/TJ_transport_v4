"""配置管理模块 - 使用OmegaConf和YAML"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

import yaml
from omegaconf import OmegaConf


@dataclass
class Config:
    """配置数据类"""

    device: str = "cuda"
    seed: int = 42
    precision: str = "bf16"

    # 路径配置
    root_dir: str = "."
    data_dir: str = "data"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    result_dir: str = "results"

    # 子配置
    environment: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    wandb: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理：创建目录"""
        self.root_dir = Path(self.root_dir)
        self.data_dir = self.root_dir / self.data_dir
        self.log_dir = self.root_dir / self.log_dir
        self.checkpoint_dir = self.root_dir / self.checkpoint_dir
        self.result_dir = self.root_dir / self.result_dir

        # 创建目录
        for dir_path in [self.data_dir, self.log_dir, self.checkpoint_dir, self.result_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从YAML文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # 处理路径配置
        paths = data.pop("paths", {})
        paths.pop("root_dir", None)  # 移除root_dir，使用当前目录

        # 将paths中的值合并到顶层
        for key, value in paths.items():
            data[key] = value

        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """从字典创建配置"""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        from dataclasses import asdict
        return asdict(self)

    def save(self, path: str):
        """保存配置到YAML文件"""
        data = self.to_dict()

        # 处理路径
        paths = {}
        path_keys = ["data_dir", "log_dir", "checkpoint_dir", "result_dir"]
        for key in path_keys:
            value = data.pop(key, None)
            if value is not None:
                paths[key] = str(value)

        if paths:
            data["paths"] = paths

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """
    加载配置

    Args:
        config_path: 配置文件路径，默认使用 config/base.yaml
        overrides: 配置覆盖

    Returns:
        Config对象
    """
    # 默认配置路径
    if config_path is None:
        # 查找相对于脚本的配置文件
        script_dir = Path(__file__).parent.parent.parent
        config_path = script_dir / "config" / "base.yaml"

    # 加载配置
    if isinstance(config_path, (str, Path)):
        config = Config.from_yaml(str(config_path))
    else:
        config = config_path

    # 应用覆盖
    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                # 尝试设置嵌套配置
                if "." in key:
                    parts = key.split(".")
                    obj = config
                    for part in parts[:-1]:
                        if hasattr(obj, part):
                            obj = getattr(obj, part)
                        elif isinstance(obj, dict) and part in obj:
                            obj = obj[part]
                    if isinstance(obj, dict):
                        obj[parts[-1]] = value

    return config


def merge_configs(base: Config, override: Config) -> Config:
    """合并两个配置"""
    base_dict = base.to_dict()
    override_dict = override.to_dict()

    def deep_merge(base_val: Any, override_val: Any) -> Any:
        if isinstance(base_val, dict) and isinstance(override_val, dict):
            result = base_val.copy()
            for key, val in override_val.items():
                if key in result:
                    result[key] = deep_merge(result[key], val)
                else:
                    result[key] = val
            return result
        return override_val

    merged = deep_merge(base_dict, override_dict)
    return Config.from_dict(merged)
