#!/usr/bin/env python3
"""
配置文件统一性验证脚本

检查所有训练脚本是否使用统一的配置文件。
"""

import re
from pathlib import Path
import sys

def extract_default_config(script_path):
    """从Python脚本中提取默认配置文件路径"""
    try:
        content = script_path.read_text()
        # 查找argparse中的default配置
        match = re.search(r"--config.*?default=['\"]([^'\"]+)['\"]", content, re.DOTALL)
        if match:
            return match.group(1)
        # 查找load_config函数的默认参数
        match = re.search(r"def load_config.*?config_path\s*=\s*['\"]([^'\"]+)['\"]", content, re.DOTALL)
        if match:
            return match.group(1)
        return None
    except Exception as e:
        print(f"  [ERROR] 读取失败: {e}")
        return None

def extract_shell_config(script_path):
    """从Shell脚本中提取配置文件路径"""
    try:
        content = script_path.read_text()
        match = re.search(r"CONFIG_FILE=[\"']?([^\"'\s]+)", content)
        if match:
            config = match.group(1)
            # 如果包含变量引用，认为是正确的（运行时会展开）
            if "${PROJECT_ROOT}" in config or "$PROJECT_ROOT" in config:
                config = config.replace("${PROJECT_ROOT}", "").replace("$PROJECT_ROOT", "")
                # 移除可能的前导斜杠
                config = config.lstrip("/")
                return config
            return config
        return None
    except Exception as e:
        print(f"  [ERROR] 读取失败: {e}")
        return None

def main():
    project_root = Path("/home/wyyyz/TJ_transport_v4")

    # 定义需要检查的训练脚本
    training_scripts = {
        "Python脚本（标准训练流程）": [
            (project_root / "train_phase1.py", "configs/competition.yaml"),
            (project_root / "train_phase2.py", "configs/competition.yaml"),
            (project_root / "train_phase3.py", "configs/competition.yaml"),
        ],
        "Python脚本（专用训练）": [
            (project_root / "train_preliminary.py", "configs/competition_preliminary.yaml"),  # 初赛专用
        ],
        "Shell脚本": [
            (project_root / "train_all.sh", None),  # 使用变量，需要特殊处理
        ]
    }

    expected_config = "configs/competition.yaml"
    issues = []

    print("="*80)
    print("配置文件统一性验证")
    print("="*80)
    print(f"\n期望配置文件: {expected_config}\n")

    for category, scripts in training_scripts.items():
        print(f"\n[{category}]")
        print("-"*80)

        for item in scripts:
            if isinstance(item, tuple):
                script, expected_config_for_script = item
            else:
                script = item
                expected_config_for_script = None

            if not script.exists():
                print(f"  ⚠️  {script.name}: 文件不存在")
                continue

            # 确定期望的配置文件
            if expected_config_for_script:
                exp_config = expected_config_for_script
            else:
                exp_config = expected_config

            # 提取实际使用的配置
            if category.startswith("Python"):
                config = extract_default_config(script)
            else:
                config = extract_shell_config(script)

            if config is None:
                print(f"  ❌ {script.name}: 未找到配置")
                if expected_config_for_script is None:  # 只对标准训练脚本报错
                    issues.append(f"{script.name}: 未找到配置")
            elif config == exp_config:
                print(f"  ✅ {script.name}: {config}")
            else:
                print(f"  ⚠️  {script.name}: {config} (应为 {exp_config})")
                if expected_config_for_script is None:  # 只对标准训练脚本报错
                    issues.append(f"{script.name}: 使用 {config} 而不是 {exp_config}")

    # 检查配置文件是否存在
    print(f"\n[配置文件检查]")
    print("-"*80)
    config_file = project_root / expected_config
    if config_file.exists():
        print(f"  ✅ {expected_config} 存在")
    else:
        print(f"  ❌ {expected_config} 不存在")
        issues.append(f"{expected_config} 文件不存在")

    # 总结
    print("\n" + "="*80)
    if not issues:
        print("✅ 所有训练脚本配置统一！")
        return 0
    else:
        print("⚠️  发现配置不统一的问题：")
        print("-"*80)
        for issue in issues:
            print(f"  ❌ {issue}")
        print("\n建议：")
        print("  1. 修改训练脚本的默认配置文件为 'configs/competition.yaml'")
        print("  2. 确保 configs/competition.yaml 包含所有必要的配置")
        print("  3. 参考文档: docs/CONFIG_UNIFICATION.md")
        return 1

if __name__ == "__main__":
    sys.exit(main())
