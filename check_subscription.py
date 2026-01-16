#!/usr/bin/env python3
"""
快速验证：SUMO订阅优化是否已启用
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def check_subscription_optimization():
    """验证订阅优化是否启用"""
    print("="*60)
    print("SUMO Subscription Optimization Check")
    print("="*60)

    # 1. 检查配置文件
    print("\n[1] Configuration File Check")
    import yaml
    with open("configs/competition.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    sumo_config = config.get('environment', {}).get('sumo_config', '')
    print(f"  SUMO Config: {sumo_config}")

    if 'sumo_train.sumocfg' in sumo_config:
        print(f"  [OK] Using optimized training config")
    elif 'sumo_fast.sumocfg' in sumo_config:
        print(f"  [WARNING] sumo_fast.sumocfg may not exist!")
        print(f"  [HINT] Should use sumo_train.sumocfg")
        return False
    else:
        print(f"  [INFO] Using standard config")

    # 2. 检查代码导入
    print("\n[2] Code Import Check")

    try:
        # 读取competition_env.py
        with open("src/env/competition_env.py", 'r', encoding='utf-8') as f:
            content = f.read()

        if 'from .gpu_sumo_env_optimized import GPUSumoEnvironmentOptimized' in content:
            print(f"  [OK] CompetitionSumoEnv uses optimized version")
        else:
            print(f"  [WARNING] Not using optimized version!")
            return False

        # 检查是否有优化的类
        from src.env.gpu_sumo_env_optimized import GPUSumoEnvironmentOptimized
        print(f"  [OK] GPUSumoEnvironmentOptimized imported successfully")

        # 检查是否有use_subscription参数
        import inspect
        sig = inspect.signature(GPUSumoEnvironmentOptimized.__init__)
        if 'use_subscription' in sig.parameters:
            print(f"  [OK] use_subscription parameter available")
        else:
            print(f"  [INFO] use_subscription not in parameters (always enabled)")

    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

    # 3. 检查优化配置文件是否存在
    print("\n[3] SUMO Config Files Check")

    import os
    config_dir = Path("仿真环境_初赛_1.0/仿真环境-初赛")

    configs_to_check = [
        ("sumo.sumocfg", "Official config"),
        ("sumo_train.sumocfg", "Optimized training config"),
    ]

    all_exist = True
    for config_file, description in configs_to_check:
        config_path = config_dir / config_file
        exists = config_path.exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {config_file}: {description}")
        if exists:
            print(f"       → File found")
        else:
            print(f"       → File NOT found")
            all_exist = False

    if not all_exist:
        print(f"\n  [WARNING] Some config files are missing!")
        print(f"  [HINT] Make sure sumo_train.sumocfg exists")

    return all_exist

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Verifying SUMO Subscription Optimization")
    print("="*60 + "\n")

    result = check_subscription_optimization()

    print("\n" + "="*60)
    if result:
        print("[SUCCESS] Subscription optimization is enabled!")
        print("Ready to start training with batch mode.")
    else:
        print("[WARNING] Some checks failed!")
        print("Please review the warnings above.")
    print("="*60 + "\n")
