"""
系统测试脚本
功能：验证系统基本功能是否正常
"""

import os
import sys

print("="*70)
print("🧪 智能交通控制系统 - 功能测试")
print("="*70)

# 测试1: Python版本
print("\n1️⃣  检查Python版本...")
python_version = sys.version_info
if python_version.major >= 3 and python_version.minor >= 8:
    print(f"   ✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
else:
    print(f"   ❌ Python版本过低: {python_version.major}.{python_version.minor}.{python_version.micro}")
    print("   需要 Python 3.8+")
    sys.exit(1)

# 测试2: 依赖包
print("\n2️⃣  检查依赖包...")
required_packages = {
    'numpy': 'NumPy',
    'torch': 'PyTorch',
    'pandas': 'Pandas',
}

missing_packages = []

for package, name in required_packages.items():
    try:
        mod = __import__(package)
        version = getattr(mod, '__version__', 'unknown')
        print(f"   ✅ {name}: {version}")
    except ImportError:
        print(f"   ❌ {name}: 未安装")
        missing_packages.append(package)

# PyTorch Geometric特殊检查
print("\n3️⃣  检查PyTorch Geometric...")
try:
    import torch_geometric
    version = getattr(torch_geometric, '__version__', 'unknown')
    print(f"   ✅ PyTorch Geometric: {version}")
except ImportError:
    print(f"   ⚠️  PyTorch Geometric: 未安装 (可选)")
    print("      如需图神经网络功能，请安装:")
    print("      pip install torch_geometric")

# TraCI检查
print("\n4️⃣  检查TraCI...")
try:
    import traci
    print(f"   ✅ TraCI: 已安装")
except ImportError:
    print(f"   ⚠️  TraCI: 未安装")
    print("      SUMO接口将不可用")
    missing_packages.append('traci')

# openpyxl检查
print("\n5️⃣  检查openpyxl...")
try:
    import openpyxl
    version = getattr(openpyxl, '__version__', 'unknown')
    print(f"   ✅ openpyxl: {version}")
except ImportError:
    print(f"   ⚠️  openpyxl: 未安装")
    print("      XLSX生成功能将不可用")
    missing_packages.append('openpyxl')

# 测试3: SUMO
print("\n6️⃣  检查SUMO...")
sumo_available = False
try:
    import shutil
    sumo_path = shutil.which('sumo')
    if sumo_path:
        print(f"   ✅ SUMO: {sumo_path}")
        sumo_available = True
    else:
        # 尝试常见路径
        common_paths = [
            r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe",
            r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe",
        ]
        for path in common_paths:
            if os.path.exists(path):
                print(f"   ✅ SUMO: {path}")
                sumo_available = True
                break

        if not sumo_available:
            print(f"   ❌ SUMO: 未找到")
            print("      请确保SUMO已安装并添加到系统PATH")
except Exception as e:
    print(f"   ⚠️  SUMO检查失败: {e}")

# 测试4: 项目文件
print("\n7️⃣  检查项目文件...")
required_files = [
    'train.py',
    'quick_start.py',
    'configs/training_config.json',
    'src/models/traffic_controller.py',
    'src/env/sumo_env.py',
    'src/algorithms/training.py',
    'src/evaluation/results_generator.py',
]

for filepath in required_files:
    if os.path.exists(filepath):
        print(f"   ✅ {filepath}")
    else:
        print(f"   ❌ {filepath}: 不存在")

# 测试5: SUMO环境文件
print("\n8️⃣  检查SUMO环境文件...")
sumo_files = [
    '仿真环境_初赛_1.0/仿真环境-初赛/sumo.sumocfg',
    '仿真环境_初赛_1.0/仿真环境-初赛/net.xml',
    '仿真环境_初赛_1.0/仿真环境-初赛/routes.xml',
]

for filepath in sumo_files:
    if os.path.exists(filepath):
        print(f"   ✅ {filepath}")
    else:
        print(f"   ❌ {filepath}: 不存在")

# 测试6: 创建目录
print("\n9️⃣  创建必要目录...")
directories = ['checkpoints', 'logs', 'results', 'data']
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"   ✅ {directory}/")

# 测试7: 模块导入
print("\n🔟 检查模块导入...")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.utils import get_device, count_parameters
    print(f"   ✅ 工具函数导入成功")
except Exception as e:
    print(f"   ❌ 工具函数导入失败: {e}")

try:
    from src.models import create_model_from_config
    print(f"   ✅ 模型模块导入成功")
except Exception as e:
    print(f"   ❌ 模型模块导入失败: {e}")

# 总结
print("\n" + "="*70)
print("📋 测试总结")
print("="*70)

if missing_packages:
    print(f"\n⚠️  缺失的包:")
    for pkg in missing_packages:
        print(f"   - {pkg}")
    print(f"\n💡 安装命令:")
    print(f"   pip install -r requirements.txt")

if not sumo_available:
    print(f"\n⚠️  SUMO未安装或未添加到PATH")
    print(f"💡 下载地址: https://www.eclipse.org/sumo/")

print(f"\n✅ 系统测试完成!")
print(f"\n💡 下一步:")
print(f"   1. 安装缺失的依赖")
print(f"   2. 运行快速测试: python quick_start.py")
print(f"   3. 开始训练: python train.py")
