#!/bin/bash

################################################################################
# v5训练系统完整性检查脚本
#
# 验证v5轻量级架构的所有训练脚本和依赖是否正确配置
################################################################################

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[CHECK]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo "================================================================================"
echo "v5轻量级架构训练系统完整性检查"
echo "================================================================================"
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

# 检查Python环境
log_info "Python环境..."
python --version
log_success "Python可用"

# 检查关键依赖
log_info "检查关键依赖..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null && log_success "PyTorch已安装" || log_error "PyTorch未安装"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null && log_success "NumPy已安装" || log_error "NumPy未安装"
python -c "import yaml; print('PyYAML: 已安装')" 2>/dev/null && log_success "PyYAML已安装" || log_error "PyYAML未安装"
python -c "import gymnasium; print('Gymnasium: 已安装')" 2>/dev/null && log_success "Gymnasium已安装" || log_error "Gymnasium未安装"
python -c "import torch_geometric; print('PyTorch Geometric: 已安装')" 2>/dev/null && log_success "PyTorch Geometric已安装" || log_warning "PyTorch Geometric未安装（v5可选）"

echo ""

# 检查v5训练模块
log_info "检查v5训练模块..."
python -c "from src.models.v5_lightweight import create_lightweight_policy_v5" 2>/dev/null && log_success "v5轻量级模型 可导入" || log_error "v5轻量级模型 导入失败"
python -c "from src.training.custom_ppo_trainer import CustomPPOTrainer" 2>/dev/null && log_success "PPO训练器 可导入" || log_error "PPO训练器 导入失败"
python -c "from src.training.ocr_rewards import create_ocr_reward_calculator" 2>/dev/null && log_success "OCR奖励计算器 可导入" || log_error "OCR奖励计算器 导入失败"
python -c "from src.env.sparse_controller import create_sparse_controller" 2>/dev/null && log_success "稀疏控制器 可导入" || log_error "稀疏控制器 导入失败"
python -c "from src.env.gym_wrapper import GymSumoEnv" 2>/dev/null && log_success "Gym环境 可导入" || log_error "Gym环境 导入失败"

echo ""

# 检查v5配置文件
log_info "检查v5配置文件..."
if [ -f "configs/phase1_lite.yaml" ]; then
    log_success "configs/phase1_lite.yaml 存在"
else
    log_error "configs/phase1_lite.yaml 不存在（v5必需配置）"
fi

if [ -f "configs/competition_preliminary.yaml" ]; then
    log_success "configs/competition_preliminary.yaml 存在（可选）"
else
    log_warning "configs/competition_preliminary.yaml 不存在（可选配置）"
fi

echo ""

# 检查GPU
log_info "检查GPU..."
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}'); print(f'GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null

echo ""

# 检查目录结构
log_info "创建v5目录结构..."
mkdir -p checkpoints/phase1_lite
mkdir -p logs/phase1_lite
mkdir -p runs/phase1_lite
log_success "目录结构已创建"

echo ""

# 测试v5训练脚本导入
log_info "测试v5训练脚本导入..."
python -c "
import sys
sys.path.insert(0, '.')
from train_phase1_lite import main
print('v5训练脚本导入成功')
" 2>/dev/null && log_success "v5训练脚本可正常导入" || log_error "v5训练脚本导入失败"

echo ""

# 测试v5测试脚本
log_info "测试v5测试脚本..."
if [ -f "test_v5_architecture.py" ]; then
    log_success "test_v5_architecture.py 存在"
else
    log_error "test_v5_architecture.py 不存在"
fi

echo ""
echo "================================================================================"
echo "✅ v5训练系统检查完成！"
echo ""
echo "如果所有检查都通过，可以开始训练："
echo "  python train_phase1_lite.py --stage all                    # 完整训练（Stage 1 + Stage 2）"
echo "  python train_phase1_lite.py --stage stage1                # 只运行Stage 1（行为克隆）"
echo "  python train_phase1_lite.py --stage stage2                # 只运行Stage 2（PPO微调）"
echo ""
echo "测试架构："
echo "  python test_v5_architecture.py                            # 验证v5架构"
echo "================================================================================"
