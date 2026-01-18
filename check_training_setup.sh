#!/bin/bash

################################################################################
# 训练系统完整性检查脚本
#
# 验证所有训练脚本和依赖是否正确配置
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
echo "训练系统完整性检查"
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
python -c "import torch_geometric; print('PyTorch Geometric: 已安装')" 2>/dev/null && log_success "PyTorch Geometric已安装" || log_error "PyTorch Geometric未安装"

echo ""

# 检查训练模块
log_info "检查训练模块..."
python -c "from src.training.world_model_train_v4 import WorldModelTrainer" 2>/dev/null && log_success "WorldModelTrainer 可导入" || log_error "WorldModelTrainer 导入失败"
python -c "from src.training.custom_ppo_trainer import CustomPPOTrainer" 2>/dev/null && log_success "CustomPPOTrainer 可导入" || log_error "CustomPPOTrainer 导入失败"
python -c "from src.models.ideal_policy_v4 import create_ideal_traffic_policy_v4" 2>/dev/null && log_success "Policy模型 可导入" || log_error "Policy模型 导入失败"
python -c "from src.env.competition_env import CompetitionSumoEnv" 2>/dev/null && log_success "环境 可导入" || log_error "环境 导入失败"

echo ""

# 检查脚本权限
log_info "检查脚本权限..."
for script in train_all.sh train_phase2_curriculum.sh train_quick_test.sh; do
    if [ -x "${script}" ]; then
        log_success "${script} 可执行"
    else
        log_warning "${script} 不可执行，正在添加执行权限..."
        chmod +x "${script}"
        log_success "${script} 已添加执行权限"
    fi
done

echo ""

# 检查配置文件
log_info "检查配置文件..."
if [ -f "configs/competition.yaml" ]; then
    log_success "configs/competition.yaml 存在"
else
    log_error "configs/competition.yaml 不存在"
fi

if [ -f "configs/optimized_ppo.yaml" ]; then
    log_success "configs/optimized_ppo.yaml 存在"
else
    log_warning "configs/optimized_ppo.yaml 不存在（可选）"
fi

echo ""

# 检查GPU
log_info "检查GPU..."
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}'); print(f'GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null

echo ""

# 检查目录结构
log_info "检查目录结构..."
mkdir -p checkpoints/competition/phase1
mkdir -p checkpoints/competition/curriculum
mkdir -p logs/phase1
mkdir -p logs/phase2
mkdir -p logs/phase3
log_success "目录结构已创建"

echo ""

# 测试导入
log_info "测试完整导入..."
python -c "
import sys
sys.path.insert(0, '.')
from train_phase1 import main as p1_main
from train_phase2 import main as p2_main
print('所有训练脚本导入成功')
" 2>/dev/null && log_success "训练脚本可正常导入" || log_warning "训练脚本导入有问题"

echo ""
echo "================================================================================"
echo "检查完成！"
echo ""
echo "如果所有检查都通过，可以开始训练："
echo "  ./train_all.sh                    # 训练所有阶段"
echo "  ./train_quick_test.sh             # 快速测试（5-10分钟）"
echo "  python train_phase1.py            # 单独训练Phase 1"
echo "================================================================================"
