#!/bin/bash

################################################################################
# 初赛一键训练脚本
#
# 功能：
#   - 自动检查系统环境
#   - Phase 1: 世界模型预训练
#   - Phase 2: 5级课程学习（从简单到复杂）
#   - 自动保存最佳模型用于提交
#
# 使用方法：
#   ./run_preliminary.sh                    # 完整训练（Phase 1 + Phase 2）
#   ./run_preliminary.sh --skip-phase1      # 跳过Phase 1，仅训练Phase 2
#   ./run_preliminary.sh --start-level 3    # 从Level 3开始训练
################################################################################

set -e  # 遇到错误立即退出

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
    echo ""
}

# 获取脚本所在目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

# 解析命令行参数
SKIP_PHASE1=false
START_LEVEL=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-phase1)
            SKIP_PHASE1=true
            shift
            ;;
        --start-level)
            START_LEVEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "初赛一键训练脚本"
            echo ""
            echo "使用方法："
            echo "  $0                            # 完整训练（Phase 1 + Phase 2）"
            echo "  $0 --skip-phase1              # 跳过Phase 1，仅训练Phase 2"
            echo "  $0 --start-level 3            # 从Level 3开始训练"
            echo ""
            echo "选项："
            echo "  --skip-phase1                 跳过Phase 1预训练"
            echo "  --start-level LEVEL           从指定级别开始（1-5）"
            echo "  -h, --help                    显示帮助信息"
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            echo "使用 $0 --help 查看帮助"
            exit 1
            ;;
    esac
done

# ============================================================================
# 欢迎信息
# ============================================================================
print_header "初赛训练 - 以粒控流"

log_info "配置文件: configs/competition_preliminary.yaml"
log_info "训练阶段:"
if [ "$SKIP_PHASE1" = true ]; then
    log_info "  - Phase 1: [跳过]"
else
    log_info "  - Phase 1: 世界模型预训练（GNN + RSSM）"
fi
log_info "  - Phase 2: PPO课程学习（5个级别）"
if [ "$START_LEVEL" -gt 1 ]; then
    log_warning "  - 从Level $START_LEVEL开始训练"
fi
echo ""

# ============================================================================
# 环境检查
# ============================================================================
print_header "环境检查"

log_info "检查Python环境..."
python --version

log_info "检查关键依赖..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || log_error "PyTorch未安装"
python -c "import gymnasium; print('Gymnasium: 已安装')" 2>/dev/null || log_error "Gymnasium未安装"
python -c "import torch_geometric; print('PyTorch Geometric: 已安装')" 2>/dev/null || log_error "PyTorch Geometric未安装"

log_info "检查GPU..."
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}'); print(f'GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null

echo ""

# ============================================================================
# 检查已有检查点
# ============================================================================
print_header "检查已有进度"

CHECKPOINT_DIR="checkpoints/competition/preliminary"

# 检查Phase 1
if [ -f "$CHECKPOINT_DIR/phase1/world_model_final.pth" ]; then
    log_success "发现Phase 1检查点"
    if [ "$SKIP_PHASE1" = false ]; then
        read -p "是否跳过Phase 1训练？[Y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            SKIP_PHASE1=true
            log_info "将跳过Phase 1训练"
        fi
    fi
else
    log_info "未发现Phase 1检查点，将进行Phase 1训练"
fi

# 检查Phase 2各级别
for level in {1..5}; do
    if [ -f "$CHECKPOINT_DIR/level${level}/custom_ppo.zip" ]; then
        log_success "发现Level $level检查点"
    fi
done

echo ""

# ============================================================================
# 确认开始训练
# ============================================================================
print_header "准备就绪"

log_warning "训练预计需要较长时间（取决于硬件配置）"
log_warning "Phase 1: 约2-3小时（如果执行）"
log_warning "Phase 2: 约8-12小时（5个级别）"
echo ""
read -p "是否开始训练？[Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    log_info "已取消"
    exit 0
fi

echo ""

# ============================================================================
# 创建必要目录
# ============================================================================
log_info "创建必要的目录..."
mkdir -p "$CHECKPOINT_DIR/phase1"
mkdir -p "$CHECKPOINT_DIR"/level{1,2,3,4,5}
mkdir -p logs/preliminary/level{1,2,3,4,5}
log_success "目录创建完成"

echo ""

# ============================================================================
# 开始训练
# ============================================================================

# 构建命令
TRAIN_CMD="python train_preliminary.py --config configs/competition_preliminary.yaml"

if [ "$SKIP_PHASE1" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --skip-phase1"
fi

if [ "$START_LEVEL" -gt 1 ]; then
    TRAIN_CMD="$TRAIN_CMD --start-level $START_LEVEL"
fi

# 记录开始时间
START_TIME=$(date +%s)

print_header "开始训练"

log_info "执行命令: $TRAIN_CMD"
echo ""

# 执行训练（并记录日志）
LOG_FILE="logs/preliminary/training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

if eval ${TRAIN_CMD} 2>&1 | tee "$LOG_FILE"; then
    # 训练成功
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))

    echo ""
    print_header "训练完成！"
    log_success "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
    log_success "日志文件: $LOG_FILE"
    echo ""

    # 显示最终模型位置
    if [ -f "$CHECKPOINT_DIR/level5/custom_ppo.zip" ]; then
        log_success "最终模型: $CHECKPOINT_DIR/level5/custom_ppo.zip"
        echo ""
        log_info "下一步："
        log_info "  1. 评估模型性能"
        log_info "     python evaluate_v4_ideal.py --checkpoint $CHECKPOINT_DIR/level5/custom_ppo.zip"
        log_info "  2. 如果满意，可用于提交比赛"
        echo ""
    fi

else
    # 训练失败
    echo ""
    print_header "训练失败"
    log_error "请查看日志文件: $LOG_FILE"
    log_error "检查错误信息并修复问题"
    exit 1
fi

echo "================================================================================"
log_success "所有任务完成！"
echo "================================================================================"
