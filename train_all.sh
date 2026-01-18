#!/bin/bash

################################################################################
# 完整训练流程 - 自动检测并执行所有阶段
#
# 功能：
# 1. 自动检测已完成的训练阶段
# 2. 从检查点恢复训练
# 3. 支持断点续训
# 4. 详细的训练日志
#
# 使用方法：
#   ./train_all.sh                    # 训练所有阶段
#   ./train_all.sh --from-phase 2     # 从第2阶段开始
#   ./train_all.sh --resume           # 自动恢复训练
################################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/competition"
LOG_DIR="${PROJECT_ROOT}/logs"
CONFIG_FILE="${PROJECT_ROOT}/configs/competition.yaml"

# 创建必要目录
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${LOG_DIR}/phase1"
mkdir -p "${LOG_DIR}/phase2"
mkdir -p "${LOG_DIR}/phase3"

################################################################################
# 工具函数
################################################################################

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
}

# 检查文件是否存在
checkpoint_exists() {
    local checkpoint_path="$1"
    if [ -f "${checkpoint_path}" ]; then
        return 0
    else
        return 1
    fi
}

# 获取检查点文件大小（MB）
get_checkpoint_size() {
    local checkpoint_path="$1"
    if [ -f "${checkpoint_path}" ]; then
        du -m "${checkpoint_path}" | cut -f1
    else
        echo "0"
    fi
}

################################################################################
# Phase 1: 世界模型训练
################################################################################

train_phase1() {
    print_header "[PHASE 1] World Model Training"

    local checkpoint="${CHECKPOINT_DIR}/phase1/world_model_final.pth"

    if checkpoint_exists "${checkpoint}"; then
        local size=$(get_checkpoint_size "${checkpoint}")
        log_warning "Phase 1 checkpoint already exists: ${checkpoint} (${size}MB)"
        read -p "是否跳过 Phase 1？[Y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            log_success "Skipping Phase 1"
            return 0
        fi
        log_info "Retraining Phase 1..."
    fi

    # 备份现有检查点
    if checkpoint_exists "${checkpoint}"; then
        local backup_dir="${CHECKPOINT_DIR}/phase1/backup_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "${backup_dir}"
        mv "${checkpoint}" "${backup_dir}/"
        log_info "Old checkpoint backed up to: ${backup_dir}"
    fi

    # 训练 Phase 1
    log_info "Starting Phase 1 training..."
    cd "${PROJECT_ROOT}"

    if python train_phase1.py 2>&1 | tee "${LOG_DIR}/phase1/$(date +%Y%m%d_%H%M%S).log"; then
        log_success "Phase 1 training completed!"
    else
        log_error "Phase 1 training failed!"
        exit 1
    fi
}

################################################################################
# Phase 2: PPO训练（课程学习）
################################################################################

train_phase2_stage() {
    local stage=$1
    local prev_checkpoint=$2

    local stage_config=(
        [1]="level:1,name:基础场景,max_vehicles:10,inflow:800"
        [2]="level:2,name:中等流量,max_vehicles:15,inflow:1200"
        [3]="level:3,name:高流量场景,max_vehicles:20,inflow:1800"
        [4]="level:4,name:极端场景,max_vehicles:32,inflow:2400"
        [5]="level:5,name:赛题场景,max_vehicles:32,inflow:2000"
    )

    local info="${stage_config[$stage]}"
    print_header "[PHASE 2] Stage ${stage}: ${info}"

    local checkpoint="${CHECKPOINT_DIR}/curriculum/level${stage}/custom_ppo.zip"

    if checkpoint_exists "${checkpoint}"; then
        local size=$(get_checkpoint_size "${checkpoint}")
        log_warning "Stage ${stage} checkpoint already exists: ${checkpoint} (${size}MB)"
        read -p "是否跳过 Stage ${stage}？[Y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            log_success "Skipping Stage ${stage}"
            return 0
        fi
        log_info "Retraining Stage ${stage}..."
    fi

    # 训练命令
    local cmd="python train_phase2.py --stage ${stage}"

    if [ -n "${prev_checkpoint}" ]; then
        cmd="${cmd} --prev-checkpoint ${prev_checkpoint}"
    fi

    log_info "Starting Stage ${stage} training..."
    cd "${PROJECT_ROOT}"

    if eval ${cmd} 2>&1 | tee "${LOG_DIR}/phase2/stage${stage}_$(date +%Y%m%d_%H%M%S).log"; then
        log_success "Stage ${stage} training completed!"

        # 返回检查点路径供下一阶段使用
        echo "${checkpoint}"
    else
        log_error "Stage ${stage} training failed!"
        exit 1
    fi
}

train_phase2() {
    print_header "[PHASE 2] PPO Training with Curriculum Learning"

    local phase1_checkpoint="${CHECKPOINT_DIR}/phase1/world_model_final.pth"

    if ! checkpoint_exists "${phase1_checkpoint}"; then
        log_error "Phase 1 checkpoint not found: ${phase1_checkpoint}"
        log_info "Please run Phase 1 first or use --from-phase 2 with a valid checkpoint"
        exit 1
    fi

    log_info "Using Phase 1 checkpoint: ${phase1_checkpoint}"

    # 训练所有课程阶段
    local current_checkpoint=""
    local stages=(1 2 3 4 5)

    for stage in "${stages[@]}"; do
        # 检查是否需要训练此阶段
        local checkpoint="${CHECKPOINT_DIR}/curriculum/level${stage}/custom_ppo.zip"

        if checkpoint_exists "${checkpoint}"; then
            local size=$(get_checkpoint_size "${checkpoint}")
            log_warning "Stage ${stage} already completed (${size}MB)"

            # 询问是否跳过
            read -p "跳过已完成的 Stage ${stage}？[Y/n] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                log_success "Skipping Stage ${stage}"
                current_checkpoint="${checkpoint}"
                continue
            fi
        fi

        # 训练当前阶段
        if [ ${stage} -eq 1 ]; then
            current_checkpoint=$(train_phase2_stage ${stage} "${phase1_checkpoint}")
        else
            current_checkpoint=$(train_phase2_stage ${stage} "${current_checkpoint}")
        fi
    done

    log_success "All Phase 2 stages completed!"
}

################################################################################
# Phase 3: 微调优化（可选）
################################################################################

train_phase3() {
    print_header "[PHASE 3] Fine-tuning and Optimization"

    local phase2_checkpoint="${CHECKPOINT_DIR}/curriculum/level5/custom_ppo.zip"

    if ! checkpoint_exists "${phase2_checkpoint}"; then
        log_error "Phase 2 final checkpoint not found: ${phase2_checkpoint}"
        log_info "Please complete Phase 2 first"
        return 1
    fi

    log_warning "Phase 3 is optional and requires manual configuration"
    log_info "Skipping Phase 3 (to be implemented)"
}

################################################################################
# 主函数
################################################################################

main() {
    print_header "完整训练流程 - 自动化脚本"

    log_info "项目根目录: ${PROJECT_ROOT}"
    log_info "检查点目录: ${CHECKPOINT_DIR}"
    log_info "日志目录: ${LOG_DIR}"

    # 解析命令行参数
    START_FROM_PHASE=1
    RESUME_MODE=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --from-phase)
                START_FROM_PHASE="$2"
                shift 2
                ;;
            --resume)
                RESUME_MODE=true
                shift
                ;;
            --help|-h)
                echo "使用方法："
                echo "  $0                          # 训练所有阶段"
                echo "  $0 --from-phase 2          # 从第2阶段开始"
                echo "  $0 --resume                # 自动恢复训练"
                echo "  $0 --help                  # 显示帮助"
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                echo "使用 --help 查看帮助"
                exit 1
                ;;
        esac
    done

    # 执行训练流程
    if [ ${START_FROM_PHASE} -le 1 ]; then
        train_phase1
    fi

    if [ ${START_FROM_PHASE} -le 2 ]; then
        train_phase2
    fi

    if [ ${START_FROM_PHASE} -le 3 ]; then
        train_phase3
    fi

    # 最终总结
    print_header "训练完成！"

    log_success "所有训练阶段已完成！"
    echo ""
    echo "检查点文件："
    echo "  - Phase 1: ${CHECKPOINT_DIR}/phase1/world_model_final.pth"
    echo "  - Phase 2 (Stage 1-5): ${CHECKPOINT_DIR}/curriculum/level*/*.zip"
    echo ""
    echo "日志文件："
    echo "  - ${LOG_DIR}/"
    echo ""
}

# 运行主函数
main "$@"
