#!/bin/bash

echo "================================================================================"
echo "Phase 2 Training - Complete Curriculum (All 5 Levels)"
echo "================================================================================"
echo
echo "This script will train all 5 curriculum levels sequentially:"
echo "  Level 1: Basic Scenario (126k steps - ~35 min)"
echo "  Level 2: Medium Flow (126k steps - ~35 min)"
echo "  Level 3: High Flow (126k steps - ~35 min)"
echo "  Level 4: Extreme Scenario (126k steps - ~35 min)"
echo "  Level 5: Competition Scenario (1,494k steps - ~4 hours)"
echo
echo "Total estimated time: ~5.5 hours"
echo
echo "[AUTO-RESUME] Completed stages will be automatically skipped"
echo "================================================================================"
echo

CONFIG="configs/competition.yaml"
PHASE1_CHECKPOINT="checkpoints/competition/phase1/world_model_final.pth"

# Check if Phase 1 checkpoint exists
if [ ! -f "$PHASE1_CHECKPOINT" ]; then
    echo "[ERROR] Phase 1 checkpoint not found: $PHASE1_CHECKPOINT"
    echo "[INFO] Please run Phase 1 training first:"
    echo "  python train_phase1.py"
    echo
    exit 1
fi

echo "[OK] Phase 1 checkpoint found"
echo

# 计数器
COMPLETED_STAGES=0

# ====================================================================================
# Stage 1: 基础场景
# ====================================================================================
echo "================================================================================"
echo "[Stage 1/5] Basic Scenario"
echo "================================================================================"
echo "Max vehicles: 10, Inflow: 800, Disturbance: 0%"
echo "Steps: 126,000 - Estimated: ~35 minutes"
echo "================================================================================"
echo

STAGE1_CHECKPOINT="checkpoints/competition/curriculum/level1/custom_ppo.zip"

# 检查阶段1是否已完成
if [ -f "$STAGE1_CHECKPOINT" ]; then
    echo "[SKIP] Stage 1 checkpoint already exists: $STAGE1_CHECKPOINT"
    echo "[INFO] Skipping Stage 1 training..."
    ((COMPLETED_STAGES++))
    echo
    # 跳转到STAGE2
else
    echo "[TRAIN] Starting Stage 1 training..."
    echo

    python train_phase2.py --stage 1 --phase1-checkpoint "$PHASE1_CHECKPOINT" --config "$CONFIG"
    if [ $? -ne 0 ]; then
        echo
        echo "[ERROR] Stage 1 failed!"
        exit 1
    fi

    echo
    echo "[SUCCESS] Stage 1 completed!"
    echo
fi

# ====================================================================================
# Stage 2: 中等流量
# ====================================================================================
echo "================================================================================"
echo "[Stage 2/5] Medium Flow"
echo "================================================================================"
echo "Max vehicles: 15, Inflow: 1200, Disturbance: 20%"
echo "Steps: 126,000 - Estimated: ~35 minutes"
echo "Loading checkpoint from: Stage 1"
echo "================================================================================"
echo

STAGE1="checkpoints/competition/curriculum/level1/custom_ppo.zip"
STAGE2_CHECKPOINT="checkpoints/competition/curriculum/level2/custom_ppo.zip"

# 检查依赖和本阶段是否已完成
if [ ! -f "$STAGE1" ]; then
    echo "[ERROR] Stage 1 checkpoint not found: $STAGE1"
    echo "[INFO] Please complete Stage 1 first"
    exit 1
fi

if [ -f "$STAGE2_CHECKPOINT" ]; then
    echo "[SKIP] Stage 2 checkpoint already exists: $STAGE2_CHECKPOINT"
    echo "[INFO] Skipping Stage 2 training..."
    ((COMPLETED_STAGES++))
    echo
    # 跳转到STAGE3
else
    echo "[TRAIN] Starting Stage 2 training..."
    echo

    python train_phase2.py --stage 2 --prev-checkpoint "$STAGE1" --config "$CONFIG"
    if [ $? -ne 0 ]; then
        echo
        echo "[ERROR] Stage 2 failed!"
        exit 1
    fi

    echo
    echo "[SUCCESS] Stage 2 completed!"
    echo
fi

# ====================================================================================
# Stage 3: 高流量场景
# ====================================================================================
echo "================================================================================"
echo "[Stage 3/5] High Flow Scenario"
echo "================================================================================"
echo "Max vehicles: 20, Inflow: 1800, Disturbance: 40%"
echo "Steps: 126,000 - Estimated: ~35 minutes"
echo "Loading checkpoint from: Stage 2"
echo "================================================================================"
echo

STAGE2="checkpoints/competition/curriculum/level2/custom_ppo.zip"
STAGE3_CHECKPOINT="checkpoints/competition/curriculum/level3/custom_ppo.zip"

# 检查依赖和本阶段是否已完成
if [ ! -f "$STAGE2" ]; then
    echo "[ERROR] Stage 2 checkpoint not found: $STAGE2"
    echo "[INFO] Please complete Stage 2 first"
    exit 1
fi

if [ -f "$STAGE3_CHECKPOINT" ]; then
    echo "[SKIP] Stage 3 checkpoint already exists: $STAGE3_CHECKPOINT"
    echo "[INFO] Skipping Stage 3 training..."
    ((COMPLETED_STAGES++))
    echo
    # 跳转到STAGE4
else
    echo "[TRAIN] Starting Stage 3 training..."
    echo

    python train_phase2.py --stage 3 --prev-checkpoint "$STAGE2" --config "$CONFIG"
    if [ $? -ne 0 ]; then
        echo
        echo "[ERROR] Stage 3 failed!"
        exit 1
    fi

    echo
    echo "[SUCCESS] Stage 3 completed!"
    echo
fi

# ====================================================================================
# Stage 4: 极端场景
# ====================================================================================
echo "================================================================================"
echo "[Stage 4/5] Extreme Scenario"
echo "================================================================================"
echo "Max vehicles: 32, Inflow: 2400, Disturbance: 70%"
echo "Steps: 126,000 - Estimated: ~35 minutes"
echo "Loading checkpoint from: Stage 3"
echo "================================================================================"
echo

STAGE3="checkpoints/competition/curriculum/level3/custom_ppo.zip"
STAGE4_CHECKPOINT="checkpoints/competition/curriculum/level4/custom_ppo.zip"

# 检查依赖和本阶段是否已完成
if [ ! -f "$STAGE3" ]; then
    echo "[ERROR] Stage 3 checkpoint not found: $STAGE3"
    echo "[INFO] Please complete Stage 3 first"
    exit 1
fi

if [ -f "$STAGE4_CHECKPOINT" ]; then
    echo "[SKIP] Stage 4 checkpoint already exists: $STAGE4_CHECKPOINT"
    echo "[INFO] Skipping Stage 4 training..."
    ((COMPLETED_STAGES++))
    echo
    # 跳转到STAGE5
else
    echo "[TRAIN] Starting Stage 4 training..."
    echo

    python train_phase2.py --stage 4 --prev-checkpoint "$STAGE3" --config "$CONFIG"
    if [ $? -ne 0 ]; then
        echo
        echo "[ERROR] Stage 4 failed!"
        exit 1
    fi

    echo
    echo "[SUCCESS] Stage 4 completed!"
    echo
fi

# ====================================================================================
# Stage 5: 赛题场景
# ====================================================================================
echo "================================================================================"
echo "[Stage 5/5] Competition Scenario"
echo "================================================================================"
echo "Max vehicles: 32, Inflow: 2000, Disturbance: 50%"
echo "Steps: 1,494,000 - Estimated: ~4 hours"
echo "Loading checkpoint from: Stage 4"
echo "================================================================================"
echo

STAGE4="checkpoints/competition/curriculum/level4/custom_ppo.zip"
STAGE5_CHECKPOINT="checkpoints/competition/curriculum/level5/custom_ppo.zip"

# 检查依赖和本阶段是否已完成
if [ ! -f "$STAGE4" ]; then
    echo "[ERROR] Stage 4 checkpoint not found: $STAGE4"
    echo "[INFO] Please complete Stage 4 first"
    exit 1
fi

if [ -f "$STAGE5_CHECKPOINT" ]; then
    echo "[SKIP] Stage 5 checkpoint already exists: $STAGE5_CHECKPOINT"
    echo "[INFO] Skipping Stage 5 training..."
    ((COMPLETED_STAGES++))
    echo
    # 跳转到ALL_COMPLETED
else
    echo "[TRAIN] Starting Stage 5 training..."
    echo

    python train_phase2.py --stage 5 --prev-checkpoint "$STAGE4" --config "$CONFIG"
    if [ $? -ne 0 ]; then
        echo
        echo "[ERROR] Stage 5 failed!"
        exit 1
    fi

    echo
    echo "[SUCCESS] Stage 5 completed!"
    echo
fi

# ====================================================================================
# All Stages Completed
# ====================================================================================
echo "================================================================================"
echo "[SUCCESS] Phase 2 Training Completed!"
echo "================================================================================"
echo

if [ $COMPLETED_STAGES -gt 0 ]; then
    echo "[AUTO-RESUME] Skipped $COMPLETED_STAGES already completed stage(s)"
    echo "[AUTO-RESUME] Only trained new stages"
    echo
fi

echo "All 5 curriculum levels have been trained successfully!"
echo
echo "Final model checkpoints:"
echo "  - Level 1: checkpoints/competition/curriculum/level1/custom_ppo.zip"
echo "  - Level 2: checkpoints/competition/curriculum/level2/custom_ppo.zip"
echo "  - Level 3: checkpoints/competition/curriculum/level3/custom_ppo.zip"
echo "  - Level 4: checkpoints/competition/curriculum/level4/custom_ppo.zip"
echo "  - Level 5: checkpoints/competition/curriculum/level5/custom_ppo.zip"
echo
echo "[RECOMMENDED] Use Level 5 model for competition evaluation:"
echo "  checkpoints/competition/curriculum/level5/custom_ppo.zip"
echo
echo "================================================================================"