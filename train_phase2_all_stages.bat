@echo off
setlocal enabledelayedexpansion

echo ================================================================================
echo Phase 2 Training - Complete Curriculum (All 5 Levels)
echo ================================================================================
echo.
echo This script will train all 5 curriculum levels sequentially:
echo   Level 1: Basic Scenario (126k steps - ~35 min)
echo   Level 2: Medium Flow (126k steps - ~35 min)
echo   Level 3: High Flow (126k steps - ~35 min)
echo   Level 4: Extreme Scenario (126k steps - ~35 min)
echo   Level 5: Competition Scenario (1,494k steps - ~4 hours)
echo.
echo Total estimated time: ~5.5 hours
echo.
echo [AUTO-RESUME] Completed stages will be automatically skipped
echo ================================================================================
echo.

set CONFIG=configs/competition.yaml
set PHASE1_CHECKPOINT=checkpoints/competition/phase1/world_model_final.pth

REM Check if Phase 1 checkpoint exists
if not exist "%PHASE1_CHECKPOINT%" (
    echo [ERROR] Phase 1 checkpoint not found: %PHASE1_CHECKPOINT%
    echo [INFO] Please run Phase 1 training first:
    echo   python train_phase1.py
    echo.
    pause
    exit /b 1
)

echo [OK] Phase 1 checkpoint found
echo.

REM 计数器
set COMPLETED_STAGES=0

REM ==============================================================================
REM Stage 1: 基础场景
REM ==============================================================================
echo ================================================================================
echo [Stage 1/5] Basic Scenario
echo ================================================================================
echo Max vehicles: 10, Inflow: 800, Disturbance: 0%%
echo Steps: 126,000 - Estimated: ~35 minutes
echo ================================================================================
echo.

set STAGE1_CHECKPOINT=checkpoints/competition/curriculum/level1/custom_ppo.zip

REM 检查阶段1是否已完成
if exist "%STAGE1_CHECKPOINT%" (
    echo [SKIP] Stage 1 checkpoint already exists: %STAGE1_CHECKPOINT%
    echo [INFO] Skipping Stage 1 training...
    set /A COMPLETED_STAGES+=1
    echo.
    goto STAGE2
)

echo [TRAIN] Starting Stage 1 training...
echo.

conda activate sumo && python train_phase2.py --stage 1 --phase1-checkpoint %PHASE1_CHECKPOINT% --config %CONFIG%
if errorlevel 1 (
    echo.
    echo [ERROR] Stage 1 failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Stage 1 completed!
echo.

:STAGE2
REM ==============================================================================
REM Stage 2: 中等流量
REM ==============================================================================
echo ================================================================================
echo [Stage 2/5] Medium Flow
echo ================================================================================
echo Max vehicles: 15, Inflow: 1200, Disturbance: 20%%
echo Steps: 126,000 - Estimated: ~35 minutes
echo Loading checkpoint from: Stage 1
echo ================================================================================
echo.

set STAGE1=checkpoints/competition/curriculum/level1/custom_ppo.zip
set STAGE2_CHECKPOINT=checkpoints/competition/curriculum/level2/custom_ppo.zip

REM 检查依赖和本阶段是否已完成
if not exist "%STAGE1%" (
    echo [ERROR] Stage 1 checkpoint not found: %STAGE1%
    echo [INFO] Please complete Stage 1 first
    pause
    exit /b 1
)

if exist "%STAGE2_CHECKPOINT%" (
    echo [SKIP] Stage 2 checkpoint already exists: %STAGE2_CHECKPOINT%
    echo [INFO] Skipping Stage 2 training...
    set /A COMPLETED_STAGES+=1
    echo.
    goto STAGE3
)

echo [TRAIN] Starting Stage 2 training...
echo.

conda activate sumo && python train_phase2.py --stage 2 --prev-checkpoint %STAGE1% --config %CONFIG%
if errorlevel 1 (
    echo.
    echo [ERROR] Stage 2 failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Stage 2 completed!
echo.

:STAGE3
REM ==============================================================================
REM Stage 3: 高流量场景
REM ==============================================================================
echo ================================================================================
echo [Stage 3/5] High Flow Scenario
echo ================================================================================
echo Max vehicles: 20, Inflow: 1800, Disturbance: 40%%
echo Steps: 126,000 - Estimated: ~35 minutes
echo Loading checkpoint from: Stage 2
echo ================================================================================
echo.

set STAGE2=checkpoints/competition/curriculum/level2/custom_ppo.zip
set STAGE3_CHECKPOINT=checkpoints/competition/curriculum/level3/custom_ppo.zip

REM 检查依赖和本阶段是否已完成
if not exist "%STAGE2%" (
    echo [ERROR] Stage 2 checkpoint not found: %STAGE2%
    echo [INFO] Please complete Stage 2 first
    pause
    exit /b 1
)

if exist "%STAGE3_CHECKPOINT%" (
    echo [SKIP] Stage 3 checkpoint already exists: %STAGE3_CHECKPOINT%
    echo [INFO] Skipping Stage 3 training...
    set /A COMPLETED_STAGES+=1
    echo.
    goto STAGE4
)

echo [TRAIN] Starting Stage 3 training...
echo.

conda activate sumo && python train_phase2.py --stage 3 --prev-checkpoint %STAGE2% --config %CONFIG%
if errorlevel 1 (
    echo.
    echo [ERROR] Stage 3 failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Stage 3 completed!
echo.

:STAGE4
REM ==============================================================================
REM Stage 4: 极端场景
REM ==============================================================================
echo ================================================================================
echo [Stage 4/5] Extreme Scenario
echo ================================================================================
echo Max vehicles: 32, Inflow: 2400, Disturbance: 70%%
echo Steps: 126,000 - Estimated: ~35 minutes
echo Loading checkpoint from: Stage 3
echo ================================================================================
echo.

set STAGE3=checkpoints/competition/curriculum/level3/custom_ppo.zip
set STAGE4_CHECKPOINT=checkpoints/competition/curriculum/level4/custom_ppo.zip

REM 检查依赖和本阶段是否已完成
if not exist "%STAGE3%" (
    echo [ERROR] Stage 3 checkpoint not found: %STAGE3%
    echo [INFO] Please complete Stage 3 first
    pause
    exit /b 1
)

if exist "%STAGE4_CHECKPOINT%" (
    echo [SKIP] Stage 4 checkpoint already exists: %STAGE4_CHECKPOINT%
    echo [INFO] Skipping Stage 4 training...
    set /A COMPLETED_STAGES+=1
    echo.
    goto STAGE5
)

echo [TRAIN] Starting Stage 4 training...
echo.

conda activate sumo && python train_phase2.py --stage 4 --prev-checkpoint %STAGE3% --config %CONFIG%
if errorlevel 1 (
    echo.
    echo [ERROR] Stage 4 failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Stage 4 completed!
echo.

:STAGE5
REM ==============================================================================
REM Stage 5: 赛题场景
REM ==============================================================================
echo ================================================================================
echo [Stage 5/5] Competition Scenario
echo ================================================================================
echo Max vehicles: 32, Inflow: 2000, Disturbance: 50%%
echo Steps: 1,494,000 - Estimated: ~4 hours
echo Loading checkpoint from: Stage 4
echo ================================================================================
echo.

set STAGE4=checkpoints/competition/curriculum/level4/custom_ppo.zip
set STAGE5_CHECKPOINT=checkpoints/competition/curriculum/level5/custom_ppo.zip

REM 检查依赖和本阶段是否已完成
if not exist "%STAGE4%" (
    echo [ERROR] Stage 4 checkpoint not found: %STAGE4%
    echo [INFO] Please complete Stage 4 first
    pause
    exit /b 1
)

if exist "%STAGE5_CHECKPOINT%" (
    echo [SKIP] Stage 5 checkpoint already exists: %STAGE5_CHECKPOINT%
    echo [INFO] Skipping Stage 5 training...
    set /A COMPLETED_STAGES+=1
    echo.
    goto ALL_COMPLETED
)

echo [TRAIN] Starting Stage 5 training...
echo.

conda activate sumo && python train_phase2.py --stage 5 --prev-checkpoint %STAGE4% --config %CONFIG%
if errorlevel 1 (
    echo.
    echo [ERROR] Stage 5 failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Stage 5 completed!
echo.

:ALL_COMPLETED
REM ==============================================================================
REM All Stages Completed
REM ==============================================================================
echo ================================================================================
echo [SUCCESS] Phase 2 Training Completed!
echo ================================================================================
echo.

if %COMPLETED_STAGES% GTR 0 (
    echo [AUTO-RESUME] Skipped %COMPLETED_STAGES% already completed stage^(s^)
    echo [AUTO-RESUME] Only trained new stages
    echo.
)

echo All 5 curriculum levels have been trained successfully!
echo.
echo Final model checkpoints:
echo   - Level 1: checkpoints/competition/curriculum/level1/custom_ppo.zip
echo   - Level 2: checkpoints/competition/curriculum/level2/custom_ppo.zip
echo   - Level 3: checkpoints/competition/curriculum/level3/custom_ppo.zip
echo   - Level 4: checkpoints/competition/curriculum/level4/custom_ppo.zip
echo   - Level 5: checkpoints/competition/curriculum/level5/custom_ppo.zip
echo.
echo [RECOMMENDED] Use Level 5 model for competition evaluation:
echo   checkpoints/competition/curriculum/level5/custom_ppo.zip
echo.
echo ================================================================================
pause
