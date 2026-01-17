@echo off
setlocal enabledelayedexpansion

echo ================================================================================
echo Complete Training Pipeline - Phase 1 + Phase 2 (All 5 Curriculum Levels)
echo ================================================================================
echo.

set CONFIG=configs/competition.yaml
set PHASE1_CHECKPOINT=checkpoints/competition/phase1/world_model_final.pth

REM Phase 1
echo [Phase 1] World Model Training...
conda activate sumo && python train_phase1.py
if errorlevel 1 (
    echo [ERROR] Phase 1 failed!
    pause
    exit /b 1
)

REM Stage 1
echo [Stage 1] Basic Scenario...
conda activate sumo && python train_phase2.py --stage 1 --phase1-checkpoint %PHASE1_CHECKPOINT% --config %CONFIG%
if errorlevel 1 (
    echo [ERROR] Stage 1 failed!
    pause
    exit /b 1
)

REM Stage 2
set STAGE1=checkpoints/competition/curriculum/level1/custom_ppo.zip
conda activate sumo && python train_phase2.py --stage 2 --prev-checkpoint %STAGE1% --config %CONFIG%
if errorlevel 1 (
    echo [ERROR] Stage 2 failed!
    pause
    exit /b 1
)

REM Stage 3
set STAGE2=checkpoints/competition/curriculum/level2/custom_ppo.zip
conda activate sumo && python train_phase2.py --stage 3 --prev-checkpoint %STAGE2% --config %CONFIG%
if errorlevel 1 (
    echo [ERROR] Stage 3 failed!
    pause
    exit /b 1
)

REM Stage 4
set STAGE3=checkpoints/competition/curriculum/level3/custom_ppo.zip
conda activate sumo && python train_phase2.py --stage 4 --prev-checkpoint %STAGE3% --config %CONFIG%
if errorlevel 1 (
    echo [ERROR] Stage 4 failed!
    pause
    exit /b 1
)

REM Stage 5
set STAGE4=checkpoints/competition/curriculum/level4/custom_ppo.zip
conda activate sumo && python train_phase2.py --stage 5 --prev-checkpoint %STAGE4% --config %CONFIG%
if errorlevel 1 (
    echo [ERROR] Stage 5 failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] All stages completed!
echo Final model: checkpoints/competition/curriculum/level5/custom_ppo.zip
echo.
pause
