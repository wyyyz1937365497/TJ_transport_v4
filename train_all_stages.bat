@echo off
REM Multi-Stage Training Automation Script
REM 使用 train_phase2_stable.py 依次训练所有5个阶段

SETLOCAL EnableDelayedExpansion

echo ================================================================================
echo Multi-Stage Curriculum Training
echo ================================================================================
echo.
echo This script will train all 5 curriculum stages sequentially.
echo Each stage is independent and will load weights from the previous stage.
echo.
echo Press Ctrl+C to stop at any time.
echo.

REM 配置
SET PHASE1_CHECKPOINT=
SET CHECKPOINT=
SET TOTAL_STAGES=5

REM 可选：使用 Phase 1 权重初始化阶段1
SET PHASE1_CHECKPOINT=--phase1-checkpoint checkpoints/competition/phase1/world_model_final.pth

FOR %%S IN (1 2 3 4 5) DO (
    echo ================================================================================
    echo Stage %%S of %TOTAL_STAGES%
    echo ================================================================================
    echo.

    IF %%S==1 (
        echo [INFO] Starting Stage 1 (Basic Scenario)
        python train_phase2_stable.py --stage %%S %PHASE1_CHECKPOINT%
    ) ELSE (
        echo [INFO] Starting Stage %%S, loading from previous stage
        python train_phase2_stable.py --stage %%S --prev-checkpoint !CHECKPOINT!
    )

    IF ERRORLEVEL 1 (
        echo.
        echo [ERROR] Stage %%S failed with exit code !ERRORLEVEL!
        echo [INFO] Please check the error messages above.
        echo.
        pause
        EXIT /B !ERRORLEVEL!
    )

    REM 更新检查点路径
    IF %%S==1 SET CHECKPOINT=checkpoints/competition/curriculum/level1/ppo.zip
    IF %%S==2 SET CHECKPOINT=checkpoints/competition/curriculum/level2/ppo.zip
    IF %%S==3 SET CHECKPOINT=checkpoints/competition/curriculum/level3/ppo.zip
    IF %%S==4 SET CHECKPOINT=checkpoints/competition/curriculum/level4/ppo.zip
    IF %%S==5 SET CHECKPOINT=checkpoints/competition/phase2/shielded_ppo.zip

    echo.
    echo [OK] Stage %%S completed successfully!
    echo [INFO] Checkpoint: !CHECKPOINT!
    echo.
    echo [INFO] Waiting 3 seconds before next stage...
    timeout /t 3 /nobreak >nul
    echo.
)

echo ================================================================================
echo All Stages Completed!
echo ================================================================================
echo.
echo [INFO] All 5 curriculum stages have been trained successfully.
echo [INFO] Final model: checkpoints/competition/phase2/shielded_ppo.zip
echo.
echo You can now use the final model for evaluation or submission.
echo.

pause
