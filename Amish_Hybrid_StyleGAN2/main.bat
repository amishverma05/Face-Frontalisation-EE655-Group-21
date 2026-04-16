@echo off
:: Activate Python UTF-8 Mode (prevents UnicodeDecodeError)
set PYTHONUTF8=1

:: Activate the virtual environment
call venv\Scripts\activate.bat

echo ====================================================
echo  Face Frontalization — Full Pipeline
echo  Sanity Check → StyleGAN Setup → Train
echo ====================================================
echo.

:: ── Step 1: Sanity checks ────────────────────────────────────────────────────
echo [1/3] Running sanity checks...
python sanity_check.py --skip_legacy
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Sanity check failed! Fix the issues above and retry.
    pause
    exit /b %ERRORLEVEL%
)

:: ── Step 2: StyleGAN2 setup (always recomputes w_avg with fixed mapping net) ─
echo.
echo [2/3] Running StyleGAN2 setup (recomputes frontal w_avg)...
echo       Note: checkpoint download is skipped if already present.
if exist checkpoints\ffhq-256-config-e.pt (
    python setup_stylegan.py --skip_download
) else (
    python setup_stylegan.py
)
if %ERRORLEVEL% neq 0 (
    echo [ERROR] StyleGAN setup failed! Exiting.
    pause
    exit /b %ERRORLEVEL%
)

:: ── Step 3: Training ─────────────────────────────────────────────────────────
echo.
echo [3/3] Starting training (fresh run — do NOT use --resume on first launch)...
python train.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Training exited with an error.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ====================================================
echo  Pipeline finished successfully!
echo ====================================================
pause
