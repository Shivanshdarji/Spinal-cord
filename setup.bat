@echo off
REM ═══════════════════════════════════════════════════════════════════
REM  SpinalCord LLM — Environment Setup Script
REM  AppDice | Shivansh Darji
REM  Run this ONCE to set up your development environment.
REM ═══════════════════════════════════════════════════════════════════

echo.
echo ╔═══════════════════════════════════════════════╗
echo ║   SpinalCord LLM — Setup Script              ║
echo ║   AppDice ^| Shivansh Darji                   ║
echo ╚═══════════════════════════════════════════════╝
echo.

REM ─── STEP 1: Check Python ──────────────────────────────────────────
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! 
    echo    Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo ✅ Python found

REM ─── STEP 2: Install PyTorch with CUDA 11.8 ───────────────────────
echo.
echo [2/5] Installing PyTorch with CUDA 11.8 support (RTX 2050)...
echo    This may take a few minutes...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
if errorlevel 1 (
    echo ❌ PyTorch installation failed
    pause
    exit /b 1
)
echo ✅ PyTorch installed

REM ─── STEP 3: Install other requirements ───────────────────────────
echo.
echo [3/5] Installing remaining dependencies...
pip install transformers datasets accelerate safetensors tqdm numpy matplotlib rich --quiet
if errorlevel 1 (
    echo ⚠️  Some packages may have failed. Check output above.
) else (
    echo ✅ Dependencies installed
)

REM ─── STEP 4: Clone llama.cpp ──────────────────────────────────────
echo.
echo [4/5] Cloning llama.cpp...
if exist "..\llama.cpp" (
    echo ✅ llama.cpp already cloned at ..\llama.cpp
) else (
    git clone https://github.com/ggml-org/llama.cpp ..\llama.cpp
    if errorlevel 1 (
        echo ❌ Git clone failed. Make sure Git is installed.
        echo    Download: https://git-scm.com/download/win
    ) else (
        echo ✅ llama.cpp cloned to ..\llama.cpp
    )
)

REM ─── STEP 5: Verify GPU ───────────────────────────────────────────
echo.
echo [5/5] Checking GPU/CUDA availability...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo ═══════════════════════════════════════════════════════════════
echo   ✨ Setup complete!
echo.
echo   NEXT STEPS:
echo   1. Open dashboard\index.html in your browser to see the UI
echo   2. cd train ^&^& python train.py    (to train the model)
echo   3. Build llama.cpp with CUDA (see README.md for instructions)
echo ═══════════════════════════════════════════════════════════════
echo.
pause
