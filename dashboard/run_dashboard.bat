@echo off
setlocal
REM Always resolve ..\models relative to THIS folder (dashboard\), not your shell cwd.
REM Otherwise running from spinalcord\ makes ..\models = Desktop\models (wrong).
cd /d "%~dp0"

echo â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
echo â•‘       ðŸ§  SpinalCord LLM â€” AppDice Dashboard        â•‘
echo â•‘            Shivansh Darji - SpinalCord v2           â•‘
echo â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
echo.

set "PATH=%PATH%;C:\Users\SHIVANSH\Desktop\llama.cpp\build\bin\Release"
set LLAMA_SERVER=C:\Users\SHIVANSH\Desktop\llama.cpp\build\bin\Release\llama-server.exe

set "CUSTOM_BRAIN=%~dp0..\models\scbrain_1b.gguf"
set "CUSTOM_DRAFT=%~dp0..\models\scdraft_120m.gguf"

set "SCAFFOLD_BRAIN=%~dp0..\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf"
set "SCAFFOLD_DRAFT=%~dp0..\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf"

REM â”€â”€â”€ Auto-detect which models to use â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if exist "%CUSTOM_BRAIN%" if exist "%CUSTOM_DRAFT%" (
    echo [Mode] Using YOUR CUSTOM SpinalCord models! ðŸš€
    echo   Brain:  "%CUSTOM_BRAIN%"
    echo   Draft:  "%CUSTOM_DRAFT%"
    set BRAIN=%CUSTOM_BRAIN%
    set DRAFT=%CUSTOM_DRAFT%
set DRAFT_MAX=8
) else (
    echo [Mode] Using Llama-3.2 scaffold ^(custom models not trained yet^)
    echo   Brain:  "%SCAFFOLD_BRAIN%"
    echo   Draft:  "%SCAFFOLD_DRAFT%"
    echo.
    echo   To switch to YOUR OWN custom LLM:
    echo     1. cd train ^& python train_brain.py
    echo     2. python distill_draft.py  
    echo     3. cd convert ^& python convert_both.py
    echo     4. Run this script again!
    set BRAIN=%SCAFFOLD_BRAIN%
    set DRAFT=%SCAFFOLD_DRAFT%
    set DRAFT_MAX=8
)

echo.
echo Resolved Brain path: "%BRAIN%"
echo Resolved Draft path: "%DRAFT%"
echo.
if not defined PORT set "PORT=8080"
if not defined NGL set "NGL=99"
if not defined NGLD set "NGLD=99"
echo Launching server on http://127.0.0.1:%PORT%...
echo GPU layers: brain=%NGL%  draft=%NGLD%  ^(set NGLD=0 if VRAM is low^)
echo.

REM --jinja is REQUIRED for SpinalCord GGUF: chat template is Jinja (hf_export/chat_template.jinja).
REM Without it, POST /v1/chat/completions returns 400 and the dashboard chat stays empty.
REM --webui serves static files from --path; without it, GET / returns 404 if LLAMA_ARG_WEBUI disables UI.
"%LLAMA_SERVER%" ^
  --model "%BRAIN%" ^
  --model-draft "%DRAFT%" ^
  --webui ^
  --jinja ^
  --draft-max 8 ^
  --draft-min 2 ^
  -c 4096 ^
  -ngl %NGL% ^
  -ngld %NGLD% ^
  --host 127.0.0.1 ^
  --port %PORT% ^
  --path "%~dp0."

endlocal
pause

