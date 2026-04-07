@echo off
setlocal
REM Normal Llama-3.2 Instruct chat (NOT SpinalCord weights).
REM Use this when you want sensible replies while your scbrain_1b.gguf is still training / story-biased.
REM Requires both scaffold GGUFs under spinalcord\models\ (download from Hugging Face / TheBloke, etc.).
REM
REM PowerShell (from spinalcord folder):  .\dashboard\run_dashboard_llama_scaffold.bat
REM Do NOT use Unix paths like /dashboard/... â€” use .\dashboard\... instead.
cd /d "%~dp0"

echo â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
echo â•‘  Llama-3.2 scaffold â€” normal chat ^(not SpinalCord^)   â•‘
echo â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
echo.

set "PATH=%PATH%;C:\Users\SHIVANSH\Desktop\llama.cpp\build\bin\Release"
set LLAMA_SERVER=C:\Users\SHIVANSH\Desktop\llama.cpp\build\bin\Release\llama-server.exe

set "SCAFFOLD_BRAIN=%~dp0..\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf"
set "SCAFFOLD_DRAFT=%~dp0..\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf"

if not exist "%SCAFFOLD_BRAIN%" (
    echo [Error] Missing scaffold Brain GGUF:
    echo   "%SCAFFOLD_BRAIN%"
    echo Download a Llama-3.2-3B-Instruct GGUF into spinalcord\models\ or fix the path above.
    goto :end
)
if not exist "%SCAFFOLD_DRAFT%" (
    echo [Error] Missing scaffold Draft GGUF:
    echo   "%SCAFFOLD_DRAFT%"
    echo Download Llama-3.2-1B-Instruct Q4_K_M into spinalcord\models\ or fix the path above.
    goto :end
)

set "BRAIN=%SCAFFOLD_BRAIN%"
set "DRAFT=%SCAFFOLD_DRAFT%"

echo [Mode] Llama-3.2 scaffold ^(stable chat^)
echo   Brain:  "%BRAIN%"
echo   Draft:  "%DRAFT%"
echo.
if not defined PORT set "PORT=8080"
if not defined NGL set "NGL=99"
if not defined NGLD set "NGLD=99"
echo SpinalCord custom models are IGNORED. Close other llama-server on port %PORT% first.
echo Open http://127.0.0.1:%PORT% and refresh model list.
echo GPU layers: brain=%NGL%  draft=%NGLD%  ^(set NGLD=0 if VRAM is low^)
echo.

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

:end
endlocal
pause

