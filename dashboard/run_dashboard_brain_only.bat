@echo off
setlocal
REM Brain-only llama-server (no Draft / no speculative decoding).
REM Use this to compare chat quality vs run_dashboard.bat (Draft+Brain).
REM Resolves paths relative to dashboard\ â€” same as run_dashboard.bat.
cd /d "%~dp0"

echo â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
echo â•‘  SpinalCord - llama-server BRAIN ONLY ^(no draft^)    â•‘
echo â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
echo.

set "PATH=%PATH%;C:\Users\SHIVANSH\Desktop\llama.cpp\build\bin\Release"
set LLAMA_SERVER=C:\Users\SHIVANSH\Desktop\llama.cpp\build\bin\Release\llama-server.exe

set "CUSTOM_BRAIN=%~dp0..\models\scbrain_1b.gguf"
set "SCAFFOLD_BRAIN=%~dp0..\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf"

if exist "%CUSTOM_BRAIN%" (
    echo [Mode] Using YOUR SpinalCord Brain GGUF ^(no draft^).
    set "BRAIN=%CUSTOM_BRAIN%"
) else (
    echo [Mode] Custom brain GGUF not found; using scaffold brain.
    set "BRAIN=%SCAFFOLD_BRAIN%"
)

echo Resolved Brain path: "%BRAIN%"
echo.
echo No --model-draft - generation is plain autoregressive on the Brain model.
echo Compare with run_dashboard.bat for Draft+Brain speculative decoding.
if not defined PORT set "PORT=8080"
if not defined HOST set "HOST=127.0.0.1"
echo Using host: %HOST%  port: %PORT%
echo For LAN / phone testing A/B compare, set HOST=0.0.0.0 ^(see deploy\README.md^).
echo.

REM --webui is required for GET / to serve dashboard/index.html (override LLAMA_ARG_WEBUI=0 in env).
REM --path must be the folder containing index.html (same as run_dashboard.bat).
"%LLAMA_SERVER%" ^
  --model "%BRAIN%" ^
  --webui ^
  --jinja ^
  -c 4096 ^
  -ngl 99 ^
  --host %HOST% ^
  --port %PORT% ^
  --path "%~dp0."

endlocal
pause

