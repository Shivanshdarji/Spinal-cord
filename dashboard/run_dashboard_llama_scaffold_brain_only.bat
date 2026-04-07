@echo off
setlocal
REM Llama-3.2 Instruct brain-only mode (no draft/speculative).
REM Use this to compare speed/quality against run_dashboard_llama_scaffold.bat fairly.
cd /d "%~dp0"

echo â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
echo â•‘  Llama-3.2 scaffold - BRAIN ONLY ^(no draft^)         â•‘
echo â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
echo.

set "PATH=%PATH%;C:\Users\SHIVANSH\Desktop\llama.cpp\build\bin\Release"
set LLAMA_SERVER=C:\Users\SHIVANSH\Desktop\llama.cpp\build\bin\Release\llama-server.exe

set "SCAFFOLD_BRAIN=%~dp0..\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf"

if not exist "%SCAFFOLD_BRAIN%" (
    echo [Error] Missing scaffold Brain GGUF:
    echo   "%SCAFFOLD_BRAIN%"
    echo Download a Llama-3.2-3B-Instruct GGUF into spinalcord\models\ or fix the path above.
    goto :end
)

set "BRAIN=%SCAFFOLD_BRAIN%"

echo [Mode] Llama-3.2 scaffold brain-only
echo   Brain: "%BRAIN%"
echo.
echo Close other llama-server on the speculative port ^(e.g. 8080^) first.
if not defined PORT set "PORT=8080"
if not defined HOST set "HOST=127.0.0.1"
echo Open http://%HOST%:%PORT%/  ^(use LAN IP from other devices if HOST=0.0.0.0^)
echo.

"%LLAMA_SERVER%" ^
  --model "%BRAIN%" ^
  --webui ^
  --jinja ^
  -c 4096 ^
  -ngl 99 ^
  --host %HOST% ^
  --port %PORT% ^
  --path "%~dp0."

:end
endlocal
pause

