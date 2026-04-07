@echo off
setlocal
REM Llama scaffold + speculative, bind 0.0.0.0 for LAN / deploy testing.
cd /d "%~dp0"

set "PATH=%PATH%;C:\Users\SHIVANSH\Desktop\llama.cpp\build\bin\Release"
set LLAMA_SERVER=C:\Users\SHIVANSH\Desktop\llama.cpp\build\bin\Release\llama-server.exe
set "SCAFFOLD_BRAIN=%~dp0..\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf"
set "SCAFFOLD_DRAFT=%~dp0..\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf"

if not exist "%SCAFFOLD_BRAIN%" (
    echo [Error] Missing "%SCAFFOLD_BRAIN%"
    goto :end
)
if not exist "%SCAFFOLD_DRAFT%" (
    echo [Error] Missing "%SCAFFOLD_DRAFT%"
    goto :end
)

if not defined PORT set "PORT=8080"
if not defined HOST set "HOST=0.0.0.0"
if not defined NGL set "NGL=99"
if not defined NGLD set "NGLD=99"

echo PROD scaffold: http://%HOST%:%PORT%/
echo.

"%LLAMA_SERVER%" ^
  --model "%SCAFFOLD_BRAIN%" ^
  --model-draft "%SCAFFOLD_DRAFT%" ^
  --webui ^
  --jinja ^
  --draft-max 8 ^
  --draft-min 2 ^
  -c 4096 ^
  -ngl %NGL% ^
  -ngld %NGLD% ^
  --host %HOST% ^
  --port %PORT% ^
  --path "%~dp0."

:end
endlocal
pause
