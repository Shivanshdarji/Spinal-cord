@echo off
setlocal
REM Production: bind all interfaces so other machines on the LAN (or a reverse proxy) can reach llama-server.
REM Set HOST=127.0.0.1 to stay local-only. Set PORT / NGL / NGLD as usual.
REM For LAN deploy with Meta Llama 3.2 + speculative ^(draft^) even if custom scbrain/scdraft exist:
REM   set USE_LLAMA_SCAFFOLD=1
cd /d "%~dp0"

set "PATH=%PATH%;C:\Users\SHIVANSH\Desktop\llama.cpp\build\bin\Release"
set LLAMA_SERVER=C:\Users\SHIVANSH\Desktop\llama.cpp\build\bin\Release\llama-server.exe

set "CUSTOM_BRAIN=%~dp0..\models\scbrain_1b.gguf"
set "CUSTOM_DRAFT=%~dp0..\models\scdraft_120m.gguf"
set "SCAFFOLD_BRAIN=%~dp0..\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf"
set "SCAFFOLD_DRAFT=%~dp0..\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf"

if /i "%USE_LLAMA_SCAFFOLD%"=="1" goto :llama_scaffold

if exist "%CUSTOM_BRAIN%" if exist "%CUSTOM_DRAFT%" (
    echo [Mode] Custom SpinalCord GGUF ^(scbrain + scdraft^)
    set BRAIN=%CUSTOM_BRAIN%
    set DRAFT=%CUSTOM_DRAFT%
    goto :have_models
)

echo [Mode] Llama 3.2 scaffold + speculative ^(custom GGUF not found^)
goto :llama_scaffold_pick

:llama_scaffold
echo [Mode] Llama 3.2 + speculative decoding ^(USE_LLAMA_SCAFFOLD=1^)
:llama_scaffold_pick
if not exist "%SCAFFOLD_BRAIN%" (
    echo [Error] Missing scaffold brain: "%SCAFFOLD_BRAIN%"
    goto :end
)
if not exist "%SCAFFOLD_DRAFT%" (
    echo [Error] Missing scaffold draft: "%SCAFFOLD_DRAFT%"
    goto :end
)
set BRAIN=%SCAFFOLD_BRAIN%
set DRAFT=%SCAFFOLD_DRAFT%

:have_models

if not defined PORT set "PORT=8080"
if not defined HOST set "HOST=0.0.0.0"
if not defined NGL set "NGL=99"
if not defined NGLD set "NGLD=99"

echo.
echo PROD: http://%HOST%:%PORT%/   ^(use your LAN IP or DNS from other PCs^)
echo Open Windows Firewall for TCP %PORT% if clients fail to connect.
echo GPU layers: brain=%NGL% draft=%NGLD%
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
  --host %HOST% ^
  --port %PORT% ^
  --path "%~dp0."

:end
endlocal
pause
