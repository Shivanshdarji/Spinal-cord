@echo off
setlocal
set PATH=%PATH%;C:\Users\SHIVANSH\Desktop\llama.cpp\build\bin\Release
cd /d %~dp0
.\build\Release\spinalcord.exe %*
endlocal
