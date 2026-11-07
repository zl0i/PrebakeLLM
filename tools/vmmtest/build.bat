@echo off
set "CUDA13=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5"
call "D:\MicrosoftVS\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d %~dp0
"%CUDA13%\bin\nvcc.exe" -O2 -arch=sm_89 -o vmmtest.exe vmmtest.cu -lcuda
