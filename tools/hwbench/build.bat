@echo off
rem Builds hwbench.exe with MSVC + nvcc (CUDA 13.5). sm_89 = Ada (RTX 40xx).
set "CUDA13=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5"
call "D:\MicrosoftVS\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d %~dp0
"%CUDA13%\bin\nvcc.exe" -O2 -arch=sm_89 -Xcompiler "/O2 /arch:AVX2 /EHsc" -o hwbench.exe hwbench.cu
