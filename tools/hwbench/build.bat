@echo off
rem Builds hwbench.exe with MSVC + nvcc. sm_89 = Ada (RTX 40xx).
call "D:\MicrosoftVS\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
nvcc -O2 -arch=sm_89 -Xcompiler "/O2 /arch:AVX2 /EHsc" -o hwbench.exe hwbench.cu
