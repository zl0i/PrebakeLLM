@echo off
rem Builds llama.cpp from a source dir with MSVC + CUDA 13.5 (sm_89 only, faster build).
rem   tools\build_llamacpp.bat <source_dir> <build_dir>
rem Example (pristine baseline): tools\build_llamacpp.bat third_party\llama.cpp build\llama-ref
setlocal
set "SRC=%~1"
set "BLD=%~2"
if "%SRC%"=="" set "SRC=third_party\llama.cpp"
if "%BLD%"=="" set "BLD=build\llama-ref"
set "CUDA13=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5"
set "CUDA_PATH=%CUDA13%"
set "PATH=%CUDA13%\bin;%PATH%"
call "D:\MicrosoftVS\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d %~dp0..
cmake -S "%SRC%" -B "%BLD%" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=ON ^
  -DCMAKE_CUDA_COMPILER="%CUDA13%\bin\nvcc.exe" -DCUDAToolkit_ROOT="%CUDA13%" ^
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF || exit /b 1
cmake --build "%BLD%" --config Release -j 20 --target llama-server llama-completion llama-perplexity llama-bench || exit /b 1
