@echo off
rem Builds ik_llama.cpp (third_party\ik_llama.cpp -> build\ik) with MSVC + CUDA 13.5, sm_89 only.
setlocal
set "CUDA13=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5"
set "CUDA_PATH=%CUDA13%"
set "PATH=%CUDA13%\bin;%PATH%"
call "D:\MicrosoftVS\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d %~dp0..
cmake -S third_party\ik_llama.cpp -B build\ik -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=ON ^
  -DCMAKE_CUDA_COMPILER="%CUDA13%\bin\nvcc.exe" -DCUDAToolkit_ROOT="%CUDA13%" ^
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF || exit /b 1
cmake --build build\ik --config Release -j 20 --target llama-server llama-perplexity llama-bench || exit /b 1
