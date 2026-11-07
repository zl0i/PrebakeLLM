@echo off
rem Builds trace_experts.exe against the source build of llama.cpp (build\llama-ref).
call "D:\MicrosoftVS\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d %~dp0
set "LC=..\..\third_party\llama.cpp"
set "LB=..\..\build\llama-ref"
cl /nologo /O2 /EHsc /std:c++17 /I "%LC%\include" /I "%LC%\ggml\include" trace_experts.cpp ^
  /link "%LB%\src\llama.lib" "%LB%\ggml\src\ggml.lib" "%LB%\ggml\src\ggml-base.lib" /OUT:trace_experts.exe
