@echo off
setlocal

rem --- Vulkan shared-library build for the Go demo server (trellis2.dll) ---
rem   Same DLL as cmake-ninja-win64-rocm.bat, but with the Vulkan backend
rem   instead of HIP. Third data point next to CPU and ROCm when a stage
rem   comes out wrong on one backend only (e.g. the 1024^3 shape decode
rem   truncating at 2^21 voxels on HIP, which the CPU path does not do).
rem
rem   Needs the Vulkan SDK: ggml-vulkan compiles its compute shaders at build
rem   time with glslc. CMake locates it via the VULKAN_SDK env var.
rem   The compiler still comes from ROCm — it is a plain clang for host code
rem   and saves needing a VS developer prompt.

if not defined VULKAN_SDK (
  echo VULKAN_SDK ist nicht gesetzt - Vulkan SDK installieren oder Shell neu oeffnen.
  pause
  exit /b 1
)

set ROCM_BIN=C:\Program Files\AMD\ROCm\6.4\bin

rem glslc greifbar machen, falls der SDK-Installer den PATH nicht gesetzt hat
set PATH=%VULKAN_SDK%\Bin;%PATH%

rem CMAKE_RUNTIME_OUTPUT_DIRECTORY must be absolute (relative paths get
rem reinterpreted per-subdir by ggml's nested CMakeLists).
set DLL_OUT=%CD%\build\bin

rmdir /s /q build 2>nul

cmake -B build ^
  -G "Ninja Multi-Config" ^
  -DCMAKE_C_COMPILER="%ROCM_BIN%\clang.exe" ^
  -DCMAKE_CXX_COMPILER="%ROCM_BIN%\clang++.exe" ^
  -DGGML_VULKAN=ON ^
  -DGGML_HIP=OFF ^
  -DBUILD_SHARED_LIBS=ON ^
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="%DLL_OUT%" ^
  -DTRELLIS2_BUILD_EXAMPLES=OFF ^
  -DTRELLIS2_BUILD_TESTS=OFF ^
  .

if errorlevel 1 (
  echo CMake-Konfiguration fehlgeschlagen.
  pause
  exit /b 1
)

cmake --build build --config Release

if errorlevel 1 (
  echo Build fehlgeschlagen.
  pause
  exit /b 1
)

echo.
echo Fertig: build\bin\Release\trellis2.dll  (+ ggml.dll, ggml-base.dll, ggml-cpu.dll, ggml-vulkan.dll)
echo.
echo Starten aus server\:
echo   trellis2-server.exe -lib ..\build\bin\Release\trellis2.dll
pause
endlocal
