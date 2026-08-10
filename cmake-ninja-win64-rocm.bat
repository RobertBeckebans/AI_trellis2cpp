@echo off
setlocal

rem --- Shared-library build for the Go demo server (libtrellis2.dll) ---
rem   Static build for the CLI examples (trellis2.lib) is the regular
rem   cmake-ninja-win64-rocm.bat. This one builds the DLL into build/
rem   so the Go server can LoadLibrary it.

set ROCM_ROOT=C:\Program Files\AMD\ROCm\6.4
set ROCM_BIN=%ROCM_ROOT%\bin

set PATH=%ROCM_BIN%;%PATH%

rem CMAKE_RUNTIME_OUTPUT_DIRECTORY must be absolute (relative paths get
rem reinterpreted per-subdir by ggml's nested CMakeLists).
set DLL_OUT=%CD%\build\bin

rmdir /s /q build 2>nul

cmake -B build ^
  -G "Ninja Multi-Config" ^
  -DCMAKE_C_COMPILER="%ROCM_BIN%\clang.exe" ^
  -DCMAKE_CXX_COMPILER="%ROCM_BIN%\clang++.exe" ^
  -DGGML_VULKAN=OFF ^
  -DGGML_HIP=ON ^
  -DAMDGPU_TARGETS=gfx1201 ^
  -DBUILD_SHARED_LIBS=ON ^
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="%DLL_OUT%" ^
  -DTRELLIS2_BUILD_EXAMPLES=OFF ^
  -DTRELLIS2_BUILD_TESTS=OFF ^
  -Dhipblas_DIR="%ROCM_ROOT%\lib\cmake\hipblas" ^
  .

if errorlevel 1 (
  echo CMake-Konfiguration fehlgeschlagen.
  pause
  exit /b 1
)

cmake --build build --config Release

echo.
echo Fertig: build\bin\Release\libtrellis2.dll  (+ ggml.dll, ggml-base.dll, ggml-cpu.dll, ggml-hip.dll)
pause
endlocal
