@echo off
setlocal

rem --- CPU-only shared-library build for the Go demo server (trellis2.dll) ---
rem   Same DLL as cmake-ninja-win64-rocm-shared.bat, but with the HIP backend
rem   off, so every stage runs on ggml-cpu. Used to tell GPU-kernel bugs apart
rem   from host-code bugs (e.g. the 1024^3 decode truncating at 2^21 voxels).
rem
rem   Builds into build/ so it does NOT overwrite the GPU DLLs in
rem   build/ — both can coexist, pick one with the server's -lib flag.
rem
rem   Only the compiler comes from ROCm (it is already installed and needs no
rem   VS developer prompt); no HIP/hipBLAS flags are needed without GGML_HIP.

set ROCM_BIN=C:\Program Files\AMD\ROCm\6.4\bin

rem CMAKE_RUNTIME_OUTPUT_DIRECTORY must be absolute (relative paths get
rem reinterpreted per-subdir by ggml's nested CMakeLists).
set DLL_OUT=%CD%\build\bin

cmake -B build ^
  -G "Ninja Multi-Config" ^
  -DCMAKE_C_COMPILER="%ROCM_BIN%\clang.exe" ^
  -DCMAKE_CXX_COMPILER="%ROCM_BIN%\clang++.exe" ^
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
echo Fertig: build\bin\Release\trellis2.dll  (+ ggml.dll, ggml-base.dll, ggml-cpu.dll)
echo.
echo Starten aus server\:
echo   trellis2-server.exe -lib ..\build\bin\Release\trellis2.dll
pause
endlocal
