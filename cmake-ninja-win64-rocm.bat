@echo off
setlocal

rem --- Pfade anpassen falls nötig ---
set ROCM_ROOT=C:\Program Files\AMD\ROCm\6.4
set ROCM_BIN=%ROCM_ROOT%\bin

rem ROCm-Bin ins PATH
set PATH=%ROCM_BIN%;%PATH%

rem Sauberer Build-Ordner
rmdir /s /q build 2>nul

rem CMake-Konfiguration mit Ninja + Clang + HIP
cmake -B build ^
  -G "Ninja Multi-Config" ^
  -DCMAKE_C_COMPILER="%ROCM_BIN%\clang.exe" ^
  -DCMAKE_CXX_COMPILER="%ROCM_BIN%\clang++.exe" ^
  -DGGML_VULKAN=OFF ^
  -DGGML_HIP=ON ^
  -DAMDGPU_TARGETS=gfx1201 ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DTRELLIS2_BUILD_TESTS=ON ^
  -Dhipblas_DIR="%ROCM_ROOT%\lib\cmake\hipblas" ^
  .

if errorlevel 1 (
  echo CMake-Konfiguration fehlgeschlagen.
  pause
  exit /b 1
)

rem Build im Release-Mode
cmake --build build --config Release

echo Fertig.
pause
endlocal
