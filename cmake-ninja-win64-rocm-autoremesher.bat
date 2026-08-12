@echo off
setlocal

rem --- Shared-library build for the Go demo server (libtrellis2.dll) ---
rem   cmake-ninja-win64-rocm.bat, with the AutoRemesher quad remeshing stage.
rem
rem   The quad stage needs Eigen 5.x, which is MPL-2.0 and therefore consumed
rem   through the vcpkg manifest instead of being vendored (see THIRD_PARTY.md).
rem   The toolchain file is what makes that work: without it there is no Eigen,
rem   and TRELLIS2_AUTOREMESHER silently switches itself off.
rem
rem   Deliberately CGAL-free: no -DVCPKG_MANIFEST_FEATURES=cgal, so vcpkg never
rem   resolves the GPL-3.0-or-later package, and -DTRELLIS2_CGAL=OFF so the
rem   probe cannot pick up a stray installation either. This is the MIT-only
rem   path. To get both backends in one build, use the cgal script and add
rem   -DTRELLIS2_AUTOREMESHER=ON there instead.
rem
rem   Examples are ON here, unlike the other ROCm scripts: quad_remesh_cli is
rem   currently the only consumer of the quad stage, so a build without it
rem   would produce nothing to test the feature with.

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
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DGGML_VULKAN=OFF ^
  -DGGML_HIP=ON ^
  -DAMDGPU_TARGETS=gfx1201 ^
  -DBUILD_SHARED_LIBS=ON ^
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="%DLL_OUT%" ^
  -DTRELLIS2_AUTOREMESHER=ON ^
  -DTRELLIS2_CGAL=OFF ^
  -DTRELLIS2_BUILD_EXAMPLES=ON ^
  -DTRELLIS2_BUILD_TESTS=ON ^
  -Dhipblas_DIR="%ROCM_ROOT%\lib\cmake\hipblas" ^
  .

if errorlevel 1 (
  echo CMake-Konfiguration fehlgeschlagen.
  pause
  exit /b 1
)

rem The configure output above must contain:
rem   -- AutoRemesher quad remeshing: enabled (Eigen 5.0.1)
rem If it says "unavailable" instead, vcpkg did not provide Eigen 5.x and the
rem DLL is being built without the quad stage.

cmake --build build --config Release

if errorlevel 1 (
  echo Build fehlgeschlagen.
  pause
  exit /b 1
)

echo.
echo Fertig: build\bin\Release\libtrellis2.dll  (+ ggml.dll, ggml-base.dll, ggml-cpu.dll, ggml-hip.dll)
echo Smoke test: build\bin\Release\quad_remesh_cli.exe in.obj out.obj --target-quads 20000
pause
endlocal
