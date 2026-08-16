@echo off
setlocal

rem --- Shared-library build with BOTH optional backends, for comparing them ---
rem   The primary build for this fork: HIP and Vulkan in one library, plus the
rem   CGAL Alpha Wrap print wrap and the AutoRemesher quad stage, so the viewer
rem   offers "watertight print wrap (CGAL)" and "quad retopology" at the same
rem   time and they can be judged against each other on one model.
rem
rem   The interesting third case is both together: Alpha Wrap first hands the
rem   remesher a closed 2-manifold, which is the input AutoRemesher expects.
rem   On raw dual-grid geometry it produces thousands of open edges; on wrapped
rem   geometry it should not. That combination is still NOT guaranteed
rem   watertight - the quad stage can reopen boundaries - so check the
rem   "boundary edges" figure the viewer reports before trusting it.
rem
rem   Requires (once):  vcpkg install eigen3 cgal
rem   Both are resolved through the vcpkg.json manifest; the cgal feature is
rem   opt-in so a default build never pulls a GPL-3.0-or-later package.

set ROCM_ROOT=C:\Program Files\AMD\ROCm\6.4
set ROCM_BIN=%ROCM_ROOT%\bin

rem Both GPU backends go into one library. ggml registers each with the same
rem backend registry, so a single server can be pointed at either at runtime
rem with TRELLIS2_DEVICE=rocm / vulkan / cpu - no second build tree, which is
rem what comparing them used to cost. Vulkan needs glslc from the SDK to
rem generate its shaders at build time.
if not defined VULKAN_SDK (
  echo VULKAN_SDK is not set - install the Vulkan SDK, or open a new shell.
  echo To build ROCm only, set -DGGML_VULKAN=OFF below.
  pause
  exit /b 1
)

set PATH=%ROCM_BIN%;%VULKAN_SDK%\Bin;%PATH%

rem CMAKE_RUNTIME_OUTPUT_DIRECTORY must be absolute (relative paths get
rem reinterpreted per-subdir by ggml's nested CMakeLists).
set DLL_OUT=%CD%\build\bin

rem Which library does the closest-surface search for the texture rebake.
rem OFF = CGAL (the reference), ON = tinybvh. Flip it to compare the two
rem projection backends; the configure output prints which one is active.
set FORCE_TINYBVH=ON

rmdir /s /q build 2>nul

cmake -B build ^
  -G "Ninja Multi-Config" ^
  -DCMAKE_C_COMPILER="%ROCM_BIN%\clang.exe" ^
  -DCMAKE_CXX_COMPILER="%ROCM_BIN%\clang++.exe" ^
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_MANIFEST_FEATURES=cgal ^
  -DGGML_VULKAN=ON ^
  -DGGML_HIP=ON ^
  -DAMDGPU_TARGETS=gfx1201 ^
  -DBUILD_SHARED_LIBS=ON ^
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="%DLL_OUT%" ^
  -DTRELLIS2_CGAL=ON ^
  -DTRELLIS2_AUTOREMESHER=ON ^
  -DTRELLIS2_FORCE_TINYBVH=%FORCE_TINYBVH% ^
  -DTRELLIS2_BUILD_EXAMPLES=ON ^
  -DTRELLIS2_BUILD_TESTS=ON ^
  -Dhipblas_DIR="%ROCM_ROOT%\lib\cmake\hipblas" ^
  .

if errorlevel 1 (
  echo CMake configure failed.
  pause
  exit /b 1
)

rem The configure output above must contain all three:
rem   -- CGAL Alpha Wrap print remeshing: enabled
rem   -- PBR projection backend: cgal            (or tinybvh)
rem   -- AutoRemesher quad remeshing: enabled (Eigen 5.x.x)
rem If CGAL says "unavailable", vcpkg did not provide it and the print wrap
rem will be missing from the viewer.

cmake --build build --config Release

if errorlevel 1 (
  echo Build failed.
  echo.
  echo If it breaks in print_remesh.cpp with
  echo   'boost/mpl/aux_/preprocessed/plain / or.hpp' file not found
  echo then that is the known Boost.MPL/clang problem - it happens with
  echo unmodified sources too and has nothing to do with this change. See
  echo docs/progress/autoremesher-quad-remesh_phase2-tinybvh-projection.md
  pause
  exit /b 1
)

echo.
echo Done: build\bin\Release\trellis2.dll
echo       (+ ggml.dll, ggml-base.dll, ggml-cpu.dll, ggml-hip.dll, ggml-vulkan.dll)
echo.
echo Pick the backend at run time, without rebuilding:
echo   set TRELLIS2_DEVICE=rocm     :: HIP (default, first GPU device)
echo   set TRELLIS2_DEVICE=vulkan   :: Vulkan
echo   set TRELLIS2_DEVICE=cpu      :: CPU
echo The variable has to be in the environment BEFORE the server starts - on
echo Windows, os.Setenv does not reach the DLL's own getenv().
echo If the name matches no GPU device it falls back to the CPU and says so;
echo it deliberately does not take the other GPU backend instead.
echo.
echo Comparison in the viewer: "watertight print wrap (CGAL)" and "quad
echo retopology" can be toggled separately and together. For every export the
echo server logs a timing block and reports quads / open edges / area kept.
echo.
echo Offline on an already generated mesh, without the server:
echo   build\bin\Release\quad_remesh_cli.exe generations\^<job^>\mesh.t2mesh out.obj --target-quads 20000 --min-area 0
pause
endlocal
