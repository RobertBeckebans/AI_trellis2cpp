@echo off
setlocal

rem --- CUDA + Vulkan + CPU via the Visual Studio generator, into build-cuda\ ---
rem   The MSBuild counterpart to cmake-ninja-win64-cuda.bat. Same library, same
rem   runtime switch:
rem     set TRELLIS2_DEVICE=cuda / vulkan / cpu
rem
rem   Why this one exists next to the Ninja script: the VS generator finds MSVC
rem   and the CUDA toolset through Visual Studio itself, so there is no vcvars
rem   dance and no Ninja-plus-cl.exe environment to get right. It is the more
rem   forgiving route on a machine that already has Visual Studio.
rem
rem   IMPORTANT: this generator drives CUDA through MSBuild, so the CUDA
rem   Toolkit's **Visual Studio Integration** component has to be installed -
rem   it supplies the .props files that teach MSBuild about .cu. The Ninja
rem   script does not need it; this one does. If it is missing, CMake fails at
rem   "No CUDA toolset found".
rem
rem   Requires: Visual Studio 2022, CUDA Toolkit 12.8+ (for sm_120 / RTX 50xx),
rem   Vulkan SDK, and vcpkg with eigen3 + cgal. vcpkg is located automatically -
rem   see the lookup below; a clone in the project root is enough.

rem Which NVIDIA architectures to generate code for.
rem   75 = Turing (RTX 20xx)   86 = Ampere (RTX 30xx)
rem   89 = Ada    (RTX 40xx)   120 = Blackwell (RTX 50xx)
rem sm_120 needs CUDA 12.8 or newer. Several are allowed ("86;89;120"); each
rem entry costs build time, and CMake emits PTX per entry so a newer card than
rem the newest listed still runs by JIT.
set CUDA_ARCHS=120

rem Deliberately not build\, so an existing ROCm or Vulkan tree survives and
rem the server can pick between them with -lib.
set BUILD_DIR=build-cuda

rem Which library does the closest-surface search for the texture rebake.
rem OFF = CGAL (the reference), ON = tinybvh.
set FORCE_TINYBVH=ON

rem Find vcpkg, in this order:
rem   1. VCPKG_ROOT, if it is set and actually holds a vcpkg
rem   2. a copy in the project root - the layout TrenchBroom and friends use,
rem      and the one that survives being handed to someone else
rem   3. C:\vcpkg, where the other Windows scripts have always expected it
rem Nobody has to edit this file; setting VCPKG_ROOT still wins over all of it.
set VCPKG_DIR=
if defined VCPKG_ROOT if exist "%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" set VCPKG_DIR=%VCPKG_ROOT%
if not defined VCPKG_DIR if exist "%~dp0vcpkg\scripts\buildsystems\vcpkg.cmake" set VCPKG_DIR=%~dp0vcpkg
if not defined VCPKG_DIR if exist "C:\vcpkg\scripts\buildsystems\vcpkg.cmake" set VCPKG_DIR=C:\vcpkg
if not defined VCPKG_DIR (
  echo No vcpkg found. Looked at VCPKG_ROOT, %~dp0vcpkg and C:\vcpkg.
  echo.
  echo   git clone https://github.com/microsoft/vcpkg
  echo   vcpkg\bootstrap-vcpkg.bat
  echo   vcpkg\vcpkg install eigen3 cgal
  echo.
  echo Cloning it into the project root works and needs no environment
  echo variable - it is one of the locations checked above.
  pause
  exit /b 1
)
echo Using vcpkg at %VCPKG_DIR%

if not defined VULKAN_SDK (
  echo VULKAN_SDK is not set - install the Vulkan SDK, or open a new shell.
  echo To build CUDA only, set -DGGML_VULKAN=OFF below.
  pause
  exit /b 1
)

where nvcc >nul 2>nul
if errorlevel 1 (
  echo nvcc is not on PATH - install the CUDA Toolkit, or open a new shell.
  echo Without it ggml only reports "CUDA Toolkit not found" and drops the
  echo backend silently instead of failing.
  pause
  exit /b 1
)

rem Make glslc reachable in case the SDK installer did not put it on PATH.
set PATH=%VULKAN_SDK%\Bin;%PATH%

rem CMAKE_RUNTIME_OUTPUT_DIRECTORY must be absolute (relative paths get
rem reinterpreted per-subdir by ggml's nested CMakeLists). The VS generator
rem appends the configuration itself, so the DLLs land in %BUILD_DIR%\bin\Release.
set DLL_OUT=%CD%\%BUILD_DIR%\bin

rmdir /s /q %BUILD_DIR% 2>nul

cmake -B %BUILD_DIR% ^
  -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE="%VCPKG_DIR%\scripts\buildsystems\vcpkg.cmake" ^
  -DVCPKG_MANIFEST_FEATURES=cgal ^
  -DGGML_VULKAN=ON ^
  -DGGML_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCHS% ^
  -DBUILD_SHARED_LIBS=ON ^
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="%DLL_OUT%" ^
  -DTRELLIS2_CGAL=ON ^
  -DTRELLIS2_AUTOREMESHER=ON ^
  -DTRELLIS2_FORCE_TINYBVH=%FORCE_TINYBVH% ^
  -DTRELLIS2_BUILD_EXAMPLES=ON ^
  -DTRELLIS2_BUILD_TESTS=ON ^
  .

if errorlevel 1 (
  echo.
  echo CMake configure failed.
  echo   "No CUDA toolset found" means the CUDA Toolkit was installed without
  echo   its Visual Studio Integration component - rerun the CUDA installer and
  echo   select it, or use cmake-ninja-win64-cuda.bat, which does not need it.
  pause
  exit /b 1
)

rem The configure output above must contain:
rem   -- CUDA Toolkit found
rem   -- CGAL Alpha Wrap print remeshing: enabled
rem   -- PBR projection backend: cgal            (or tinybvh)
rem   -- AutoRemesher quad remeshing: enabled (Eigen 5.x.x)
rem "CUDA Toolkit not found" means the backend is missing from the DLL, and
rem TRELLIS2_DEVICE=cuda will later fall back to the CPU without saying much.

rem --parallel: MSBuild builds projects serially by default, unlike ninja.
cmake --build %BUILD_DIR% --config Release --parallel

if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)

echo.
echo Done: %BUILD_DIR%\bin\Release\trellis2.dll
echo       (+ ggml.dll, ggml-base.dll, ggml-cpu.dll, ggml-cuda.dll, ggml-vulkan.dll)
echo.
echo Start from server\:
echo   start_server.bat %BUILD_DIR%
echo.
echo Pick the backend at run time, without rebuilding:
echo   set TRELLIS2_DEVICE=cuda     :: CUDA (default, first GPU device)
echo   set TRELLIS2_DEVICE=vulkan   :: Vulkan
echo   set TRELLIS2_DEVICE=cpu      :: CPU
echo The variable has to be in the environment BEFORE the server starts - on
echo Windows, os.Setenv does not reach the DLL's own getenv().
echo.
echo Note: the library disables ggml's CUDA graphs, because they crash on HIP
echo and both backends share that code. On NVIDIA that may cost performance -
echo turn them back on with:
echo   set TRELLIS2_CUDA_GRAPHS=1
echo.
echo Handing the binaries to someone else? Ship the WHOLE
echo %BUILD_DIR%\bin\Release\ directory - it also contains the vcpkg
echo dependencies (gmp-10.dll, mpfr-6.dll, ...) that CGAL pulls in, and a
echo missing one fails at load time with a misleading "module not found"
echo about trellis2.dll. Plus server\trellis2-server.exe (the web UI is
echo embedded in it) and the GGUFs, which they can fetch themselves with
echo scripts/download_ggufs.sh. The target machine needs the Microsoft Visual
echo C++ Redistributable.
pause
endlocal
