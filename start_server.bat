@echo off
setlocal

rem --- Build what is stale, then start the Go demo server (cargo-run style) ---
rem   Usage: start_server.bat [build dir]      (default: build)
rem   e.g.   start_server.bat build-master
rem
rem   Both build steps are incremental and no-op when nothing changed, so this is
rem   the normal way to start the server, not just a first-run helper.
rem
rem   The CMake build dir has to be configured once beforehand — that is where
rem   the backend is chosen, which this script must not guess. Use
rem   cmake-ninja-win64-rocm.bat (or -cpu / -vulkan).

rem Everything is relative to the script's own directory, not the caller's, so
rem this works from any shell and from a double-click.
cd /d "%~dp0"

set BUILD=build
if not "%~1"=="" set BUILD=%~1

set ROCM_BIN=C:\Program Files\AMD\ROCm\6.4\bin
set DLL_DIR=%~dp0%BUILD%\bin\Release

rem The server executable lives in server\, and Windows resolves a loaded DLL's
rem own imports (ggml.dll, ggml-base.dll, ggml-cpu.dll, ggml-hip.dll) from the
rem EXE's directory and PATH — never from the directory the DLL itself sits in.
rem Without DLL_DIR on PATH, LoadLibrary fails with a misleading "module not
rem found" that appears to be about trellis2.dll rather than its dependencies.
rem Set before the builds too, so the ROCm toolchain is reachable.
set PATH=%ROCM_BIN%;%DLL_DIR%;%PATH%

if not exist "%~dp0%BUILD%\CMakeCache.txt" (
  echo %BUILD%\ is not configured.
  echo Configure and build it once, e.g.:
  echo   cmake-ninja-win64-rocm.bat      ^(HIP + Vulkan^)
  echo   cmake-ninja-win64-cuda.bat      ^(CUDA + Vulkan^)
  echo   cmake-ninja-win64-vulkan.bat    ^(Vulkan^)
  echo   cmake-ninja-win64-cpu.bat       ^(CPU only^)
  pause
  exit /b 1
)

echo [1/2] trellis2.dll
cmake --build "%BUILD%" --config Release
if errorlevel 1 (
  echo.
  echo Library build failed - see the output above. Common causes:
  echo   * A server is still running and holding trellis2.dll locked.
  echo   * "CMake Generate step failed" after a ggml bump: stale cache entries.
  echo     Drop them individually, e.g. for MATH_LIBRARY-NOTFOUND:
  echo       cmake -B %BUILD% -U MATH_LIBRARY
  echo     Otherwise reconfigure %BUILD%\ ^(cmake-ninja-win64-*.bat^).
  pause
  exit /b 1
)

echo [2/2] trellis2-server.exe
pushd server
go build -o trellis2-server.exe .
if errorlevel 1 (
  popd
  echo.
  echo Server build failed.
  pause
  exit /b 1
)

rem GGML_CUDA_DISABLE_GRAPHS is deliberately not set here: the server sets it
rem itself (see the comment in server/main.go), because on Windows it has to go
rem through the CRT's own environment copy to reach the library's getenv().

rem TRELLIS2_TIMING, by contrast, belongs exactly here. A variable that is
rem already in the process environment when the exe starts is part of the
rem environment block the DLL's CRT snapshots at load, so it reaches the
rem library's getenv() without the setNativeEnv detour that os.Setenv needs.
rem
rem It makes the library report per-stage timings plus the two things that
rem decide whether a cascade run can finish at all: the scaffold token count the
rem budget settled on ([cascade]), and the shape decoder's per-level voxel
rem counts against this backend's mul_mat column cap ([shape_dec] / [shape_enc]).
rem See docs/bugs/rocm-texture-stage-invalid-configuration.md.
rem
rem To turn it off, delete the variable rather than setting it to 0 - the
rem library tests for presence, so TRELLIS2_TIMING=0 still enables it:
rem   set TRELLIS2_TIMING=
set TRELLIS2_TIMING=1
::set TRELLIS2_SDPA_FLASH=1
::set TRELLIS2_SDPA_EXACT=1
::set TRELLIS2_SHAPE_ENC_CPU=1
::set TRELLIS2_SHAPE_DEC_CPU=1
::set TRELLIS2_NO_CHUNK=1

rem Explicit .\ — cmd does not necessarily resolve executables from the current
rem directory (NoDefaultCurrentDirectoryInExePath), which fails with exit 9009.
echo.
.\trellis2-server.exe ^
  -lib "%DLL_DIR%\trellis2.dll" ^
  -ggufs ..\ggufs ^
  -store ..\generations ^
  -addr :8742 ^
  -unload-idle

set RC=%ERRORLEVEL%
popd
if not "%RC%"=="0" (
  echo.
  echo Server exited with an error ^(exit %RC%^).
  pause
)
endlocal
