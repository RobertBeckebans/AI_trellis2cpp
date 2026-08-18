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
rem ── Numerics knobs ──────────────────────────────────────────────────────
rem Everything below changes WHAT is computed, not just how fast. All of them
rem are recorded per generation in the manifest's "runConfig", so a stored
rem result says which settings produced it - do not rely on remembering.
rem Like TRELLIS2_TIMING they test for presence, so "=0" does NOT switch one
rem off; comment the line out or clear it with "set TRELLIS2_XXX=".

rem Pin the attention path. Default is size-gated: the exact materialized
rem [L_k, L_q, heads] softmax while that matrix stays under 1 GiB - SS-flow
rem (~0.8 GiB) and 512-SLAT (~0.4 GiB) - and ggml_flash_attn_ext above it.
rem
rem That gate answered a MEMORY question and no longer does. The exact path now
rem builds its score matrix in blocks of query rows (TRELLIS2_SDPA_BLOCK_MB,
rem default 1 GiB), which is exact without any correction term because soft_max
rem runs along keys and every query row is independent. So EXACT at the cascade
rem tiers allocates fine - measured 13-15 GB of 32 at 1024 - and it is what
rem restores the geometry there: finest-level expansion 3.55 -> 4.06 into the
rem healthy band, 4.3M -> 7.9M voxels, and FEWER non-manifold edges, at ~2.3x
rem wall clock. The server UI exposes this per job as "attention"; these
rem variables are the headless equivalent and win over a host that sends AUTO.
rem See docs/progress/backend-parity_1024-exact-attention.md.
rem
rem The two are NOT interchangeable. ggml's CUDA/HIP flash kernels accumulate in
rem F16 and ignore ggml_flash_attn_ext_set_prec() outright, so on ROCm flash
rem costs 5.198e-02 rel-L2 per SS-flow forward against 3.050e-06 exact, and
rem 89.8% latent sign agreement against 99.9% - roughly one voxel in ten flips
rem in the coarse occupancy structure. FLASH is therefore a deliberate choice of
rem the less faithful kernel, and it is the faster one.
rem
rem TRELLIS2_SDPA_EXACT_MAX_MB moves the gate, 0 disables exact entirely. On its
rem own it is rarely what you want: it only lets a bigger score through the gate,
rem which without the blocking meant more memory and more time for no visible
rem change. TRELLIS2_SDPA_EXACT is the switch that matters.
::set TRELLIS2_SDPA_FLASH=1
::set TRELLIS2_SDPA_EXACT=1
::set TRELLIS2_SDPA_EXACT_MAX_MB=16384
::set TRELLIS2_SDPA_EXACT_MAX_MB=2048

rem Force the texture stage's shape ENCODER onto the CPU. Escape hatch only -
rem the dispatch-limit crash it existed for is fixed inside the encoder (the
rem skip mean is summed as strided slices now). Slow.
::set TRELLIS2_SHAPE_ENC_CPU=1

rem Force the shape DECODER onto the CPU. By default it is placed by free VRAM
rem - GPU when the card can hold the decode once the flow DiTs are freed, CPU
rem otherwise - and it falls back to the CPU by itself on a load OOM, so there
rem is never a mid-generation failure. TRELLIS2_SHAPE_DEC_GPU forces the other
rem way. The manifest records where it actually landed, not what was asked.
::set TRELLIS2_SHAPE_DEC_CPU=1

rem Run every level in ONE graph, past the column count the backend will write.
rem Not a performance knob: on ROCm/HIP the columns past 2^21 (f16 weights) or
rem 2^19 (f32) are never written and stay zero, silently and with
rem GGML_STATUS_SUCCESS - the 1024^3 Einstein bust losing every face past 2^21
rem voxels is what this produces. It exists to reproduce that defect on purpose,
rem next to tools/rocblas_col_truncation.py. Never for a real generation.
rem See docs/bugs/ggml-rocm-mul-mat-column-limit.md.
::set TRELLIS2_NO_CHUNK=1

rem Raise the ENCODER's block-wise gate without touching the block size (that
rem stays TRELLIS2_CHUNK_ROWS, default 262144). Unset, the gate is the block
rem size, so a level splits as soon as it outgrows one block rather than when
rem the backend requires it. 2097152 keeps level 0 chunked at 1024 and returns
rem level 1 to a single graph. Measured 2026-08-16: bit-identical output either
rem way at 1024, so this is a diagnostic, not a fix. Levels that still split are
rem unaffected.
::set TRELLIS2_ENC_CHUNK_ROWS=2097152

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
