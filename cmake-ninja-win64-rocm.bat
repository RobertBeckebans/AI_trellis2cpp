@echo off
setlocal

rem --- Shared-library build with BOTH optional backends, for comparing them ---
rem   Like cmake-ninja-win64-rocm-cgal.bat, plus the AutoRemesher quad stage, so
rem   the viewer offers "watertight print wrap (CGAL)" and "quad retopology"
rem   at the same time and they can be judged against each other on one model.
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
  echo VULKAN_SDK ist nicht gesetzt - Vulkan SDK installieren oder Shell neu
  echo oeffnen. Nur ROCm bauen: -DGGML_VULKAN=OFF unten setzen.
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
  echo CMake-Konfiguration fehlgeschlagen.
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
  echo Build fehlgeschlagen.
  echo.
  echo Bricht der Build in print_remesh.cpp mit
  echo   'boost/mpl/aux_/preprocessed/plain / or.hpp' file not found
  echo ab, dann ist das das bekannte Boost.MPL/clang-Problem - es tritt auch
  echo mit unveraenderten Quellen auf und hat nichts mit dieser Aenderung zu
  echo tun. Siehe docs/progress/autoremesher-quad-remesh_phase2-tinybvh-projection.md
  pause
  exit /b 1
)

echo.
echo Fertig: build\bin\Release\trellis2.dll
echo         (+ ggml.dll, ggml-base.dll, ggml-cpu.dll, ggml-hip.dll, ggml-vulkan.dll)
echo.
echo Backend zur Laufzeit waehlen, ohne neu zu bauen:
echo   set TRELLIS2_DEVICE=rocm     :: HIP  (Default, erstes GPU-Geraet)
echo   set TRELLIS2_DEVICE=vulkan   :: Vulkan
echo   set TRELLIS2_DEVICE=cpu      :: CPU
echo Die Variable muss in der Umgebung stehen, BEVOR der Server startet - der
echo Weg ueber os.Setenv erreicht das getenv() der DLL auf Windows nicht.
echo Passt der Name auf kein GPU-Geraet, faellt es auf die CPU zurueck und sagt
echo es; es wird bewusst nicht das andere Backend genommen.
echo.
echo Vergleich im Viewer: "watertight print wrap (CGAL)" und "quad retopology"
echo sind einzeln und gemeinsam schaltbar. Der Server loggt fuer jeden Export
echo einen Zeitblock und meldet Quads / offene Kanten / Flaechenerhalt.
echo.
echo Offline auf einem bereits erzeugten Mesh, ohne Server:
echo   build\bin\Release\quad_remesh_cli.exe generations\^<job^>\mesh.t2mesh out.obj --target-quads 20000 --min-area 0
pause
endlocal
