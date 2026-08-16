@echo off
setlocal
cd ..

rem --- PyTorch reference generation of the Einstein bust, 1024 cascade -------
rem   Three steps: the reference pipeline down to the decoder's 7 channels,
rem   our dual-grid extractor, then the generations/ record.
rem
rem   Pass --resume through as the first argument to skip straight to the
rem   decode using the checkpoint the last attempt left behind:
rem       refgen_einstein.bat --resume
rem
rem   Every step pauses on failure, so the error stays readable instead of
rem   scrolling away behind the two commands that used to run after it.

set EXTRA=%1

echo [1/3] reference pipeline
uv run --extra rocm python scripts/ref_generate.py ^
    --image assets/einstein.png --seed 42 --pipeline-type 1024_cascade ^
    --out dumps/einstein_ref1024.fdgvox --out-json dumps/einstein_ref1024.json ^
    %EXTRA%
if errorlevel 1 (
  echo.
  echo Schritt 1 fehlgeschlagen ^(exit %ERRORLEVEL%^) - siehe Ausgabe oben.
  echo Haeufigste Ursache ist der Decode-Schritt am Ende: der expandiert auf
  echo ~3,8 Mio. Voxel und ist der einzige Speicherfresser im Lauf.
  echo Der Checkpoint nach dem Sampler liegt in
  echo   dumps\einstein_ref1024.state.npz
  echo Ein erneuter Versuch nur des Decodes:  refgen_einstein.bat --resume
  pause
  exit /b 1
)

echo.
echo [2/3] dual-grid extraction
build\bin\Release\dual_grid_cli.exe dumps\einstein_ref1024.fdgvox dumps\einstein_ref1024.t2mesh
if errorlevel 1 (
  echo.
  echo Schritt 2 fehlgeschlagen ^(exit %ERRORLEVEL%^).
  pause
  exit /b 1
)

echo.
echo [3/3] publish to generations\
uv run --extra rocm python scripts/ref_publish_generation.py ^
    --mesh dumps/einstein_ref1024.t2mesh --image assets/einstein.png ^
    --info dumps/einstein_ref1024.json --force
if errorlevel 1 (
  echo.
  echo Schritt 3 fehlgeschlagen ^(exit %ERRORLEVEL%^).
  pause
  exit /b 1
)

echo.
echo Fertig. Im Viewer die Seite neu laden, die Kachel heisst "1024 . PyTorch".
pause
endlocal
