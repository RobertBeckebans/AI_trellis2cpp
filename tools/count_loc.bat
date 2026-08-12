cd ..
cloc-2.04.exe ^
  --exclude-dir=.venv,build,target,libs,extern,docs,assets ^
  --exclude-ext=json,xml,txt ^
  --fullpath ^
  --by-file-by-lang ^
  .
pause
