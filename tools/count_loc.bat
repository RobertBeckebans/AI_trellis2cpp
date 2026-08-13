cd ..
cloc-2.04.exe ^
  --exclude-dir=.venv,build,target,libs,third_party,ggml,extern,docs,assets ^
  --exclude-ext=json,xml,txt,log,bat ^
  --fullpath ^
  --by-file-by-lang ^
  .
pause
