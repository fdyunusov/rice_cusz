@echo off
setlocal
cd /d "%~dp0"

where cl >nul 2>&1
if %ERRORLEVEL%==0 (
  echo Building with nvcc + MSVC ...
  nvcc -O3 -std=c++17 -Iinclude -arch=native src\main.cu src\lorenzo.cu src\rice.cu -o sz_rice.exe
  if errorlevel 1 exit /b 1
  sz_rice.exe --n 1048576 --eb 1e-3 --rice-k 2 --self-test
  exit /b %ERRORLEVEL%
)

echo cl.exe not on PATH — building via WSL instead ...
wsl -e bash -lc "cd '/mnt/c/Fuad D diski/COP6527/sz_rice' && nvcc -O3 -std=c++17 -Iinclude -arch=sm_89 src/main.cu src/lorenzo.cu src/rice.cu -o sz_rice && ./sz_rice --n 1048576 --eb 1e-3 --rice-k 2 --self-test"
exit /b %ERRORLEVEL%
