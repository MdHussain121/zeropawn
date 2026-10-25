@echo off
setlocal
title ZeroPawn Control Center

rem Probe standard C++ compiler locations on Windows (WinLibs, MinGW, MSYS2)
if defined LOCALAPPDATA (
    if exist "%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin\g++.exe" (
        set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin;%PATH%"
    )
)
if exist "C:\mingw64\bin\g++.exe" (
    set "PATH=C:\mingw64\bin;%PATH%"
)
if exist "C:\msys64\mingw64\bin\g++.exe" (
    set "PATH=C:\msys64\mingw64\bin;%PATH%"
)

set PYTHONIOENCODING=utf-8

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH. Please install Python 3.11+ and ensure it is on PATH. 1>&2
    if "%~1"=="" pause
    exit /b 1
)

python -u "%~dp0cli.py" %*
set "EXITCODE=%errorlevel%"

if "%~1"=="" (
    pause
)

exit /b %EXITCODE%
