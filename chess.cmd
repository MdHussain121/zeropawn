@echo off
setlocal
if /i not "%~1"=="uci" (
    echo [DEPRECATION] 'chess.cmd' is deprecated and will be removed in a future release. Please use 'zeropawn.cmd'. 1>&2
)
"%~dp0zeropawn.cmd" %*
exit /b %errorlevel%
