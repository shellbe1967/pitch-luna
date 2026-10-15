@echo off
rem Singing trainer: setup and test run (double-click)
chcp 65001 >NUL
cd /d %~dp0
where python >NUL 2>&1
if %errorlevel%==0 (
    python setup_check.py %*
) else (
    py setup_check.py %*
)
echo.
pause
