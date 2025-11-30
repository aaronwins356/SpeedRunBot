@echo off
REM =============================================================================
REM Minecraft Elytra Finder Bot - Live Bot Runner (Windows)
REM =============================================================================
REM This script starts the live Elytra Finder Bot on Windows.
REM
REM Usage:
REM   scripts\run_live_bot.bat                    Normal run
REM   scripts\run_live_bot.bat --dry-run          Test without sending commands
REM   scripts\run_live_bot.bat --help             Show help
REM
REM Environment:
REM   - Set MC_USERNAME and MC_PASSWORD environment variables, or
REM   - Create a .env file from .env.example
REM =============================================================================

setlocal enabledelayedexpansion

REM Configuration
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
set "PYTHON=python"
set "CONFIG_FILE=%PROJECT_DIR%\config.yaml"
set "VENV_DIR=%PROJECT_DIR%\venv"

REM Load .env file if it exists
if exist "%PROJECT_DIR%\.env" (
    echo [Elytra Bot] Loading environment from .env file
    for /f "usebackq tokens=1,* delims==" %%a in ("%PROJECT_DIR%\.env") do (
        REM Skip comments
        set "line=%%a"
        if not "!line:~0,1!"=="#" (
            set "%%a=%%b"
        )
    )
)

REM Activate virtual environment if it exists
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [Elytra Bot] Activating virtual environment
    call "%VENV_DIR%\Scripts\activate.bat"
)

REM Check Python
where %PYTHON% >nul 2>&1
if %errorlevel% neq 0 (
    echo [Error] Python not found. Please install Python 3.8+
    exit /b 1
)

REM Check dependencies
echo [Elytra Bot] Checking dependencies...
%PYTHON% -c "import torch; import numpy; import yaml" 2>nul
if %errorlevel% neq 0 (
    echo [Warning] Some dependencies are missing. Installing...
    pip install -r "%PROJECT_DIR%\requirements.txt"
)

REM Change to project directory
cd /d "%PROJECT_DIR%"

REM Print banner
echo.
echo ============================================================
echo            Minecraft Elytra Finder Bot
echo ============================================================
echo.

REM Safety reminder
echo WARNING: SAFETY REMINDER:
echo This bot is intended to be used only where automation is
echo explicitly allowed by the server owner. Do not use this
echo in violation of any server's terms of service.
echo.

REM Run the bot
echo [Elytra Bot] Starting live bot...
%PYTHON% main.py live-bot --config "%CONFIG_FILE%" %*

endlocal
