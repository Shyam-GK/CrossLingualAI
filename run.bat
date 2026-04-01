@echo off
REM SecureNews - Run Script for Windows
REM This script sets up and runs the SecureNews application

echo 🚀 Starting SecureNews Application...
echo.

REM Check if py launcher (Windows Python Launcher) is installed
where py >nul 2>&1
if %errorlevel% equ 0 (
    echo 🔍 Checking for Python 3.11 using py launcher...
    py -3.11 --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=py -3.11
        echo ✅ Python 3.11 detected via 'py -3.11'
    ) else (
        echo ⚠️  Python 3.11 not found via py launcher.
    )
)

if "%PYTHON_CMD%"=="" (
    echo 🔍 Checking for default 'python'...
    python --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=python
        echo ✅ Default 'python' detected.
    ) else (
        echo ❌ Python is not installed. Please install Python 3.11 (Recommended).
        pause
        exit /b 1
    )
)

REM Check if ffmpeg is installed
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  FFmpeg is not installed.
    echo    Please download and install FFmpeg from: https://ffmpeg.org/download.html
    echo    Make sure to add it to your PATH.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv_311" (
    echo 📦 Creating virtual environment using %PYTHON_CMD%...
    %PYTHON_CMD% -m venv venv_311
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv_311\Scripts\activate.bat

REM Install/upgrade pip
echo 📥 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📦 Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt

REM Create necessary directories
if not exist "uploads" mkdir uploads
if not exist "outputs" mkdir outputs

echo.
echo ✅ Setup complete!
echo.
echo 🌐 Starting Flask server...
echo    Open your browser and navigate to: http://localhost:5001
echo.
echo ⚠️  Note: First run will download ML models (Whisper, NLLB, etc.)
echo    This may take 10-15 minutes depending on your internet connection.
echo.

REM Run the application
python app.py

pause

