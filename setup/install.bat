@echo off
echo ================================================
echo  YOLO11 food ingredient detection - environment install
echo ================================================

echo [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: please make sure Python 3.11 is installed
    pause
    exit /b 1
)

echo [2/3] Installing packages...
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements_training.txt
if errorlevel 1 (
    echo ERROR: package installation failed
    pause
    exit /b 1
)

echo [3/3] Creating folders...
mkdir weights 2>nul
mkdir preview 2>nul

echo.
echo Done!
echo.
echo Next steps:
echo   1. venv\Scripts\activate
echo   2. python scripts\prepare_yaml.py
echo   3. python setup\check_env.py
echo.
pause
