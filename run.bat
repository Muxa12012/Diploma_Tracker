@echo off
REM Скрипт для запуска главного меню проекта на Windows

REM Установка кодировки UTF-8 для поддержки кириллицы
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ========================================
echo    Запуск Practice_Diploma_Tracker
echo ========================================
echo.

REM Проверка наличия Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Python не найден. Установите Python.
    pause
    exit /b 1
)

REM Получение версии Python
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python версии %PYTHON_VERSION% обнаружен

REM Проверка виртуального окружения
if exist venv\ (
    echo 📦 Виртуальное окружение обнаружено
    call venv\Scripts\activate.bat
    echo ✅ Виртуальное окружение активировано
) else if exist .venv\ (
    echo 📦 Виртуальное окружение обнаружено
    call .venv\Scripts\activate.bat
    echo ✅ Виртуальное окружение активировано
)

REM Проверка зависимостей
echo 🔍 Проверка зависимостей...
if exist requirements.txt (
    echo 📋 Файл requirements.txt найден
    
    REM Проверяем основные зависимости
    python -c "import ultralytics" >nul 2>nul
    if %errorlevel% equ 0 (
        echo ✅ Ultralytics YOLO установлен
    ) else (
        echo ⚠️  Ultralytics YOLO не установлен
        set /p install_deps="Установить зависимости? (y/n): "
        if /i "%install_deps%"=="y" (
            pip install -r requirements.txt
        )
    )
) else (
    echo ⚠️  Файл requirements.txt не найден
)

REM Запуск главного меню
echo.
echo 🚀 Запуск главного меню...
echo.

python run_project.py

REM Деактивация виртуального окружения (если было активировано)
if not "%VIRTUAL_ENV%"=="" (
    deactivate
    echo ✅ Виртуальное окружение деактивировано
)

echo.
echo ========================================
echo    Работа завершена
echo ========================================
pause