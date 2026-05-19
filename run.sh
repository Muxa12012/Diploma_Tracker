#!/bin/bash
# Скрипт для запуска главного меню проекта на Linux/macOS

echo "========================================"
echo "   Запуск Practice_Diploma_Tracker"
echo "========================================"
echo ""

# Проверка наличия Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден. Установите Python3."
    exit 1
fi

# Проверка версии Python
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Python версии $PYTHON_VERSION обнаружен"

# Проверка виртуального окружения
if [ -d "venv" ] || [ -d ".venv" ]; then
    echo "📦 Виртуальное окружение обнаружено"
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        source .venv/bin/activate
    fi
    echo "✅ Виртуальное окружение активировано"
fi

# Проверка зависимостей
echo "🔍 Проверка зависимостей..."
if [ -f "requirements.txt" ]; then
    echo "📋 Файл requirements.txt найден"
    
    # Проверяем основные зависимости
    if python3 -c "import ultralytics" &> /dev/null; then
        echo "✅ Ultralytics YOLO установлен"
    else
        echo "⚠️  Ultralytics YOLO не установлен"
        read -p "Установить зависимости? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pip install -r requirements.txt
        fi
    fi
else
    echo "⚠️  Файл requirements.txt не найден"
fi

# Запуск главного меню
echo ""
echo "🚀 Запуск главного меню..."
echo ""

python3 run_project.py

# Деактивация виртуального окружения (если было активировано)
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
    echo "✅ Виртуальное окружение деактивировано"
fi

echo ""
echo "========================================"
echo "   Работа завершена"
echo "========================================"