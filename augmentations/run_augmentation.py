#!/usr/bin/env python3
"""
Скрипт для запуска аугментации из командной строки.
Просто вызывает main.py с поддержкой аргументов командной строки.
"""
import sys
import os

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import main

if __name__ == "__main__":
    sys.exit(main())