#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный скрипт для запуска всех основных частей программы:
1. Создание датасета
2. Обучение модели YOLO
3. Запуск обученной модели для детекции
4. Управление датасетами и классами
"""

import subprocess
import sys
import os
import fnmatch
from pathlib import Path
import locale
import io

# Настройка кодировки вывода для поддержки кириллицы в Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    # Установка локали для корректного отображения
    locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')

# ANSI коды для цветного вывода
COLORS = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "CYAN": "\033[96m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
    "ENDC": "\033[0m",
}

def color_text(text, color):
    """Возвращает текст с ANSI кодами цвета."""
    return f"{COLORS.get(color, '')}{text}{COLORS['ENDC']}"

def print_header():
    """Выводит заголовок меню."""
    print(color_text("\n" + "=" * 70, "HEADER"))
    print(color_text("   ГЛАВНОЕ МЕНЮ ПРОЕКТА COMPUTER VISION YOLO", "BOLD"))
    print(color_text("=" * 70, "HEADER"))
    print()

def print_menu(options):
    """Выводит пронумерованное меню с доступными опциями."""
    for i, (title, desc) in enumerate(options, start=1):
        print(f"  {color_text(str(i), 'CYAN')}. {color_text(title, 'GREEN')}")
        print(f"     {desc}")
        print()

def get_user_choice(max_option):
    """Запрашивает у пользователя выбор опции."""
    while True:
        try:
            choice = input(color_text("Выберите опцию (1-" + str(max_option) + "): ", "BLUE"))
            if choice.lower() in ['q', 'quit', 'exit']:
                return 0
            choice = int(choice)
            if 1 <= choice <= max_option:
                return choice
            else:
                print(color_text(f"Пожалуйста, введите число от 1 до {max_option}.", "RED"))
        except ValueError:
            print(color_text("Пожалуйста, введите число.", "RED"))

def run_script(script_path, description):
    """Запускает Python скрипт."""
    if not os.path.exists(script_path):
        print(color_text(f"Ошибка: файл {script_path} не найден.", "RED"))
        return False
    
    print(color_text(f"\nЗапуск: {description}", "BLUE"))
    print(color_text(f"Скрипт: {script_path}", "CYAN"))
    print(color_text("Для выхода из скрипта нажмите Ctrl+C в его окне.", "YELLOW"))
    print("-" * 40)
    
    try:
        result = subprocess.run([sys.executable, script_path], check=False)
        if result.returncode == 0:
            print(color_text(f"\nСкрипт завершился успешно.", "GREEN"))
        else:
            print(color_text(f"\nСкрипт завершился с кодом возврата {result.returncode}.", "YELLOW"))
        return True
    except FileNotFoundError:
        print(color_text(f"Ошибка: не удалось запустить Python или файл {script_path}.", "RED"))
        return False
    except KeyboardInterrupt:
        print(color_text("\nЗапуск прерван пользователем.", "YELLOW"))
        return False
    except Exception as e:
        print(color_text(f"Неожиданная ошибка при запуске: {e}", "RED"))
        return False

def run_dataset_creation():
    """Запускает меню создания датасета."""
    return run_script("run_dataset_creator.py", "Создание датасета YOLO")

def run_training():
    """Запускает обучение модели YOLO."""
    print(color_text("\nДоступные скрипты обучения:", "BLUE"))
    
    training_scripts = [
        ("train_yolo.py", "Основное обучение YOLO (простой)"),
        ("train_target_yolo.py", "Целевое обучение YOLO (интерактивный с выбором датасета)"),
        ("train_target_yolo_simple.py", "Целевое обучение YOLO (с аргументами командной строки)"),
    ]
    
    for i, (script, desc) in enumerate(training_scripts, start=1):
        exists = "✓" if os.path.exists(script) else "✗"
        print(f"  {i}. {script} - {desc} {exists}")
    
    choice = input(color_text("\nВыберите скрипт обучения (1-3) или Enter для train_yolo.py: ", "BLUE"))
    
    if choice == "1" or choice == "":
        script = "train_yolo.py"
        description = "Основное обучение YOLO"
    elif choice == "2":
        script = "train_target_yolo.py"
        description = "Целевое обучение YOLO с интерактивным выбором датасета"
    elif choice == "3":
        script = "train_target_yolo_simple.py"
        # Предлагаем параметры для простого скрипта
        print(color_text("\n📊 Параметры для train_target_yolo_simple.py:", "CYAN"))
        print("  Доступные аргументы командной строки:")
        print("    --dataset augmented/original/custom (по умолчанию: augmented)")
        print("    --dataset-path <путь> (только для custom)")
        print("    --model <путь> (по умолчанию: yolo26n.pt)")
        print("    --epochs <число> (по умолчанию: 50)")
        print("    --imgsz <число> (по умолчанию: 640)")
        print("    --batch <число> (по умолчанию: 16)")
        print("    --name <имя> (по умолчанию: targeted_yolo_model)")
        print("    --force (перезаписать существующие модели)")
        
        args_input = input(color_text("\nВведите аргументы командной строки (или Enter для значений по умолчанию): ", "BLUE"))
        
        if args_input:
            script = f"train_target_yolo_simple.py {args_input}"
            description = f"Целевое обучение YOLO с аргументами: {args_input}"
        else:
            description = "Целевое обучение YOLO со значениями по умолчанию"
    else:
        print(color_text("Неверный выбор, используется train_yolo.py", "YELLOW"))
        script = "train_yolo.py"
        description = "Основное обучение YOLO"
    
    # Проверяем существование основного файла скрипта (без аргументов)
    script_file = script.split()[0] if ' ' in script else script
    if not os.path.exists(script_file):
        print(color_text(f"Ошибка: файл {script_file} не найден.", "RED"))
        return False
    
    return run_script(script, description)

def run_detection():
    """Запускает детекцию с камеры с использованием каскадного классификатора."""
    return run_script("detect_from_camera_cascade.py", "Детекция объектов с камеры (каскадный классификатор)")

def show_project_info():
    """Показывает информацию о проекте."""
    print(color_text("\n" + "=" * 70, "HEADER"))
    print(color_text("   ИНФОРМАЦИЯ О ПРОЕКТЕ", "BOLD"))
    print(color_text("=" * 70, "HEADER"))
    
    print(color_text("\n📁 Структура проекта:", "CYAN"))
    
    # Проверяем существование основных файлов
    files_to_check = [
        ("run_dataset_creator.py", "Меню создания датасета"),
        ("train_yolo.py", "Обучение модели YOLO"),
        ("detect_from_camera.py", "Детекция с камеры"),
        ("requirements.txt", "Зависимости проекта"),
    ]
    
    for file, desc in files_to_check:
        exists = color_text("✓ СУЩЕСТВУЕТ", "GREEN") if os.path.exists(file) else color_text("✗ ОТСУТСТВУЕТ", "RED")
        print(f"  {file:30} - {desc:30} {exists}")
    
    # Проверяем существование папок
    folders_to_check = [
        ("datasets/", "Датасеты"),
        ("augmentations/", "Аугментации"),
        ("runs/", "Результаты обучения"),
        ("logs/", "Логи"),
    ]
    
    print(color_text("\n📂 Папки проекта:", "CYAN"))
    for folder, desc in folders_to_check:
        exists = color_text("✓ СУЩЕСТВУЕТ", "GREEN") if os.path.exists(folder) else color_text("✗ ОТСУТСТВУЕТ", "YELLOW")
        print(f"  {folder:30} - {desc:30} {exists}")
    
    # Проверяем существование обученных моделей
    print(color_text("\n🤖 Обученные модели:", "CYAN"))
    model_paths = [
        "best_one_yet/weights/best.pt",
        "best_one_yet/weights/last.pt",
        "runs/detect/face_detection_yolo26/weights/best.pt",
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / (1024*1024)
            print(f"  {model_path:50} - {size:.1f} MB {color_text('✓', 'GREEN')}")
        else:
            print(f"  {model_path:50} - {color_text('✗ не найден', 'YELLOW')}")
    
    print(color_text("\n💡 Рекомендации:", "YELLOW"))
    print("  1. Сначала создайте датасет (опция 1)")
    print("  2. Затем обучите модель (опция 2)")
    print("  3. Наконец, запустите детекцию (опция 3)")
    print("  4. Используйте аугментацию для улучшения датасета (опция 4)")
    
    input(color_text("\nНажмите Enter для продолжения...", "BLUE"))

def check_dependencies():
    """Проверяет установленные зависимости."""
    print(color_text("\n🔍 Проверка зависимостей...", "BLUE"))
    
    try:
        import importlib
        dependencies = [
            ("ultralytics", "YOLO"),
            ("cv2", "OpenCV"),
            ("numpy", "NumPy"),
            ("torch", "PyTorch"),
        ]
        
        all_ok = True
        for package, name in dependencies:
            try:
                importlib.import_module(package)
                print(f"  {name:15} - {color_text('✓ УСТАНОВЛЕН', 'GREEN')}")
            except ImportError:
                print(f"  {name:15} - {color_text('✗ ОТСУТСТВУЕТ', 'RED')}")
                all_ok = False
        
        if not all_ok:
            print(color_text("\n⚠️  Некоторые зависимости отсутствуют.", "YELLOW"))
            print(color_text("   Установите их командой: pip install -r requirements.txt", "CYAN"))
        else:
            print(color_text("\n✅ Все зависимости установлены.", "GREEN"))
    
    except Exception as e:
        print(color_text(f"Ошибка при проверке зависимостей: {e}", "RED"))
    
    input(color_text("\nНажмите Enter для продолжения...", "BLUE"))

def main():
    """Главная функция."""
    # Основные опции меню
    main_options = [
        ("📁 Создание датасета", "Создание датасетов YOLO из различных источников"),
        ("🤖 Обучение модели", "Обучение модели YOLO на вашем датасете"),
        ("🎯 Детекция объектов", "Запуск обученной модели для детекции с камеры"),
        ("ℹ️  Информация о проекте", "Показать структуру и состояние проекта"),
        ("🔍 Проверка зависимостей", "Проверить установленные библиотеки"),
        ("🚪 Выход", "Завершить работу программы"),
    ]
    
    while True:
        print_header()
        print_menu(main_options)
        
        choice = get_user_choice(len(main_options))
        
        if choice == 0 or choice == len(main_options):
            print(color_text("\nДо свидания! 👋", "BLUE"))
            break
        
        if choice == 1:
            run_dataset_creation()
        elif choice == 2:
            run_training()
        elif choice == 3:
            run_detection()
        elif choice == 4:
            show_project_info()
        elif choice == 5:
            check_dependencies()
        
        # Пауза между операциями
        if choice != len(main_options):
            input(color_text("\nНажмите Enter для возврата в меню...", "BLUE"))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(color_text("\n\nПрограмма прервана пользователем.", "YELLOW"))
    except Exception as e:
        print(color_text(f"\nКритическая ошибка: {e}", "RED"))
        import traceback
        traceback.print_exc()