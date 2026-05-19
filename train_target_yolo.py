#!/usr/bin/env python3
"""
Скрипт для целевого обучения модели YOLO с возможностью выбора датасета.
Поддерживает обучение на аугментированном или неаугментированном датасете.
Модель сохраняется в папку targeted_yolo и перезаписывает предыдущую версию.
"""

from ultralytics import YOLO
import os
import shutil
import sys
import random
import yaml
from pathlib import Path

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
    """Выводит заголовок."""
    print(color_text("\n" + "=" * 60, "HEADER"))
    print(color_text("   ЦЕЛЕВОЕ ОБУЧЕНИЕ МОДЕЛИ YOLO", "BOLD"))
    print(color_text("=" * 60, "HEADER"))
    print()

def get_user_choice(prompt, options):
    """Запрашивает у пользователя выбор из списка опций."""
    print(prompt)
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    
    while True:
        try:
            choice = input(color_text("\nВыберите опцию (1-" + str(len(options)) + "): ", "BLUE"))
            if choice.lower() in ['q', 'quit', 'exit']:
                return None
            choice = int(choice)
            if 1 <= choice <= len(options):
                return choice - 1
            else:
                print(color_text(f"Пожалуйста, введите число от 1 до {len(options)}.", "RED"))
        except ValueError:
            print(color_text("Пожалуйста, введите число.", "RED"))

def check_dataset_exists(dataset_path):
    """Проверяет существование датасета и data.yaml файла."""
    if not os.path.exists(dataset_path):
        print(color_text(f"⚠️  Папка датасета не найдена: {dataset_path}", "YELLOW"))
        return False
    
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml_path):
        print(color_text(f"⚠️  Файл data.yaml не найден в датасете: {data_yaml_path}", "YELLOW"))
        return False
    
    print(color_text(f"✅ Датасет найден: {dataset_path}", "GREEN"))
    return True

def prepare_targeted_yolo_folder():
    """Подготавливает папку targeted_yolo для сохранения моделей."""
    target_folder = "targeted_yolo"
    
    # Создаем папку если не существует
    os.makedirs(target_folder, exist_ok=True)
    
    # Проверяем существующие модели в папке
    existing_models = []
    for file in os.listdir(target_folder):
        if file.endswith('.pt'):
            existing_models.append(os.path.join(target_folder, file))
    
    if existing_models:
        print(color_text(f"📁 Найдено {len(existing_models)} существующих моделей в {target_folder}:", "CYAN"))
        for model in existing_models:
            size = os.path.getsize(model) / (1024*1024)
            print(f"  - {os.path.basename(model)} ({size:.1f} MB)")
        
        # Предлагаем удалить старые модели
        choice = input(color_text("\nУдалить старые модели перед обучением? (y/n): ", "YELLOW"))
        if choice.lower() == 'y':
            for model in existing_models:
                try:
                    os.remove(model)
                    print(color_text(f"  Удалено: {os.path.basename(model)}", "GREEN"))
                except Exception as e:
                    print(color_text(f"  Ошибка удаления {model}: {e}", "RED"))
    
    return target_folder

def get_training_parameters():
    """Запрашивает у пользователя параметры обучения."""
    print(color_text("\n📊 Настройка параметров обучения:", "CYAN"))
    
    # Количество эпох
    while True:
        epochs_input = input(color_text("Количество эпох (по умолчанию 50): ", "BLUE"))
        if epochs_input == "":
            epochs = 50
            break
        try:
            epochs = int(epochs_input)
            if epochs > 0:
                break
            else:
                print(color_text("Количество эпох должно быть положительным числом.", "RED"))
        except ValueError:
            print(color_text("Пожалуйста, введите число.", "RED"))
    
    # Размер изображения
    while True:
        imgsz_input = input(color_text("Размер изображения (по умолчанию 640): ", "BLUE"))
        if imgsz_input == "":
            imgsz = 640
            break
        try:
            imgsz = int(imgsz_input)
            if imgsz >= 32:
                break
            else:
                print(color_text("Размер изображения должен быть не менее 32.", "RED"))
        except ValueError:
            print(color_text("Пожалуйста, введите число.", "RED"))
    
    # Размер батча
    while True:
        batch_input = input(color_text("Размер батча (по умолчанию 16): ", "BLUE"))
        if batch_input == "":
            batch = 16
            break
        try:
            batch = int(batch_input)
            if batch > 0:
                break
            else:
                print(color_text("Размер батча должен быть положительным числом.", "RED"))
        except ValueError:
            print(color_text("Пожалуйста, введите число.", "RED"))
    
    return epochs, imgsz, batch

def split_dataset(dataset_path, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15):
    """
    Автоматически разделяет изображения датасета на тренировочную, тестовую и валидационную выборки.
    Создает файлы train.txt, target.txt, valid.txt в папке датасета.
    """
    # Проверяем пропорции
    total_ratio = train_ratio + test_ratio + val_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Сумма пропорций должна быть равна 1.0, получено {total_ratio}")
    
    # Ищем папку с изображениями
    images_dir = os.path.join(dataset_path, "images")
    if not os.path.exists(images_dir):
        # Проверяем вложенную структуру images/train/
        images_dir = os.path.join(dataset_path, "images", "train")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Папка с изображениями не найдена в {dataset_path}")
    
    # Собираем все изображения
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                # Полный путь к изображению
                full_path = os.path.join(root, file)
                # Относительный путь от корня проекта (текущей рабочей директории)
                rel_path = os.path.relpath(full_path, os.getcwd())
                # Нормализуем разделители для совместимости с YOLO
                rel_path = rel_path.replace('\\', '/')
                image_files.append(rel_path)
    
    if not image_files:
        raise ValueError(f"Не найдено изображений в {images_dir}")
    
    print(color_text(f"📊 Найдено {len(image_files)} изображений", "CYAN"))
    
    # Перемешиваем случайным образом
    random.shuffle(image_files)
    
    # Вычисляем размеры выборок
    total = len(image_files)
    train_count = int(total * train_ratio)
    test_count = int(total * test_ratio)
    val_count = total - train_count - test_count  # остаток для валидации
    
    # Гарантируем минимум одно изображение в каждой выборке при достаточном количестве данных
    if total >= 3:
        if test_count == 0:
            test_count = 1
            val_count = 1
            train_count = total - 2
        elif val_count == 0:
            val_count = 1
            train_count = total - test_count - 1
    elif total == 2:
        # Если всего 2 изображения: train=1, test=1, val=0
        if test_count == 0:
            test_count = 1
            train_count = 1
            val_count = 0
        else:
            train_count = 1
            val_count = 0
    elif total == 1:
        # Если всего 1 изображение: train=1, test=0, val=0
        train_count = 1
        test_count = 0
        val_count = 0
    
    # Пересчитываем разделение с обновленными counts
    train_files = image_files[:train_count]
    test_files = image_files[train_count:train_count + test_count]
    val_files = image_files[train_count + test_count:]
    
    print(color_text(f"  🚂 Тренировочная выборка: {len(train_files)} изображений ({train_ratio*100:.0f}%)", "GREEN"))
    print(color_text(f"  🎯 Тестовая выборка (target): {len(test_files)} изображений ({test_ratio*100:.0f}%)", "YELLOW"))
    print(color_text(f"  📋 Валидационная выборка: {len(val_files)} изображений ({val_ratio*100:.0f}%)", "BLUE"))
    
    # Записываем файлы списков
    train_txt = os.path.join(dataset_path, "train.txt")
    target_txt = os.path.join(dataset_path, "target.txt")
    valid_txt = os.path.join(dataset_path, "valid.txt")
    
    with open(train_txt, 'w', encoding='utf-8') as f:
        for path in train_files:
            f.write(path + '\n')
    
    with open(target_txt, 'w', encoding='utf-8') as f:
        for path in test_files:
            f.write(path + '\n')
    
    with open(valid_txt, 'w', encoding='utf-8') as f:
        for path in val_files:
            f.write(path + '\n')
    
    print(color_text(f"✅ Файлы списков созданы:", "GREEN"))
    print(color_text(f"   - {train_txt}", "GREEN"))
    print(color_text(f"   - {target_txt}", "GREEN"))
    print(color_text(f"   - {valid_txt}", "GREEN"))
    
    return train_txt, target_txt, valid_txt

def update_data_yaml(dataset_path, train_file, test_file, val_file):
    """
    Обновляет data.yaml файл с указанием путей к train.txt, target.txt, valid.txt.
    """
    yaml_path = os.path.join(dataset_path, "data.yaml")
    
    # Загружаем существующий YAML
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    if data is None:
        data = {}
    
    # Обновляем пути (относительные от датасета)
    data['train'] = os.path.relpath(train_file, dataset_path)
    data['val'] = os.path.relpath(val_file, dataset_path)
    # Для тестовой выборки используем ключ 'test' (YOLO ожидает 'test' для оценки)
    data['test'] = os.path.relpath(test_file, dataset_path)
    
    # Сохраняем обновленный YAML
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    print(color_text(f"✅ data.yaml обновлен с новыми путями", "GREEN"))
    
    # Также создаем backup
    backup_path = yaml_path + '.backup'
    shutil.copy2(yaml_path, backup_path)
    print(color_text(f"   (Создан backup: {backup_path})", "CYAN"))

def ask_split_option(dataset_path):
    """
    Спрашивает пользователя, использовать существующие файлы разделения или создать новые.
    Возвращает True, если нужно создать новые файлы.
    """
    train_txt = os.path.join(dataset_path, "train.txt")
    target_txt = os.path.join(dataset_path, "target.txt")
    valid_txt = os.path.join(dataset_path, "valid.txt")
    
    existing = all(os.path.exists(f) for f in [train_txt, target_txt, valid_txt])
    
    if existing:
        print(color_text("📁 Найдены существующие файлы разделения данных:", "CYAN"))
        print(f"   - train.txt ({os.path.getsize(train_txt)} байт)")
        print(f"   - target.txt ({os.path.getsize(target_txt)} байт)")
        print(f"   - valid.txt ({os.path.getsize(valid_txt)} байт)")
        
        choice = input(color_text("\nИспользовать существующие файлы? (y/n): ", "YELLOW"))
        if choice.lower() == 'y':
            print(color_text("✅ Используем существующие файлы разделения.", "GREEN"))
            return False
        else:
            print(color_text("🔄 Создадим новые файлы разделения.", "CYAN"))
            return True
    else:
        print(color_text("📁 Файлы разделения не найдены. Создаем новые.", "CYAN"))
        return True

def main():
    """Главная функция."""
    print_header()
    
    # 1. Выбор датасета
    print(color_text("📁 ВЫБОР ДАТАСЕТА:", "BOLD"))
    
    dataset_options = [
        "Аугментированный датасет (datasets/camera_dataset_augmented)",
        "Неаугментированный датасет (datasets/camera_dataset)"
    ]
    
    dataset_choice = get_user_choice("Выберите датасет для обучения:", dataset_options)
    
    if dataset_choice is None:
        print(color_text("Обучение отменено.", "YELLOW"))
        return
    
    if dataset_choice == 0:
        dataset_path = "datasets/camera_dataset_augmented"
        dataset_name = "аугментированный"
    else:
        dataset_path = "datasets/camera_dataset"
        dataset_name = "неаугментированный"
    
    # Проверяем существование выбранного датасета
    if not check_dataset_exists(dataset_path):
        print(color_text("❌ Не удалось найти датасет. Обучение отменено.", "RED"))
        return
    
    data_yaml = os.path.join(dataset_path, "data.yaml")
    print(color_text(f"✅ Используется data.yaml: {data_yaml}", "GREEN"))
    
    # 2. Разделение данных
    print(color_text("\n📊 РАЗДЕЛЕНИЕ ДАННЫХ:", "BOLD"))
    
    need_split = ask_split_option(dataset_path)
    if need_split:
        print(color_text("\n🔄 Создание нового разделения данных 70/15/15...", "CYAN"))
        try:
            train_txt, target_txt, valid_txt = split_dataset(dataset_path)
            update_data_yaml(dataset_path, train_txt, target_txt, valid_txt)
            print(color_text("✅ Разделение данных завершено.", "GREEN"))
        except Exception as e:
            print(color_text(f"❌ Ошибка при разделении данных: {e}", "RED"))
            print(color_text("Продолжаем с существующими файлами (если есть).", "YELLOW"))
    else:
        print(color_text("✅ Используем существующие файлы разделения.", "GREEN"))
    
    # 3. Подготовка папки для сохранения моделей
    target_folder = prepare_targeted_yolo_folder()
    
    # 3. Выбор предобученной модели
    print(color_text("\n🤖 ВЫБОР ПРЕДОБУЧЕННОЙ МОДЕЛИ:", "BOLD"))
    
    model_options = [
        "YOLO26 Nano (yolo26n.pt) - быстрая, легкая",
        "YOLOv8 Nano (yolov8n.pt) - баланс скорости и точности",
        "Другая модель (указать путь вручную)"
    ]
    
    model_choice = get_user_choice("Выберите предобученную модель:", model_options)
    
    if model_choice is None:
        print(color_text("Обучение отменено.", "YELLOW"))
        return
    
    if model_choice == 0:
        model_path = "yolo26n.pt"
    elif model_choice == 1:
        model_path = "yolov8n.pt"
    else:
        model_path = input(color_text("Введите путь к предобученной модели: ", "BLUE"))
    
    # Проверяем существование модели
    if not os.path.exists(model_path):
        print(color_text(f"❌ Предобученная модель не найдена: {model_path}", "RED"))
        print(color_text("Проверьте наличие файла или используйте другую модель.", "YELLOW"))
        return
    
    print(color_text(f"✅ Используется модель: {model_path}", "GREEN"))
    
    # 4. Настройка параметров обучения
    epochs, imgsz, batch = get_training_parameters()
    
    # 5. Имя для сохранения модели
    model_name = input(color_text("Имя модели для сохранения (по умолчанию 'targeted_yolo_model'): ", "BLUE"))
    if model_name == "":
        model_name = "targeted_yolo_model"
    
    # 6. Подтверждение параметров
    print(color_text("\n✅ ПОДТВЕРЖДЕНИЕ ПАРАМЕТРОВ:", "BOLD"))
    print(f"  Датасет: {dataset_path} ({dataset_name})")
    print(f"  Предобученная модель: {model_path}")
    print(f"  Параметры обучения:")
    print(f"    - Эпохи: {epochs}")
    print(f"    - Размер изображения: {imgsz}")
    print(f"    - Размер батча: {batch}")
    print(f"  Сохранение в: {target_folder}/{model_name}")
    
    confirm = input(color_text("\nНачать обучение? (y/n): ", "YELLOW"))
    if confirm.lower() != 'y':
        print(color_text("Обучение отменено.", "YELLOW"))
        return
    
    # 7. Загрузка и обучение модели
    print(color_text("\n🚀 ЗАГРУЗКА МОДЕЛИ И НАЧАЛО ОБУЧЕНИЯ...", "BOLD"))
    
    try:
        # Загрузка предобученной модели
        model = YOLO(model_path)
        print(color_text(f"✅ Модель загружена: {model_path}", "GREEN"))
        
        # Обучение модели
        print(color_text(f"📊 Начало обучения на {dataset_name} датасете...", "CYAN"))
        
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=model_name,
            project=target_folder,
            exist_ok=True,  # Перезаписываем если существует
            save=True,
            save_period=10,  # Сохраняем каждые 10 эпох
            verbose=True
        )
        
        print(color_text("\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!", "GREEN"))
        
        # 8. Копирование лучшей модели в целевую папку
        print(color_text("\n📁 КОПИРОВАНИЕ РЕЗУЛЬТАТОВ...", "CYAN"))
        
        # Пути к результатам обучения
        runs_path = os.path.join(target_folder, model_name)
        best_model_path = os.path.join(runs_path, "weights", "best.pt")
        last_model_path = os.path.join(runs_path, "weights", "last.pt")
        
        # Копируем лучшую модель в корень targeted_yolo
        if os.path.exists(best_model_path):
            target_best_path = os.path.join(target_folder, f"{model_name}_best.pt")
            shutil.copy2(best_model_path, target_best_path)
            print(color_text(f"✅ Лучшая модель сохранена: {target_best_path}", "GREEN"))
        
        # Копируем последнюю модель в корень targeted_yolo
        if os.path.exists(last_model_path):
            target_last_path = os.path.join(target_folder, f"{model_name}_last.pt")
            shutil.copy2(last_model_path, target_last_path)
            print(color_text(f"✅ Последняя модель сохранена: {target_last_path}", "GREEN"))
        
        # 9. Вывод статистики
        print(color_text("\n📈 СТАТИСТИКА ОБУЧЕНИЯ:", "BOLD"))
        print(f"  Датасет: {dataset_name}")
        print(f"  Эпохи: {epochs}")
        print(f"  Результаты сохранены в: {runs_path}")
        print(f"  Модели доступны в: {target_folder}/")
        
        # Проверяем размеры сохраненных моделей
        print(color_text("\n📊 РАЗМЕРЫ СОХРАНЕННЫХ МОДЕЛЕЙ:", "CYAN"))
        for model_file in [f"{model_name}_best.pt", f"{model_name}_last.pt"]:
            model_path = os.path.join(target_folder, model_file)
            if os.path.exists(model_path):
                size = os.path.getsize(model_path) / (1024*1024)
                print(f"  {model_file}: {size:.1f} MB")
        
        print(color_text("\n🎯 Обучение успешно завершено! Модели готовы к использованию.", "GREEN"))
        
    except Exception as e:
        print(color_text(f"\n❌ ОШИБКА ПРИ ОБУЧЕНИИ: {e}", "RED"))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(color_text("\n\nОбучение прервано пользователем.", "YELLOW"))
    except Exception as e:
        print(color_text(f"\nКритическая ошибка: {e}", "RED"))