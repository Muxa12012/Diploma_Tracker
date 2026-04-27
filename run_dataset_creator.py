#!/usr/bin/env python3
"""
Скрипт для запуска одного из четырёх файлов создания датасета.
Автоматически определяет доступные файлы и предоставляет меню выбора.
"""

import subprocess
import sys
import os
import fnmatch
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

# Целевые папки датасетов для операций удаления/переименования классов
TARGET_DATASET_PATHS = [
    "datasets/camera_dataset",
    "datasets/camera_dataset_augmented",
]

def color_text(text, color):
    """Возвращает текст с ANSI кодами цвета."""
    return f"{COLORS.get(color, '')}{text}{COLORS['ENDC']}"

def print_header():
    """Выводит заголовок меню."""
    print(color_text("\n" + "=" * 60, "HEADER"))
    print(color_text("   МЕНЮ СОЗДАНИЯ ДАТАСЕТА ДЛЯ YOLO", "BOLD"))
    print(color_text("=" * 60, "HEADER"))
    print()

def print_menu(options):
    """Выводит пронумерованное меню с доступными опциями."""
    for i, (filename, desc) in enumerate(options, start=1):
        print(f"  {color_text(str(i), 'CYAN')}. {color_text(filename, 'GREEN')}")
        print(f"     {desc}")
        print()

def get_user_choice(max_option):
    """Запрашивает у пользователя выбор и возвращает номер."""
    while True:
        try:
            choice = input(color_text("Введите номер (1-4) или 'q' для выхода: ", "YELLOW")).strip().lower()
            if choice == 'q':
                return None
            choice_num = int(choice)
            if 1 <= choice_num <= max_option:
                return choice_num
            else:
                print(color_text(f"Ошибка: номер должен быть от 1 до {max_option}. Попробуйте снова.", "RED"))
        except ValueError:
            print(color_text("Ошибка: введите корректное число или 'q'.", "RED"))

def run_script(filename):
    """Запускает выбранный скрипт с помощью subprocess."""
    if not os.path.exists(filename):
        print(color_text(f"Ошибка: файл '{filename}' не найден в текущей директории.", "RED"))
        return False
    
    print(color_text(f"\nЗапуск {filename}...", "BLUE"))
    print(color_text("Для выхода из скрипта нажмите Ctrl+C в его окне.", "YELLOW"))
    print("-" * 40)
    
    try:
        # Запускаем скрипт с текущим интерпретатором Python
        result = subprocess.run([sys.executable, filename], check=False)
        if result.returncode == 0:
            print(color_text(f"\nСкрипт {filename} завершился успешно.", "GREEN"))
        else:
            print(color_text(f"\nСкрипт {filename} завершился с кодом возврата {result.returncode}.", "YELLOW"))
        return True
    except FileNotFoundError:
        print(color_text(f"Ошибка: не удалось запустить Python или файл {filename}.", "RED"))
        return False
    except KeyboardInterrupt:
        print(color_text("\nЗапуск прерван пользователем.", "YELLOW"))
        return False
    except Exception as e:
        print(color_text(f"Неожиданная ошибка при запуске: {e}", "RED"))
        return False


def run_augmentation():
    """Запускает скрипт аугментации изображений."""
    # Определяем возможные скрипты аугментации в порядке предпочтения
    augmentation_scripts = [
        "augmentations/run_augmentation.py",
        "augmentations/main.py",
        "augmentations/augmentation_pipeline.py"
    ]
    
    selected_script = None
    for script in augmentation_scripts:
        if os.path.exists(script):
            selected_script = script
            break
    
    if selected_script is None:
        print(color_text("Ошибка: не найден ни один скрипт аугментации в папке augmentations.", "RED"))
        print(color_text("Убедитесь, что папка augmentations содержит run_augmentation.py или main.py.", "YELLOW"))
        return False
    
    print(color_text(f"\nЗапуск аугментации изображений ({selected_script})...", "BLUE"))
    print(color_text("Для выхода из скрипта нажмите Ctrl+C в его окне.", "YELLOW"))
    print("-" * 40)
    
    try:
        result = subprocess.run([sys.executable, selected_script], check=False)
        if result.returncode == 0:
            print(color_text("\nАугментация изображений завершена успешно.", "GREEN"))
        else:
            print(color_text(f"\nАугментация завершилась с кодом возврата {result.returncode}.", "YELLOW"))
        return True
    except FileNotFoundError:
        print(color_text(f"Ошибка: не удалось запустить Python или файл {selected_script}.", "RED"))
        return False
    except KeyboardInterrupt:
        print(color_text("\nЗапуск аугментации прерван пользователем.", "YELLOW"))
        return False
    except Exception as e:
        print(color_text(f"Неожиданная ошибка при запуске аугментации: {e}", "RED"))
        return False


# ============================================================================
# Функции для удаления класса изображений
# ============================================================================

def find_data_yaml(start_path="."):
    """
    Ищет файл data.yaml только в целевых папках датасетов.
    Возвращает путь к первому найденному файлу или None.
    """
    start_path = Path(start_path)
    target_paths = [Path(p) for p in TARGET_DATASET_PATHS]
    
    for target_path in target_paths:
        # Проверяем существование папки
        if not target_path.exists():
            continue
        # Ищем data.yaml в этой папке (не рекурсивно, только на верхнем уровне)
        yaml_file = target_path / "data.yaml"
        if yaml_file.exists():
            return yaml_file
        # Также можно проверить рекурсивно внутри папки, если требуется
        for root, dirs, files in os.walk(target_path):
            if "data.yaml" in files:
                return Path(root) / "data.yaml"
    
    # Если ни в одной целевой папке не найден
    print(color_text(f"Файл data.yaml не найден в целевых папках: {TARGET_DATASET_PATHS}", "YELLOW"))
    return None


def load_data_yaml(yaml_path):
    """
    Загружает data.yaml и возвращает словарь с данными.
    """
    try:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        # Автоматическая дедупликация классов при загрузке
        data = deduplicate_classes(data)
        return data
    except ImportError:
        print(color_text("Ошибка: для работы с YAML требуется установить PyYAML.", "RED"))
        print(color_text("Установите: pip install pyyaml", "YELLOW"))
        return None
    except Exception as e:
        print(color_text(f"Ошибка загрузки {yaml_path}: {e}", "RED"))
        return None


def deduplicate_classes(data):
    """
    Удаляет дубликаты из списка классов в data.yaml и обновляет nc.
    Возвращает обновленные данные.
    """
    if 'names' not in data:
        return data
    
    original_names = data['names']
    # Сохраняем порядок первого вхождения
    seen = set()
    unique_names = []
    for name in original_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)
    
    if len(unique_names) == len(original_names):
        # Дубликатов нет
        return data
    
    data['names'] = unique_names
    data['nc'] = len(unique_names)
    return data


def save_data_yaml(yaml_path, data):
    """
    Сохраняет данные обратно в data.yaml с созданием резервной копии.
    """
    import yaml
    # Создаем backup
    backup_path = yaml_path.with_suffix('.yaml.backup')
    try:
        import shutil
        shutil.copy2(yaml_path, backup_path)
        print(color_text(f"Создан backup: {backup_path}", "CYAN"))
    except Exception as e:
        print(color_text(f"Не удалось создать backup: {e}", "YELLOW"))
    
    try:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(color_text(f"Файл {yaml_path} обновлён.", "GREEN"))
        return True
    except Exception as e:
        print(color_text(f"Ошибка сохранения {yaml_path}: {e}", "RED"))
        return False


def verify_class_in_annotation(file_path, class_index):
    """
    Проверяет, содержит ли файл аннотации YOLO указанный индекс класса.
    Возвращает True, если содержит, иначе False.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 1:
                    try:
                        if int(parts[0]) == class_index:
                            return True
                    except ValueError:
                        continue
        return False
    except Exception:
        return False


def find_all_annotation_files(search_root="."):
    """
    Находит все файлы аннотаций YOLO (.txt) в проекте рекурсивно.
    Исключает файлы с именами train.txt, valid.txt, test.txt, data.yaml и другие не-аннотации.
    Возвращает список путей Path.
    """
    # Если search_root равен "." (по умолчанию), ограничиваем поиск целевыми папками датасетов
    if search_root == ".":
        target_paths = [Path(p) for p in TARGET_DATASET_PATHS]
        existing_paths = [p for p in target_paths if p.exists()]
        if not existing_paths:
            print(color_text(f"Предупреждение: целевые папки не существуют: {TARGET_DATASET_PATHS}", "YELLOW"))
            return []
        
        annotation_files = []
        exclude_names = {'train.txt', 'valid.txt', 'test.txt', 'data.yaml', 'data.yaml.backup'}
        
        for target_path in existing_paths:
            for root, dirs, files in os.walk(target_path):
                for file in files:
                    if file.lower().endswith('.txt'):
                        # Пропускаем временные и системные файлы
                        if file.startswith('.') or file.startswith('~'):
                            continue
                        # Пропускаем исключенные имена
                        if file in exclude_names:
                            continue
                        file_path = Path(root) / file
                        annotation_files.append(file_path)
        return annotation_files
    else:
        # Если передан явный search_root, используем его (для обратной совместимости)
        search_root = Path(search_root)
        annotation_files = []
        exclude_names = {'train.txt', 'valid.txt', 'test.txt', 'data.yaml', 'data.yaml.backup'}
        
        for root, dirs, files in os.walk(search_root):
            for file in files:
                if file.lower().endswith('.txt'):
                    if file.startswith('.') or file.startswith('~'):
                        continue
                    if file in exclude_names:
                        continue
                    file_path = Path(root) / file
                    annotation_files.append(file_path)
        return annotation_files


def fix_annotation_labels(annotation_files, deleted_class_index, new_class_count, backup=True, preview=False):
    """
    Исправляет метки в аннотациях YOLO после удаления класса.
    
    Параметры:
    - annotation_files: список путей к файлам аннотаций
    - deleted_class_index: индекс удаляемого класса
    - new_class_count: новое количество классов (после удаления)
    - backup: создавать резервные копии файлов перед изменением
    - preview: только предварительный просмотр без реальных изменений
    
    Возвращает словарь со статистикой:
    {
        'total_files': общее количество обработанных файлов,
        'files_modified': количество измененных файлов,
        'labels_deleted': количество удаленных меток (класс deleted_class_index),
        'labels_shifted': количество сдвинутых меток (class_id > deleted_class_index),
        'labels_corrected': количество исправленных меток (class_id >= new_class_count),
        'errors': список ошибок
    }
    """
    import shutil
    from pathlib import Path
    
    stats = {
        'total_files': len(annotation_files),
        'files_modified': 0,
        'labels_deleted': 0,
        'labels_shifted': 0,
        'labels_corrected': 0,
        'errors': []
    }
    
    for file_path in annotation_files:
        file_path = Path(file_path)
        try:
            # Читаем содержимое файла
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            new_lines = []
            file_modified = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    # Сохраняем пустые строки (хотя в YOLO их обычно нет)
                    new_lines.append(line)
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    # Некорректная строка, оставляем как есть
                    new_lines.append(line)
                    continue
                
                try:
                    class_id = int(parts[0])
                except ValueError:
                    # Не число, оставляем как есть
                    new_lines.append(line)
                    continue
                
                # Проверяем условия
                if class_id == deleted_class_index:
                    # Удаляем метку (не добавляем в новые строки)
                    stats['labels_deleted'] += 1
                    file_modified = True
                    continue
                
                new_class_id = class_id
                if class_id > deleted_class_index:
                    # Сдвигаем классы с индексами больше удаленного
                    new_class_id = class_id - 1
                    stats['labels_shifted'] += 1
                    file_modified = True
                
                if new_class_id >= new_class_count:
                    # Класс выходит за пределы нового количества классов
                    # Исправляем на максимально допустимый индекс (new_class_count - 1)
                    # или удаляем? Выбираем исправление на последний допустимый класс.
                    new_class_id = new_class_count - 1
                    stats['labels_corrected'] += 1
                    file_modified = True
                
                # Формируем новую строку
                new_line = f"{new_class_id} {' '.join(parts[1:])}"
                new_lines.append(new_line)
            
            if file_modified:
                stats['files_modified'] += 1
                
                if not preview:
                    # Создаем backup если требуется
                    if backup:
                        backup_path = file_path.with_suffix('.txt.backup')
                        try:
                            shutil.copy2(file_path, backup_path)
                        except Exception as e:
                            stats['errors'].append(f"{file_path}: не удалось создать backup: {e}")
                    
                    # Записываем изменения
                    with open(file_path, 'w', encoding='utf-8') as f:
                        for line in new_lines:
                            f.write(line + '\n')
                else:
                    # В режиме предварительного просмотра только логируем
                    pass
        
        except Exception as e:
            stats['errors'].append(f"{file_path}: {e}")
    
    return stats


def find_files_by_class(class_name, search_root=".", class_index=None, strict=True):
    """
    Ищет все файлы изображений и аннотаций, содержащие название класса в имени.
    Использует строгие паттерны: класс должен быть в начале имени с разделителем _ или -.
    Если передан class_index, то для файлов аннотаций (.txt) проверяется содержимое.
    Параметр strict: если True, используется только строгие паттерны; если False, добавляется широкий поиск.
    Возвращает список путей к файлам.
    """
    # Если search_root равен "." (по умолчанию), ограничиваем поиск целевыми папками датасетов
    if search_root == ".":
        target_paths = [Path(p) for p in TARGET_DATASET_PATHS]
        # Удаляем несуществующие папки
        existing_paths = [p for p in target_paths if p.exists()]
        if not existing_paths:
            print(color_text(f"Предупреждение: целевые папки не существуют: {TARGET_DATASET_PATHS}", "YELLOW"))
            return []
        
        found_files = []
        # Паттерны для поиска: класс должен быть в начале имени
        patterns = [
            f"{class_name}_*",
            f"{class_name}-*",
        ]
        if not strict:
            patterns.append(f"*{class_name}*")  # более широкий поиск для обратной совместимости
        
        for target_path in existing_paths:
            for root, dirs, files in os.walk(target_path):
                for file in files:
                    # Пропускаем временные и системные файлы
                    if file.startswith('.') or file.startswith('~'):
                        continue
                    # Проверяем соответствие паттернам
                    matched = False
                    for pattern in patterns:
                        if fnmatch.fnmatch(file, pattern):
                            matched = True
                            break
                    # Также ищем файлы с суффиксом _aug (аугментации)
                    if '_aug' in file and class_name in file:
                        matched = True
                    
                    if matched:
                        file_path = Path(root) / file
                        # Если передан class_index и файл является аннотацией (.txt), проверяем содержимое
                        if class_index is not None and file_path.suffix.lower() == '.txt':
                            if not verify_class_in_annotation(file_path, class_index):
                                # Пропускаем файл, так как он не содержит указанный класс
                                continue
                        found_files.append(file_path)
        return found_files
    else:
        # Если передан явный search_root, используем его (для обратной совместимости)
        search_root = Path(search_root)
        found_files = []
        
        patterns = [
            f"{class_name}_*",
            f"{class_name}-*",
        ]
        if not strict:
            patterns.append(f"*{class_name}*")
        
        for root, dirs, files in os.walk(search_root):
            for file in files:
                if file.startswith('.') or file.startswith('~'):
                    continue
                matched = False
                for pattern in patterns:
                    if fnmatch.fnmatch(file, pattern):
                        matched = True
                        break
                if '_aug' in file and class_name in file:
                    matched = True
                
                if matched:
                    file_path = Path(root) / file
                    if class_index is not None and file_path.suffix.lower() == '.txt':
                        if not verify_class_in_annotation(file_path, class_index):
                            continue
                    found_files.append(file_path)
        return found_files


def log_deletion(class_name, files, deleted_count, errors, yaml_path, new_classes):
    """
    Создает лог-файл с информацией об удалении класса.
    """
    import datetime
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"deletion_{class_name}_{timestamp}.log"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Лог удаления класса '{class_name}'\n")
        f.write(f"Время: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Файл конфигурации: {yaml_path}\n")
        f.write(f"Исходное количество классов: {len(new_classes) + 1}\n")
        f.write(f"Новое количество классов: {len(new_classes)}\n")
        f.write(f"Всего файлов для удаления: {len(files)}\n")
        f.write(f"Успешно удалено: {deleted_count}\n")
        f.write(f"Ошибок: {len(errors)}\n")
        f.write("\n" + "="*60 + "\n")
        
        if files:
            f.write("\nСписок файлов для удаления:\n")
            for file_path in files:
                f.write(f"  {file_path}\n")
        
        if errors:
            f.write("\nОшибки удаления:\n")
            for error in errors:
                f.write(f"  {error}\n")
        
        f.write("\nНовый список классов:\n")
        for idx, cls in enumerate(new_classes):
            f.write(f"  {idx}. {cls}\n")
    
    print(color_text(f"\nЛог удаления сохранён в: {log_file}", "CYAN"))
    return log_file


def delete_class_files(files, log_errors=None):
    """
    Удаляет список файлов с подтверждением.
    Возвращает количество успешно удалённых файлов и список ошибок.
    """
    deleted_count = 0
    errors = []
    for file_path in files:
        try:
            # Получаем размер файла перед удалением (для лога)
            size = file_path.stat().st_size if file_path.exists() else 0
            file_path.unlink()
            print(color_text(f"Удалён: {file_path} ({size} байт)", "RED"))
            deleted_count += 1
        except Exception as e:
            error_msg = f"{file_path}: {e}"
            print(color_text(f"Ошибка удаления {file_path}: {e}", "YELLOW"))
            errors.append(error_msg)
            if log_errors is not None:
                log_errors.append(error_msg)
    return deleted_count, errors


def show_class_statistics():
    """
    Показывает статистику по классам в датасете.
    """
    print(color_text("\n" + "=" * 60, "HEADER"))
    print(color_text("   СТАТИСТИКА КЛАССОВ ДАТАСЕТА", "BOLD"))
    print(color_text("=" * 60, "HEADER"))
    print()
    
    # 1. Найти data.yaml
    yaml_path = find_data_yaml()
    if yaml_path is None:
        print(color_text("Ошибка: файл data.yaml не найден в проекте.", "RED"))
        print(color_text("Убедитесь, что вы находитесь в директории с датасетом YOLO.", "YELLOW"))
        input(color_text("\nНажмите Enter чтобы вернуться в меню...", "CYAN"))
        return

    print(color_text(f"Найден файл конфигурации: {yaml_path}", "GREEN"))
    
    # Определяем, с каким датасетом работаем
    dataset_name = "неизвестный датасет"
    for target_path in TARGET_DATASET_PATHS:
        if str(yaml_path).startswith(target_path):
            dataset_name = target_path
            break
    print(color_text(f"Работаем с датасетом: {dataset_name}", "CYAN"))
    print(color_text(f"Операции ограничены целевыми папками: {TARGET_DATASET_PATHS}", "CYAN"))
    
    # 2. Загрузить данные
    data = load_data_yaml(yaml_path)
    if data is None:
        input(color_text("\nНажмите Enter чтобы вернуться в меню...", "CYAN"))
        return
    
    # 3. Показать существующие классы
    if 'names' not in data:
        print(color_text("Ошибка: в data.yaml отсутствует ключ 'names'.", "RED"))
        input(color_text("\nНажмите Enter чтобы вернуться в меню...", "CYAN"))
        return
    
    classes = data['names']
    if not classes:
        print(color_text("Список классов пуст.", "YELLOW"))
        input(color_text("\nНажмите Enter чтобы вернуться в меню...", "CYAN"))
        return
    
    print(color_text(f"Всего классов: {len(classes)}", "CYAN"))
    print(color_text("Сбор статистики...", "BLUE"))
    
    # 4. Сбор статистики для каждого класса
    stats = []
    total_images = 0
    total_annotations = 0
    total_size_bytes = 0
    
    for idx, class_name in enumerate(classes):
        # Поиск файлов, связанных с классом
        found_files = find_files_by_class(class_name, strict=False)
        
        # Разделение на изображения и аннотации
        image_files = [f for f in found_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
        annotation_files = [f for f in found_files if f.suffix.lower() == '.txt']
        
        # Подсчет размера файлов изображений
        size_bytes = 0
        for img_file in image_files:
            try:
                size_bytes += img_file.stat().st_size
            except Exception:
                pass
        
        stats.append({
            'index': idx,
            'name': class_name,
            'image_count': len(image_files),
            'annotation_count': len(annotation_files),
            'total_files': len(found_files),
            'size_bytes': size_bytes
        })
        
        total_images += len(image_files)
        total_annotations += len(annotation_files)
        total_size_bytes += size_bytes
    
    # 5. Вывод таблицы
    print(color_text("\n" + "-" * 80, "CYAN"))
    print(color_text(f"{'Класс':<20} {'Индекс':<8} {'Изображения':<12} {'Аннотации':<12} {'Всего файлов':<14} {'Размер':<12}", "BOLD"))
    print(color_text("-" * 80, "CYAN"))
    
    for stat in stats:
        size_str = f"{stat['size_bytes'] / 1024:.1f} KB" if stat['size_bytes'] > 0 else "0 KB"
        print(f"{stat['name']:<20} {stat['index']:<8} {stat['image_count']:<12} {stat['annotation_count']:<12} {stat['total_files']:<14} {size_str:<12}")
    
    print(color_text("-" * 80, "CYAN"))
    
    # 6. Общая статистика
    print(color_text("\nОБЩАЯ СТАТИСТИКА ДАТАСЕТА:", "BOLD"))
    print(f"  Всего классов: {len(classes)}")
    print(f"  Всего изображений: {total_images}")
    print(f"  Всего аннотаций: {total_annotations}")
    print(f"  Общий размер изображений: {total_size_bytes / (1024*1024):.2f} MB")
    
    # 7. Проверка баланса классов
    if total_images > 0:
        print(color_text("\nБАЛАНС КЛАССОВ (доля изображений):", "BOLD"))
        for stat in stats:
            percentage = (stat['image_count'] / total_images * 100) if total_images > 0 else 0
            bar_length = int(percentage / 2)  # 50% = 25 символов
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  {stat['name']:<15} {percentage:>5.1f}% {bar}")
    
    input(color_text("\nНажмите Enter чтобы вернуться в меню...", "CYAN"))


def rename_class():
    """
    Переименовывает класс в датасете.
    """
    print(color_text("\n" + "=" * 60, "HEADER"))
    print(color_text("   ПЕРЕИМЕНОВАНИЕ КЛАССА", "BOLD"))
    print(color_text("=" * 60, "HEADER"))
    print()
    
    # 1. Найти data.yaml
    yaml_path = find_data_yaml()
    if yaml_path is None:
        print(color_text("Ошибка: файл data.yaml не найден в проекте.", "RED"))
        print(color_text("Убедитесь, что вы находитесь в директории с датасетом YOLO.", "YELLOW"))
        input(color_text("\nНажмите Enter чтобы вернуться в меню...", "CYAN"))
        return
    
    print(color_text(f"Найден файл конфигурации: {yaml_path}", "GREEN"))
    
    # Определяем, с каким датасетом работаем
    dataset_name = "неизвестный датасет"
    for target_path in TARGET_DATASET_PATHS:
        if str(yaml_path).startswith(target_path):
            dataset_name = target_path
            break
    print(color_text(f"Работаем с датасетом: {dataset_name}", "CYAN"))
    print(color_text(f"Операции ограничены целевыми папками: {TARGET_DATASET_PATHS}", "CYAN"))

    # 2. Загрузить данные
    data = load_data_yaml(yaml_path)
    if data is None:
        input(color_text("\nНажмите Enter чтобы вернуться в меню...", "CYAN"))
        return
    
    # 3. Показать существующие классы
    if 'names' not in data:
        print(color_text("Ошибка: в data.yaml отсутствует ключ 'names'.", "RED"))
        input(color_text("\nНажмите Enter чтобы вернуться в меню...", "CYAN"))
        return
    
    classes = data['names']
    if not classes:
        print(color_text("Список классов пуст.", "YELLOW"))
        input(color_text("\nНажмите Enter чтобы вернуться в меню...", "CYAN"))
        return
    
    print(color_text("\nСуществующие классы:", "CYAN"))
    for idx, cls in enumerate(classes):
        print(f"  {color_text(str(idx), 'YELLOW')}. {cls}")
    
    # 4. Запросить номер класса для переименования
    while True:
        try:
            choice = input(color_text("\nВведите номер класса для переименования (0-{}): ".format(len(classes)-1), "YELLOW")).strip()
            if choice.lower() == 'q':
                print(color_text("Отмена операции.", "YELLOW"))
                return
            class_idx = int(choice)
            if 0 <= class_idx < len(classes):
                break
            else:
                print(color_text(f"Ошибка: номер должен быть от 0 до {len(classes)-1}.", "RED"))
        except ValueError:
            print(color_text("Ошибка: введите корректное число.", "RED"))
    
    old_name = classes[class_idx]
    print(color_text(f"\nВыбран класс: {old_name}", "BOLD"))
    
    # 5. Запросить новое название
    while True:
        new_name = input(color_text("Введите новое название класса: ", "YELLOW")).strip()
        if not new_name:
            print(color_text("Название не может быть пустым.", "RED"))
            continue
        if new_name in classes:
            print(color_text(f"Класс с названием '{new_name}' уже существует. Выберите другое название.", "RED"))
            continue
        if ' ' in new_name:
            print(color_text("Рекомендуется не использовать пробелы в названиях классов. Продолжить? (y/n): ", "YELLOW"))
            confirm = input().strip().lower()
            if confirm != 'y':
                continue
        break
    
    print(color_text(f"\nПереименование: {old_name} -> {new_name}", "BOLD"))
    
    # 6. Поиск файлов, связанных с классом
    print(color_text("\nПоиск файлов...", "BLUE"))
    found_files = find_files_by_class(old_name, strict=False)
    
    if not found_files:
        print(color_text("Файлы с указанным классом не найдены.", "YELLOW"))
        print(color_text("Будет обновлено только название класса в data.yaml.", "CYAN"))
        files_to_rename = []
    else:
        print(color_text(f"Найдено файлов: {len(found_files)}", "CYAN"))
        # Показать первые 10 файлов
        for i, f in enumerate(found_files[:10]):
            print(f"  {f}")
        if len(found_files) > 10:
            print(f"  ... и ещё {len(found_files) - 10} файлов.")
        
        # 7. Предварительный просмотр изменений
        print(color_text("\nПРЕДВАРИТЕЛЬНЫЙ ПРОСМОТР:", "BOLD"))
        for f in found_files[:5]:
            new_name_in_file = str(f).replace(old_name, new_name)
            print(f"  {f.name} -> {Path(new_name_in_file).name}")
        if len(found_files) > 5:
            print(f"  ... и ещё {len(found_files) - 5} файлов.")
        
        # 8. Запросить подтверждение
        confirm = input(color_text(f"\nПереименовать {len(found_files)} файлов? (y/n): ", "RED")).strip().lower()
        if confirm != 'y':
            print(color_text("Отмена операции.", "YELLOW"))
            return
        files_to_rename = found_files
    
    # 9. Создание резервной копии data.yaml
    import shutil
    backup_path = yaml_path.with_suffix('.yaml.backup_rename')
    try:
        shutil.copy2(yaml_path, backup_path)
        print(color_text(f"Создан backup: {backup_path}", "CYAN"))
    except Exception as e:
        print(color_text(f"Не удалось создать backup: {e}", "YELLOW"))
    
    # 10. Переименование файлов
    renamed_count = 0
    rename_errors = []
    
    for file_path in files_to_rename:
        try:
            # Новое имя файла
            new_file_name = file_path.name.replace(old_name, new_name)
            new_file_path = file_path.parent / new_file_name
            
            # Проверяем, не существует ли уже файл с таким именем
            if new_file_path.exists():
                print(color_text(f"Предупреждение: файл {new_file_path} уже существует. Пропускаем.", "YELLOW"))
                rename_errors.append(f"Файл уже существует: {file_path}")
                continue
            
            # Переименовываем файл
            file_path.rename(new_file_path)
            print(color_text(f"Переименован: {file_path.name} -> {new_file_name}", "GREEN"))
            renamed_count += 1
            
            # Если это файл аннотации (.txt), обновляем содержимое
            if new_file_path.suffix.lower() == '.txt':
                update_annotation_class_index(new_file_path, class_idx, class_idx)  # индекс не меняется, только имя в содержимом не требуется
                # В YOLO аннотации содержат только индекс класса, а не имя, поэтому обновление не требуется
                # Но если в будущем потребуется, можно добавить логику
                
        except Exception as e:
            print(color_text(f"Ошибка переименования {file_path}: {e}", "RED"))
            rename_errors.append(f"{file_path}: {e}")
    
    # 11. Обновление data.yaml
    print(color_text("\nОбновление data.yaml...", "BLUE"))
    classes[class_idx] = new_name
    data['names'] = classes
    # nc не меняется
    
    if save_data_yaml(yaml_path, data):
        print(color_text(f"Класс '{old_name}' успешно переименован в '{new_name}' в конфигурации.", "GREEN"))
    else:
        print(color_text("Ошибка обновления data.yaml.", "RED"))
    
    # 12. Итоги
    print(color_text("\nИТОГИ ПЕРЕИМЕНОВАНИЯ:", "BOLD"))
    print(f"  Переименовано файлов: {renamed_count} из {len(files_to_rename)}")
    if rename_errors:
        print(color_text(f"  Ошибок: {len(rename_errors)}", "YELLOW"))
        for err in rename_errors[:5]:
            print(f"    {err}")
    
    input(color_text("\nНажмите Enter чтобы вернуться в меню...", "CYAN"))


def delete_class():
    """
    Основная функция удаления класса изображений с улучшенным предварительным просмотром и логированием.
    """
    print(color_text("\n" + "=" * 60, "HEADER"))
    print(color_text("   УДАЛЕНИЕ КЛАССА ИЗОБРАЖЕНИЙ", "BOLD"))
    print(color_text("=" * 60, "HEADER"))
    print()
    
    # 1. Найти data.yaml
    yaml_path = find_data_yaml()
    if yaml_path is None:
        print(color_text("Ошибка: файл data.yaml не найден в проекте.", "RED"))
        print(color_text("Убедитесь, что вы находитесь в директории с датасетом YOLO.", "YELLOW"))
        input(color_text("\nНажмите Enter чтобы вернуться в меню...", "CYAN"))
        return
    
    print(color_text(f"Найден файл конфигурации: {yaml_path}", "GREEN"))
    
    # 2. Загрузить данные
    data = load_data_yaml(yaml_path)
    if data is None:
        input(color_text("\nНажмите Enter чтобы вернуться в меню...", "CYAN"))
        return
    
    # 3. Показать существующие классы
    if 'names' not in data:
        print(color_text("Ошибка: в data.yaml отсутствует ключ 'names'.", "RED"))
        input(color_text("\nНажмите Enter чтобы вернуться в меню...", "CYAN"))
        return
    
    classes = data['names']
    if not classes:
        print(color_text("Список классов пуст.", "YELLOW"))
        input(color_text("\nНажмите Enter чтобы вернуться в меню...", "CYAN"))
        return
    
    print(color_text("\nСуществующие классы:", "CYAN"))
    for idx, cls in enumerate(classes):
        print(f"  {color_text(str(idx), 'YELLOW')}. {cls}")
    
    # 4. Запросить номер класса для удаления
    while True:
        try:
            choice = input(color_text("\nВведите номер класса для удаления (0-{}): ".format(len(classes)-1), "YELLOW")).strip()
            if choice.lower() == 'q':
                print(color_text("Отмена операции.", "YELLOW"))
                return
            class_idx = int(choice)
            if 0 <= class_idx < len(classes):
                break
            else:
                print(color_text(f"Ошибка: номер должен быть от 0 до {len(classes)-1}.", "RED"))
        except ValueError:
            print(color_text("Ошибка: введите корректное число.", "RED"))
    
    class_name = classes[class_idx]
    print(color_text(f"\nВыбран класс: {class_name}", "BOLD"))
    print(color_text("ВНИМАНИЕ: операция удалит все файлы изображений и аннотаций, содержащие это название в имени.", "RED"))
    print(color_text("Это включает аугментированные версии файлов (с суффиксом _aug).", "YELLOW"))
    
    # 5. Поиск файлов с улучшенной детализацией
    print(color_text("\nПоиск файлов...", "BLUE"))
    found_files = find_files_by_class(class_name)
    
    if not found_files:
        print(color_text("Файлы с указанным классом не найдены.", "YELLOW"))
        confirm = input(color_text("Всё равно удалить класс из data.yaml? (y/n): ", "YELLOW")).strip().lower()
        if confirm != 'y':
            print(color_text("Отмена операции.", "YELLOW"))
            return
        # Продолжить только с обновлением data.yaml
        files_to_delete = []
        image_files = []
        annotation_files = []
        other_files = []
    else:
        # Анализ типов файлов
        image_files = [f for f in found_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
        annotation_files = [f for f in found_files if f.suffix.lower() == '.txt']
        other_files = [f for f in found_files if f not in image_files and f not in annotation_files]
        
        print(color_text(f"\nНайдено файлов: {len(found_files)}", "CYAN"))
        print(color_text("Распределение по типам:", "CYAN"))
        print(f"  Изображений: {len(image_files)}")
        print(f"  Аннотаций: {len(annotation_files)}")
        print(f"  Прочих файлов: {len(other_files)}")
        
        # Подсчет общего размера изображений
        total_size = 0
        for img_file in image_files:
            try:
                total_size += img_file.stat().st_size
            except Exception:
                pass
        size_mb = total_size / (1024 * 1024)
        print(f"  Общий размер изображений: {size_mb:.2f} MB")
        
        # Показать первые 5 файлов каждого типа
        print(color_text("\nПервые 5 изображений:", "YELLOW"))
        for i, f in enumerate(image_files[:5]):
            size = f.stat().st_size if f.exists() else 0
            print(f"  {f.name} ({size // 1024} KB)")
        if len(image_files) > 5:
            print(f"  ... и ещё {len(image_files) - 5} изображений.")
        
        print(color_text("\nПервые 5 аннотаций:", "YELLOW"))
        for i, f in enumerate(annotation_files[:5]):
            print(f"  {f.name}")
        if len(annotation_files) > 5:
            print(f"  ... и ещё {len(annotation_files) - 5} аннотаций.")
        
        # 6. Запросить подтверждение удаления файлов
        confirm = input(color_text(f"\nУдалить {len(found_files)} файлов? (y/n): ", "RED")).strip().lower()
        if confirm != 'y':
            print(color_text("Отмена операции.", "YELLOW"))
            return
        files_to_delete = found_files
    
    # 7. Удаление файлов с логированием
    if files_to_delete:
        print(color_text("\nНачинаю удаление файлов...", "BLUE"))
        deleted_count, errors = delete_class_files(files_to_delete)
        print(color_text(f"\nУдалено файлов: {deleted_count} из {len(files_to_delete)}", "GREEN"))
        if errors:
            print(color_text(f"Ошибок при удалении: {len(errors)}", "YELLOW"))
            for err in errors[:3]:
                print(f"  {err}")
            if len(errors) > 3:
                print(f"  ... и ещё {len(errors) - 3} ошибок.")
    else:
        print(color_text("Файлы для удаления отсутствуют.", "CYAN"))
        errors = []
    
    # 8. Обновление data.yaml
    print(color_text("\nОбновление data.yaml...", "BLUE"))
    # Удаляем класс из списка
    new_classes = [cls for idx, cls in enumerate(classes) if idx != class_idx]
    data['names'] = new_classes
    data['nc'] = len(new_classes)
    
    if save_data_yaml(yaml_path, data):
        print(color_text(f"Класс '{class_name}' успешно удалён из конфигурации.", "GREEN"))
        print(color_text(f"Обновлено количество классов: nc = {data['nc']}", "CYAN"))
    else:
        print(color_text("Ошибка обновления data.yaml.", "RED"))
    
    # 9. Коррекция меток в аннотациях
    print(color_text("\n" + "=" * 60, "HEADER"))
    print(color_text("   КОРРЕКЦИЯ МЕТОК В АННОТАЦИЯХ", "BOLD"))
    print(color_text("=" * 60, "HEADER"))
    print()
    
    # Поиск всех файлов аннотаций в проекте
    print(color_text("Поиск всех файлов аннотаций...", "BLUE"))
    all_annotation_files = find_all_annotation_files()
    print(color_text(f"Найдено файлов аннотаций: {len(all_annotation_files)}", "CYAN"))
    
    if len(all_annotation_files) == 0:
        print(color_text("Файлы аннотаций не найдены, пропускаем коррекцию меток.", "YELLOW"))
    else:
        # Запрос на предварительный просмотр
        preview = input(color_text("Предварительный просмотр изменений без сохранения? (y/n): ", "YELLOW")).strip().lower() == 'y'
        if preview:
            print(color_text("Режим предварительного просмотра. Изменения не будут сохранены.", "CYAN"))
        
        # Запуск коррекции
        stats = fix_annotation_labels(
            all_annotation_files,
            deleted_class_index=class_idx,
            new_class_count=len(new_classes),
            backup=True,
            preview=preview
        )
        
        # Вывод отчета
        print(color_text("\nОТЧЕТ О КОРРЕКЦИИ МЕТОК:", "BOLD"))
        print(f"  Обработано файлов: {stats['total_files']}")
        print(f"  Изменено файлов: {stats['files_modified']}")
        print(f"  Удалено меток (класс {class_idx}): {stats['labels_deleted']}")
        print(f"  Сдвинуто меток (class_id > {class_idx}): {stats['labels_shifted']}")
        print(f"  Исправлено меток (class_id >= {len(new_classes)}): {stats['labels_corrected']}")
        
        if stats['errors']:
            print(color_text(f"  Ошибок при обработке: {len(stats['errors'])}", "YELLOW"))
            for err in stats['errors'][:3]:
                print(f"    {err}")
            if len(stats['errors']) > 3:
                print(f"    ... и ещё {len(stats['errors']) - 3} ошибок.")
        
        if preview:
            print(color_text("\nПредварительный просмотр завершен. Для применения изменений запустите удаление класса без предварительного просмотра.", "YELLOW"))
        else:
            print(color_text("\nКоррекция меток завершена.", "GREEN"))
    
    # 10. Создание лога удаления
    if files_to_delete or True:  # Всегда создаем лог, даже если файлов нет
        log_file = log_deletion(class_name, files_to_delete, deleted_count if files_to_delete else 0,
                                errors if files_to_delete else [], yaml_path, new_classes)
        print(color_text(f"Детальный лог сохранён в файл: {log_file}", "CYAN"))
    
    input(color_text("\nНажмите Enter чтобы вернуться в меню...", "CYAN"))


def main():
    """Основная функция скрипта."""
    # Определяем доступные файлы и их описания
    available_files = [
        ("create_dataset_from_camera.py", "Создание датасета с камеры (ручная разметка)"),
        ("create_dataset_from_camera_yolo.py", "Создание датасета с камеры для YOLO"),
        ("create_dataset_from_images_yolo.py", "Создание датасета из изображений для YOLO"),
        ("create_dataset_from_video_yolo.py", "Создание датасета из видео для YOLO"),
    ]
    
    # Добавляем опции управления классами (всегда доступны)
    class_management_options = [
        ("delete_class", "Удаление изображений определенного класса, их аугментаций и обновление data.yaml"),
        ("show_statistics", "Просмотр статистики классов (количество файлов, размер, баланс)"),
        ("rename_class", "Переименование класса и обновление связанных файлов"),
    ]
    
    # Проверяем, какие файлы действительно существуют
    existing_options = []
    for filename, desc in available_files:
        if Path(filename).exists():
            existing_options.append((filename, desc))
        else:
            print(color_text(f"Предупреждение: файл '{filename}' отсутствует.", "YELLOW"))
    
    # Добавляем опции управления классами как пункты 5, 6, 7
    existing_options.extend(class_management_options)
    
    if len(existing_options) == len(class_management_options) and all(opt[0] in ["delete_class", "show_statistics", "rename_class"] for opt in existing_options):
        print(color_text("Критическая ошибка: ни один из файлов создания датасета не найден.", "RED"))
        print("Убедитесь, что вы находитесь в правильной директории.")
        return
    
    while True:
        print_header()
        print_menu(existing_options)
        print(color_text("  q. Выход из программы", "CYAN"))
        print()
        
        choice = get_user_choice(len(existing_options))
        if choice is None:
            print(color_text("\nВыход из программы. До свидания!", "BLUE"))
            break
        
        selected_file, selected_desc = existing_options[choice - 1]
        print(color_text(f"\nВы выбрали: {selected_file}", "BOLD"))
        print(f"Описание: {selected_desc}")
        
        # Если выбран пункт удаления класса
        if selected_file == "delete_class":
            confirm = input(color_text("Перейти к удалению класса? (y/n): ", "YELLOW")).strip().lower()
            if confirm == 'y':
                delete_class()
            else:
                print(color_text("Отмена операции.", "YELLOW"))
            continue
        
        # Если выбран пункт просмотра статистики
        if selected_file == "show_statistics":
            confirm = input(color_text("Показать статистику классов? (y/n): ", "YELLOW")).strip().lower()
            if confirm == 'y':
                show_class_statistics()
            else:
                print(color_text("Отмена операции.", "YELLOW"))
            continue
        
        # Если выбран пункт переименования класса
        if selected_file == "rename_class":
            confirm = input(color_text("Перейти к переименованию класса? (y/n): ", "YELLOW")).strip().lower()
            if confirm == 'y':
                rename_class()
            else:
                print(color_text("Отмена операции.", "YELLOW"))
            continue
        
        # Иначе запуск скрипта
        confirm = input(color_text("Запустить? (y/n): ", "YELLOW")).strip().lower()
        if confirm == 'y':
            script_success = run_script(selected_file)
            # После завершения скрипта создания датасета предлагаем аугментацию
            if script_success:
                print(color_text("\nСкрипт завершён. Хотите запустить аугментацию изображений? (y/n)", "CYAN"))
                aug_choice = input(color_text("Ваш выбор: ", "YELLOW")).strip().lower()
                if aug_choice in ('y', 'yes'):
                    run_augmentation()
                else:
                    print(color_text("Аугментация пропущена.", "YELLOW"))
            input(color_text("\nНажмите Enter чтобы вернуться в меню...", "CYAN"))
        else:
            print(color_text("Запуск отменён.", "YELLOW"))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(color_text("\n\nПрограмма прервана пользователем.", "RED"))
        sys.exit(0)