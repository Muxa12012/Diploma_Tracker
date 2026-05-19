# -*- coding: utf-8 -*-
import cv2
import os
from ultralytics import YOLO
import numpy as np
import sys
import yaml
import time
import re

# ANSI цвета для терминала
COLOR_RESET = "\033[0m"
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_MAGENTA = "\033[95m"
COLOR_CYAN = "\033[96m"

def load_user_class_names(dataset_dir):
    """
    Загружает имена классов из data.yaml в директории датасета.
    Возвращает словарь {id: name} или пустой словарь, если файл не найден.
    """
    data_yaml_path = os.path.join(dataset_dir, 'data.yaml')
    if not os.path.exists(data_yaml_path):
        return {}
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        names = data.get('names', [])
        # Преобразуем список в словарь
        return {i: name for i, name in enumerate(names)}
    except Exception as e:
        print(f"{COLOR_YELLOW}Не удалось загрузить data.yaml: {e}{COLOR_RESET}")
        return {}

def select_class_interactive(dataset_dir, model_names):
    """Интерактивный выбор класса с цветным оформлением.
    
    Возвращает кортеж (selected_model_classes, class_mapping), где:
    - selected_model_classes: список ID классов модели для детекции (или None для всех классов)
    - class_mapping: словарь {model_class_id: user_class_id} для преобразования при сохранении аннотаций.
    """
    print(f"{COLOR_CYAN}=== ВЫБОР КЛАССА ==={COLOR_RESET}")
    print(f"{COLOR_YELLOW}Хотите добавить изображения для старого класса или нового класса?{COLOR_RESET}")
    print(f"{COLOR_GREEN}1. Старый класс (выбрать из пользовательских классов){COLOR_RESET}")
    print(f"{COLOR_GREEN}2. Новый класс (создать новый пользовательский класс){COLOR_RESET}")
    print(f"{COLOR_GREEN}3. Все классы (сохранять все обнаруженные){COLOR_RESET}")
    
    # Загружаем пользовательские классы
    user_class_names = load_user_class_names(dataset_dir)
    
    while True:
        choice = input(f"{COLOR_BLUE}Введите номер (1, 2 или 3): {COLOR_RESET}").strip()
        if choice == '1':
            # Выбор существующего пользовательского класса
            if not user_class_names:
                print(f"{COLOR_RED}Нет пользовательских классов в data.yaml. Сначала создайте новый класс.{COLOR_RESET}")
                continue
            print(f"{COLOR_CYAN}Доступные пользовательские классы:{COLOR_RESET}")
            for idx, name in user_class_names.items():
                print(f"  {idx}: {name}")
            
            while True:
                try:
                    user_class_id = int(input(f"{COLOR_BLUE}Введите номер пользовательского класса (0, 1, 2...): {COLOR_RESET}").strip())
                    if user_class_id not in user_class_names:
                        print(f"{COLOR_RED}Ошибка: класс {user_class_id} не существует в data.yaml.{COLOR_RESET}")
                        continue
                    break
                except ValueError:
                    print(f"{COLOR_RED}Ошибка: введите целое число.{COLOR_RESET}")
            
            print(f"{COLOR_GREEN}Выбран пользовательский класс ID: {user_class_id} ({user_class_names[user_class_id]}){COLOR_RESET}")
            
            # Выбор класса модели для детекции
            print(f"{COLOR_CYAN}Теперь выберите класс модели YOLO для детекции:{COLOR_RESET}")
            print(f"{COLOR_YELLOW}Доступные классы модели:{COLOR_RESET}")
            for idx, name in model_names.items():
                print(f"  {idx}: {name}")
            print(f"{COLOR_YELLOW}Вы можете выбрать любой класс модели, даже если он не соответствует пользовательскому классу.{COLOR_RESET}")
            
            while True:
                try:
                    model_class_id = int(input(f"{COLOR_BLUE}Введите номер класса модели (0, 1, 2...): {COLOR_RESET}").strip())
                    if model_class_id not in model_names:
                        print(f"{COLOR_RED}Ошибка: класс {model_class_id} не существует в модели.{COLOR_RESET}")
                        continue
                    break
                except ValueError:
                    print(f"{COLOR_RED}Ошибка: введите целое число.{COLOR_RESET}")
            
            print(f"{COLOR_GREEN}Выбран класс модели ID: {model_class_id} ({model_names[model_class_id]}){COLOR_RESET}")
            print(f"{COLOR_GREEN}Соответствие: пользовательский класс {user_class_id} ({user_class_names[user_class_id]}) ← класс модели {model_class_id} ({model_names[model_class_id]}){COLOR_RESET}")
            
            # Возвращаем список из одного класса модели и mapping
            return [model_class_id], {model_class_id: user_class_id}
            
        elif choice == '2':
            # Создание нового пользовательского класса
            data_yaml_path = os.path.join(dataset_dir, 'data.yaml')
            class_names = []
            if os.path.exists(data_yaml_path):
                try:
                    with open(data_yaml_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    class_names = data.get('names', [])
                except Exception as e:
                    print(f"{COLOR_RED}Ошибка чтения data.yaml: {e}{COLOR_RESET}")
            
            new_class_id = len(class_names)
            print(f"{COLOR_CYAN}Следующий доступный номер класса: {new_class_id}{COLOR_RESET}")
            
            class_name = input(f"{COLOR_BLUE}Введите название нового класса: {COLOR_RESET}").strip()
            if not class_name:
                class_name = f'class_{new_class_id}'
            
            # Обновление data.yaml
            if os.path.exists(data_yaml_path):
                try:
                    with open(data_yaml_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f) or {}
                    data['names'] = data.get('names', []) + [class_name]
                    data['nc'] = len(data['names'])
                    with open(data_yaml_path, 'w', encoding='utf-8') as f:
                        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                    print(f"{COLOR_GREEN}data.yaml обновлен. Добавлен класс '{class_name}' с ID {new_class_id}{COLOR_RESET}")
                except Exception as e:
                    print(f"{COLOR_RED}Ошибка обновления data.yaml: {e}{COLOR_RESET}")
            else:
                print(f"{COLOR_YELLOW}data.yaml не найден. Создайте его вручную.{COLOR_RESET}")
            
            print(f"{COLOR_GREEN}Создан новый пользовательский класс ID: {new_class_id} ('{class_name}'){COLOR_RESET}")
            
            # Выбор класса модели для детекции
            print(f"{COLOR_CYAN}Теперь выберите класс модели YOLO для детекции (или введите -1 для детекции всех классов):{COLOR_RESET}")
            print(f"{COLOR_YELLOW}Доступные классы модели:{COLOR_RESET}")
            for idx, name in model_names.items():
                print(f"  {idx}: {name}")
            
            while True:
                try:
                    model_input = input(f"{COLOR_BLUE}Введите номер класса модели (или -1 для всех классов): {COLOR_RESET}").strip()
                    if model_input == '-1':
                        model_class_id = None
                        break
                    model_class_id = int(model_input)
                    if model_class_id not in model_names:
                        print(f"{COLOR_RED}Ошибка: класс {model_class_id} не существует в модели.{COLOR_RESET}")
                        continue
                    break
                except ValueError:
                    print(f"{COLOR_RED}Ошибка: введите целое число или -1.{COLOR_RESET}")
            
            if model_class_id is None:
                print(f"{COLOR_GREEN}Будет выполнена детекция по всем классам модели.{COLOR_RESET}")
                print(f"{COLOR_YELLOW}Внимание: модель не обучена на новом классе, поэтому детекции могут отсутствовать.{COLOR_RESET}")
                # Возвращаем None для всех классов, mapping пустой (будет использоваться тождественное отображение)
                return None, {}
            else:
                print(f"{COLOR_GREEN}Выбран класс модели ID: {model_class_id} ({model_names[model_class_id]}){COLOR_RESET}")
                print(f"{COLOR_GREEN}Соответствие: пользовательский класс {new_class_id} ('{class_name}') ← класс модели {model_class_id} ({model_names[model_class_id]}){COLOR_RESET}")
                # Возвращаем список из одного класса модели и mapping
                return [model_class_id], {model_class_id: new_class_id}
            
        elif choice == '3':
            print(f"{COLOR_GREEN}Выбраны все классы.{COLOR_RESET}")
            # Возвращаем None для всех классов, mapping пустой (будет использоваться тождественное отображение)
            return None, {}
        else:
            print(f"{COLOR_RED}Неверный ввод. Пожалуйста, введите 1, 2 или 3.{COLOR_RESET}")

def generate_unique_filename(class_id, class_names, base_dir, prefix="", extension=".jpg"):
    """
    Генерирует уникальное имя файла с названием класса.
    
    Args:
        class_id: ID класса (int)
        class_names: словарь {id: name} или список имен
        base_dir: директория для сохранения (проверка существования файлов)
        prefix: дополнительный префикс (например, "video1_")
        extension: расширение файла (по умолчанию .jpg)
    
    Returns:
        base_name: имя файла без расширения (например, "face_1735123456_0001")
        full_path: полный путь к файлу с расширением
    """
    # Получаем название класса
    if isinstance(class_names, dict):
        class_name = class_names.get(class_id, f"class_{class_id}")
    elif isinstance(class_names, list):
        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"class_{class_id}"
    else:
        class_name = f"class_{class_id}"
    
    # Очищаем название класса от недопустимых символов
    class_name_clean = re.sub(r'[\\/*?:"<>| ]', '_', class_name)
    
    # Текущая временная метка (целое число)
    timestamp = int(time.time())
    
    # Счетчик для уникальности
    counter = 0
    
    while True:
        # Формируем имя файла
        if counter == 0:
            base_name = f"{class_name_clean}_{timestamp}"
        else:
            base_name = f"{class_name_clean}_{timestamp}_{counter:04d}"
        
        if prefix:
            base_name = f"{prefix}{base_name}"
        
        full_path = os.path.join(base_dir, base_name + extension)
        
        # Проверяем, существует ли файл
        if not os.path.exists(full_path):
            break
        
        counter += 1
    
    return base_name, full_path

# Пути
MODEL_PATH = 'yolo26n.pt'
DATASET_DIR = 'datasets/camera_dataset'
IMG_DIR = os.path.join(DATASET_DIR, 'images', 'train')
LABEL_DIR = os.path.join(DATASET_DIR, 'labels', 'train')
TRAIN_FILE = os.path.join(DATASET_DIR, 'train.txt')

# Определение пути к изображениям
DEFAULT_INPUT_DIR = 'datasets/input_images'
SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

if len(sys.argv) >= 2:
    # Используем переданный аргумент
    IMAGES_DIR = sys.argv[1]
    print(f"{COLOR_GREEN}Используется переданный путь: {IMAGES_DIR}{COLOR_RESET}")
else:
    # Проверяем папку по умолчанию
    if os.path.exists(DEFAULT_INPUT_DIR) and os.path.isdir(DEFAULT_INPUT_DIR):
        image_files_default = [f for f in os.listdir(DEFAULT_INPUT_DIR) if f.lower().endswith(SUPPORTED_EXT)]
        if image_files_default:
            IMAGES_DIR = DEFAULT_INPUT_DIR
            print(f"{COLOR_GREEN}Используется папка по умолчанию: {DEFAULT_INPUT_DIR}{COLOR_RESET}")
        else:
            print(f"{COLOR_YELLOW}Папка по умолчанию {DEFAULT_INPUT_DIR} пуста или не содержит изображений.{COLOR_RESET}")
            # Запрашиваем путь у пользователя
            IMAGES_DIR = input(f"{COLOR_BLUE}Введите путь к папке с изображениями: {COLOR_RESET}").strip()
    else:
        print(f"{COLOR_YELLOW}Папка по умолчанию {DEFAULT_INPUT_DIR} не существует.{COLOR_RESET}")
        # Запрашиваем путь у пользователя
        IMAGES_DIR = input(f"{COLOR_BLUE}Введите путь к папке с изображениями: {COLOR_RESET}").strip()

# Проверка существования папки
if not os.path.exists(IMAGES_DIR) or not os.path.isdir(IMAGES_DIR):
    print(f"{COLOR_RED}Ошибка: папка не найдена или не является директорией: {IMAGES_DIR}{COLOR_RESET}")
    sys.exit(1)

# Список изображений
image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(SUPPORTED_EXT)]

if not image_files:
    print(f"{COLOR_RED}Ошибка: в папке {IMAGES_DIR} не найдено изображений (поддерживаются: {SUPPORTED_EXT}){COLOR_RESET}")
    sys.exit(1)

print(f"{COLOR_GREEN}Найдено {len(image_files)} изображений для обработки.{COLOR_RESET}")

# Создание папок
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# Загрузка модели YOLO
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names  # Словарь {id: 'name'}

# Загрузка пользовательских имен классов из data.yaml
USER_CLASS_NAMES = load_user_class_names(DATASET_DIR)

# Объединение: пользовательские имена имеют приоритет над именами модели
EFFECTIVE_CLASS_NAMES = CLASS_NAMES.copy()
for class_id, name in USER_CLASS_NAMES.items():
    EFFECTIVE_CLASS_NAMES[class_id] = name

# Интерактивный выбор класса
selected_model_classes, CLASS_MAPPING = select_class_interactive(DATASET_DIR, CLASS_NAMES)
selected_classes = selected_model_classes
if selected_classes is None:
    print(f"{COLOR_YELLOW}Будут сохраняться все классы.{COLOR_RESET}")
else:
    print(f"{COLOR_GREEN}Будут сохраняться только классы модели: {selected_classes}{COLOR_RESET}")
if CLASS_MAPPING:
    print(f"{COLOR_GREEN}Соответствие классов модели → пользовательских: {CLASS_MAPPING}{COLOR_RESET}")

# Перезагрузка пользовательских имен (на случай, если был добавлен новый класс)
USER_CLASS_NAMES = load_user_class_names(DATASET_DIR)
EFFECTIVE_CLASS_NAMES = CLASS_NAMES.copy()
for class_id, name in USER_CLASS_NAMES.items():
    EFFECTIVE_CLASS_NAMES[class_id] = name

# Обработка изображений
print("\nНачало обработки...")
processed_count = 0
saved_count = 0

for img_name in image_files:
    img_path = os.path.join(IMAGES_DIR, img_name)
    
    # Чтение изображения
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Пропущено: не удалось прочитать изображение {img_name}")
        continue
    
    h, w = frame.shape[:2]

    # Детекция
    results = model(frame, verbose=False, classes=selected_classes)
    result = results[0]

    detections = []

    # Обработка боксов
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        model_cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()

        if conf < 0.5:
            continue
        
        if selected_classes is not None and model_cls_id not in selected_classes:
            continue

        # Преобразование ID класса модели в пользовательский ID
        if CLASS_MAPPING and model_cls_id in CLASS_MAPPING:
            user_cls_id = CLASS_MAPPING[model_cls_id]
        else:
            user_cls_id = model_cls_id

        # Нормализованные координаты
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        x_center = np.clip(x_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        width = np.clip(width, 0, 1)
        height = np.clip(height, 0, 1)

        detections.append((user_cls_id, x_center, y_center, width, height))

    processed_count += 1

    # Сохранение только если есть детекции
    if detections:
        # Определяем класс для имени файла (пользовательский ID)
        if selected_classes is not None and len(selected_classes) == 1:
            # Используем выбранный класс модели, преобразуем в пользовательский
            model_class_id = selected_classes[0]
            if CLASS_MAPPING and model_class_id in CLASS_MAPPING:
                class_id_for_name = CLASS_MAPPING[model_class_id]
            else:
                class_id_for_name = model_class_id
        else:
            # Используем первый обнаруженный класс (уже пользовательский ID)
            class_id_for_name = detections[0][0]
        
        # Генерация уникального имени файла с названием класса
        # Используем пользовательские имена классов (USER_CLASS_NAMES)
        base_name, save_img_path = generate_unique_filename(
            class_id=class_id_for_name,
            class_names=USER_CLASS_NAMES,
            base_dir=IMG_DIR,
            prefix="",
            extension=".jpg"
        )
        save_label_path = os.path.join(LABEL_DIR, base_name + '.txt')
        relative_img_path = os.path.join('images', 'train', base_name + '.jpg')

        # Сохраняем изображение
        cv2.imwrite(save_img_path, frame)

        # Сохраняем аннотации
        with open(save_label_path, 'w') as f:
            for det in detections:
                f.write(f'{det[0]} {det[1]} {det[2]} {det[3]} {det[4]}\n')

        # Добавляем в train.txt
        with open(TRAIN_FILE, 'a') as f:
            f.write(relative_img_path + '\n')

        saved_count += 1
        print(f'[Сохранено] {base_name}.jpg | Объектов: {len(detections)}')
    else:
        print(f'[Пропущено] {img_name} (нет объектов)')

print(f"\nГотово. Обработано: {processed_count}, сохранено: {saved_count} изображений.")
