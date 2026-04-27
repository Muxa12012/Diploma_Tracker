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

def select_class_interactive(dataset_dir, model_names):
    """Интерактивный выбор класса с цветным оформлением."""
    print(f"{COLOR_CYAN}=== ВЫБОР КЛАССА ==={COLOR_RESET}")
    print(f"{COLOR_YELLOW}Хотите добавить изображения для старого класса или нового класса?{COLOR_RESET}")
    print(f"{COLOR_GREEN}1. Старый класс{COLOR_RESET}")
    print(f"{COLOR_GREEN}2. Новый класс{COLOR_RESET}")
    print(f"{COLOR_GREEN}3. Все классы (сохранять все обнаруженные){COLOR_RESET}")
    
    while True:
        choice = input(f"{COLOR_BLUE}Введите номер (1, 2 или 3): {COLOR_RESET}").strip()
        if choice == '1':
            # Выбор существующего класса из модели
            print(f"{COLOR_CYAN}Доступные классы модели:{COLOR_RESET}")
            for idx, name in model_names.items():
                print(f"  {idx}: {name}")
            
            while True:
                try:
                    class_id = int(input(f"{COLOR_BLUE}Введите номер существующего класса (0, 1, 2...): {COLOR_RESET}").strip())
                    if class_id not in model_names:
                        print(f"{COLOR_RED}Ошибка: класс {class_id} не существует в модели.{COLOR_RESET}")
                        continue
                    break
                except ValueError:
                    print(f"{COLOR_RED}Ошибка: введите целое число.{COLOR_RESET}")
            
            print(f"{COLOR_GREEN}Выбран класс ID: {class_id} ({model_names[class_id]}){COLOR_RESET}")
            return [class_id]  # список из одного класса
            
        elif choice == '2':
            # Создание нового класса
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
            
            print(f"{COLOR_GREEN}Создан новый класс ID: {new_class_id} ('{class_name}'){COLOR_RESET}")
            # Возвращаем None, чтобы сохранять все классы (так как модель не знает о новом классе)
            # Но можно фильтровать по классам модели, новый класс не будет обнаружен.
            # Поэтому лучше вернуть None (все классы) и предупредить пользователя.
            print(f"{COLOR_YELLOW}Внимание: модель не обучена на новом классе, поэтому детекции могут отсутствовать.{COLOR_RESET}")
            return None  # сохранять все классы
            
        elif choice == '3':
            print(f"{COLOR_GREEN}Выбраны все классы.{COLOR_RESET}")
            return None  # None означает все классы
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
DATASET_DIR = 'datasets/images_dataset'
IMG_DIR = os.path.join(DATASET_DIR, 'images', 'train')
LABEL_DIR = os.path.join(DATASET_DIR, 'labels', 'train')
TRAIN_FILE = os.path.join(DATASET_DIR, 'train.txt')

# Проверка аргументов
if len(sys.argv) < 2:
    print("Использование: python create_dataset_from_images_yolo.py <папка_с_изображениями>")
    print("  <папка_с_изображениями> - путь к директории с фото (jpg, jpeg, png)")
    sys.exit(1)

IMAGES_DIR = sys.argv[1]

# Проверка существования папки
if not os.path.exists(IMAGES_DIR) or not os.path.isdir(IMAGES_DIR):
    print(f"Ошибка: папка не найдена или не является директорией: {IMAGES_DIR}")
    sys.exit(1)

# Поддерживаемые расширения
SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Список изображений
image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(SUPPORTED_EXT)]

if not image_files:
    print(f"Ошибка: в папке {IMAGES_DIR} не найдено изображений (поддерживаются: {SUPPORTED_EXT})")
    sys.exit(1)

print(f"Найдено {len(image_files)} изображений для обработки.")

# Создание папок
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# Загрузка модели YOLO
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names  # Словарь {id: 'name'}

# Интерактивный выбор класса
selected_classes = select_class_interactive(DATASET_DIR, CLASS_NAMES)
if selected_classes is None:
    print(f"{COLOR_YELLOW}Будут сохраняться все классы.{COLOR_RESET}")
else:
    print(f"{COLOR_GREEN}Будут сохраняться только классы: {selected_classes}{COLOR_RESET}")

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
    results = model(frame, verbose=False)
    result = results[0]

    detections = []

    # Обработка боксов
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()

        if conf < 0.5:
            continue
        
        if selected_classes is not None and cls_id not in selected_classes:
            continue

        # Нормализованные координаты
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        x_center = np.clip(x_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        width = np.clip(width, 0, 1)
        height = np.clip(height, 0, 1)

        detections.append((cls_id, x_center, y_center, width, height))

    processed_count += 1

    # Сохранение только если есть детекции
    if detections:
        # Определяем класс для имени файла
        if selected_classes is not None and len(selected_classes) == 1:
            # Используем выбранный класс
            class_id_for_name = selected_classes[0]
        else:
            # Используем первый обнаруженный класс
            class_id_for_name = detections[0][0]
        
        # Генерация уникального имени файла с названием класса
        base_name, save_img_path = generate_unique_filename(
            class_id=class_id_for_name,
            class_names=CLASS_NAMES,
            base_dir=IMG_DIR,
            prefix="",
            extension=".jpg"
        )
        save_label_path = os.path.join(LABEL_DIR, base_name + '.txt')
        relative_img_path = os.path.join('datasets/images_dataset/images/train', base_name + '.jpg')

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
