import cv2
import os
from ultralytics import YOLO
import numpy as np
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
MODEL_PATH = 'best_one_yet/weights/best.pt'
DATASET_DIR = 'datasets/camera_dataset'
IMG_DIR = os.path.join(DATASET_DIR, 'images', 'train')
LABEL_DIR = os.path.join(DATASET_DIR, 'labels', 'train')
TRAIN_FILE = os.path.join(DATASET_DIR, 'train.txt')

# Создание папок
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# Загрузка модели YOLO
model = YOLO(MODEL_PATH)

# Получение списка классов из модели
CLASS_NAMES = model.names  # Словарь {id: 'name'}

# Интерактивный выбор класса
selected_classes = select_class_interactive(DATASET_DIR, CLASS_NAMES)
if selected_classes is None:
    print(f"{COLOR_YELLOW}Будут сохраняться все классы.{COLOR_RESET}")
else:
    print(f"{COLOR_GREEN}Будут сохраняться только классы: {selected_classes}{COLOR_RESET}")

# Настройка камеры
cap = cv2.VideoCapture(0)
cv2.namedWindow('YOLO Dataset Creator')

print("\nНажмите 'c' для сохранения кадра с автоматической разметкой, 'q' для выхода")

img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Выполнение детекции
    results = model(frame, verbose=False)
    result = results[0]

    # Подготовка аннотаций
    h, w = frame.shape[:2]
    detections = []
    display_frame = frame.copy()

    # Рисуем bounding box и собираем данные для сохранения
    for box in result.boxes:
        # Координаты
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()

        # Фильтрация по уверенности
        if conf < 0.5:
            continue

        # Фильтрация по выбранным классам
        if selected_classes is not None and cls_id not in selected_classes:
            continue

        # Рисуем прямоугольник и класс
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{CLASS_NAMES[cls_id]}: {conf:.2f}'
        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Нормализованные координаты для YOLO
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        # Ограничение значений
        x_center = np.clip(x_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        width = np.clip(width, 0, 1)
        height = np.clip(height, 0, 1)

        detections.append((cls_id, x_center, y_center, width, height))

    # Отображение количества объектов
    cv2.putText(display_frame, f'Image: {img_count} | Saved: {len(detections)}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('YOLO Dataset Creator', display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and detections:
        # Определяем класс для имени файла
        if selected_classes is not None and len(selected_classes) == 1:
            # Используем выбранный класс
            class_id_for_name = selected_classes[0]
        else:
            # Используем первый обнаруженный класс
            class_id_for_name = detections[0][0]
        
        # Генерация уникального имени файла с названием класса
        base_name, img_path = generate_unique_filename(
            class_id=class_id_for_name,
            class_names=CLASS_NAMES,
            base_dir=IMG_DIR,
            prefix="",
            extension=".jpg"
        )
        label_path = os.path.join(LABEL_DIR, base_name + '.txt')
        relative_img_path = os.path.join('datasets/camera_dataset/images/train', base_name + '.jpg')

        # Сохраняем изображение
        cv2.imwrite(img_path, frame)

        # Сохраняем разметку
        with open(label_path, 'w') as f:
            for det in detections:
                f.write(f'{det[0]} {det[1]} {det[2]} {det[3]} {det[4]}\n')

        # Добавляем путь в train.txt
        with open(TRAIN_FILE, 'a') as f:
            f.write(relative_img_path + '\n')

        print(f'Сохранено: {base_name}.jpg | Объектов: {len(detections)}')
        img_count += 1

    elif key == ord('q'):
        break

# Очистка
cap.release()
cv2.destroyAllWindows()
print(f"Создано {img_count} изображений. Готово.")
