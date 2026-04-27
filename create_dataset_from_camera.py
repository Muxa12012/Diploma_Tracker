import cv2
import os
import numpy as np
import yaml
import time
import re
import sys

# ANSI цвета для терминала
COLOR_RESET = "\033[0m"
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_MAGENTA = "\033[95m"
COLOR_CYAN = "\033[96m"

def select_class(dataset_dir):
    """Интерактивный выбор класса с цветным оформлением."""
    print(f"{COLOR_CYAN}=== ВЫБОР КЛАССА ==={COLOR_RESET}")
    print(f"{COLOR_YELLOW}Хотите добавить изображения для старого класса или нового класса?{COLOR_RESET}")
    print(f"{COLOR_GREEN}1. Старый класс{COLOR_RESET}")
    print(f"{COLOR_GREEN}2. Новый класс{COLOR_RESET}")
    
    while True:
        choice = input(f"{COLOR_BLUE}Введите номер (1 или 2): {COLOR_RESET}").strip()
        if choice == '1':
            # Выбор существующего класса
            data_yaml_path = os.path.join(dataset_dir, 'data.yaml')
            class_names = []
            if os.path.exists(data_yaml_path):
                try:
                    with open(data_yaml_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    class_names = data.get('names', [])
                    print(f"{COLOR_CYAN}Существующие классы:{COLOR_RESET}")
                    for idx, name in enumerate(class_names):
                        print(f"  {idx}: {name}")
                except Exception as e:
                    print(f"{COLOR_RED}Ошибка чтения data.yaml: {e}{COLOR_RESET}")
            
            if not class_names:
                print(f"{COLOR_YELLOW}Конфигурация data.yaml не найдена или пуста.{COLOR_RESET}")
                print(f"{COLOR_YELLOW}Введите номер класса вручную.{COLOR_RESET}")
            
            while True:
                try:
                    class_id = int(input(f"{COLOR_BLUE}Введите номер существующего класса (0, 1, 2...): {COLOR_RESET}").strip())
                    if class_names and class_id >= len(class_names):
                        print(f"{COLOR_RED}Ошибка: класс {class_id} не существует. Максимальный индекс: {len(class_names)-1}{COLOR_RESET}")
                        continue
                    break
                except ValueError:
                    print(f"{COLOR_RED}Ошибка: введите целое число.{COLOR_RESET}")
            
            print(f"{COLOR_GREEN}Выбран класс ID: {class_id}{COLOR_RESET}")
            return class_id
            
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
            return new_class_id
            
        else:
            print(f"{COLOR_RED}Неверный ввод. Пожалуйста, введите 1 или 2.{COLOR_RESET}")

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

def find_working_camera(max_index=5, backends=None):
    """
    Автоматически находит работающую камеру, перебирая индексы и backends.
    
    Args:
        max_index: максимальный индекс камеры для проверки (0..max_index-1)
        backends: список backends для проверки (по умолчанию [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY])
    
    Returns:
        (cap, camera_index, backend_name): объект VideoCapture, индекс камеры и имя backend
        или (None, -1, None) если камера не найдена
    """
    if backends is None:
        # Приоритетные backends для Windows
        backends = [
            (cv2.CAP_DSHOW, "DSHOW"),
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_ANY, "ANY"),
        ]
    
    for backend_id, backend_name in backends:
        print(f"{COLOR_CYAN}Проверяем backend {backend_name}...{COLOR_RESET}")
        for index in range(max_index):
            cap = cv2.VideoCapture(index, backend_id)
            if cap.isOpened():
                # Пробуем прочитать кадр для подтверждения
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"{COLOR_GREEN}  Найдена камера: индекс {index}, backend {backend_name}{COLOR_RESET}")
                    return cap, index, backend_name
                else:
                    cap.release()
            else:
                cap.release()
    
    print(f"{COLOR_RED}Не удалось найти работающую камеру.{COLOR_RESET}")
    return None, -1, None

def initialize_camera(camera_index=None, backend=None):
    """
    Инициализирует камеру с указанным индексом или автоматически находит работающую.
    
    Args:
        camera_index: конкретный индекс камеры (если None, выполняется автоматический поиск)
        backend: конкретный backend (если None, используется автоматический выбор)
    
    Returns:
        cap: объект VideoCapture или None в случае ошибки
    """
    if camera_index is not None:
        # Используем указанный индекс
        if backend is not None:
            cap = cv2.VideoCapture(camera_index, backend)
        else:
            # Пробуем приоритетные backends для Windows
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            for backend_id in backends:
                cap = cv2.VideoCapture(camera_index, backend_id)
                if cap.isOpened():
                    print(f"{COLOR_GREEN}Камера {camera_index} открыта с backend {backend_id}{COLOR_RESET}")
                    return cap
                cap.release()
            # Если ни один backend не сработал, пробуем без указания backend
            cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            print(f"{COLOR_GREEN}Камера {camera_index} открыта успешно.{COLOR_RESET}")
            return cap
        else:
            print(f"{COLOR_RED}Не удалось открыть камеру с индексом {camera_index}.{COLOR_RESET}")
            return None
    else:
        # Автоматический поиск работающей камеры
        print(f"{COLOR_CYAN}=== АВТОМАТИЧЕСКИЙ ПОИСК КАМЕРЫ ==={COLOR_RESET}")
        cap, found_index, backend_name = find_working_camera()
        if cap is not None:
            print(f"{COLOR_GREEN}Используется камера с индексом {found_index} (backend {backend_name}){COLOR_RESET}")
            return cap
        else:
            print(f"{COLOR_RED}Работающая камера не найдена.{COLOR_RESET}")
            print(f"{COLOR_YELLOW}Проверьте подключение камеры и драйверы.{COLOR_RESET}")
            return None

# Параметры
DATASET_DIR = 'datasets/camera_dataset'
IMG_DIR = os.path.join(DATASET_DIR, 'images', 'train')
LABEL_DIR = os.path.join(DATASET_DIR, 'labels', 'train')
TRAIN_FILE = os.path.join(DATASET_DIR, 'train.txt')

# Создание папок
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# Выбор класса
class_id = select_class(DATASET_DIR)

# Загрузка названий классов из data.yaml
class_names = []
data_yaml_path = os.path.join(DATASET_DIR, 'data.yaml')
if os.path.exists(data_yaml_path):
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        class_names = data.get('names', [])
    except Exception as e:
        print(f"{COLOR_YELLOW}Не удалось загрузить названия классов: {e}{COLOR_RESET}")
else:
    print(f"{COLOR_YELLOW}Файл data.yaml не найден. Используются стандартные имена классов.{COLOR_RESET}")

# Переменные для разметки
drawing = False
ix, iy = -1, -1
bbox = []  # Сохраняем координаты после отпускания мыши

# Функция для разметки
def draw_bbox(event, x, y, flags, param):
    global ix, iy, drawing, bbox, frame, paused

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        bbox = []  # Сброс при новом нажатии

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Показываем прямоугольник в реальном времени
            temp_frame = frame.copy()
            cv2.rectangle(temp_frame, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Create Dataset', temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Фиксируем bounding box
        bbox = [min(ix, x), min(iy, y), max(ix, x), max(iy, y)]

# Настройка камеры с улучшенной обработкой ошибок
print(f"{COLOR_CYAN}=== ИНИЦИАЛИЗАЦИЯ КАМЕРЫ ==={COLOR_RESET}")
cap = initialize_camera(camera_index=None, backend=None)
if cap is None:
    print(f"{COLOR_RED}Не удалось инициализировать камеру. Скрипт завершается.{COLOR_RESET}")
    sys.exit(1)

cv2.namedWindow('Create Dataset')
cv2.setMouseCallback('Create Dataset', draw_bbox)

print(f"{COLOR_GREEN}Камера успешно инициализирована.{COLOR_RESET}")
print("Нажмите 'c' для захвата кадра и разметки, 'p' для паузы, 'q' для выхода")

img_count = 0
paused = False
last_frame = None

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame.copy()
    else:
        frame = last_frame.copy() if last_frame is not None else frame

    # Отображение текущего bounding box, если он есть
    display_frame = frame.copy()
    if bbox:
        cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # Отображение состояния паузы
    h, w = display_frame.shape[:2]
    status = 'PAUSED' if paused else 'RUNNING'
    cv2.putText(display_frame, status, (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if paused else (0, 255, 0), 2)
    
    cv2.putText(display_frame, f'Image: {img_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Create Dataset', display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        paused = not paused
        if paused:
            print("Видео на паузе. Теперь можно нарисовать рамку.")
        else:
            print("Продолжение захвата.")

    if key == ord('c') and bbox:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Нормализация координат
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        # Ограничение значений от 0 до 1
        x_center = np.clip(x_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        width = np.clip(width, 0, 1)
        height = np.clip(height, 0, 1)

        # Генерация уникального имени файла с названием класса
        base_name, img_path = generate_unique_filename(
            class_id=class_id,
            class_names=class_names,
            base_dir=IMG_DIR,
            prefix="",
            extension=".jpg"
        )
        label_path = os.path.join(LABEL_DIR, base_name + '.txt')
        relative_img_path = os.path.join('datasets/camera_dataset/images/train', base_name + '.jpg')

        # Сохранение изображения и разметки
        cv2.imwrite(img_path, frame)
        with open(label_path, 'w') as f:
            f.write(f'{class_id} {x_center} {y_center} {width} {height}\n')

        # Добавление пути в train.txt
        with open(TRAIN_FILE, 'a') as f:
            f.write(relative_img_path + '\n')

        print(f'Сохранено: {base_name}.jpg')
        img_count += 1
        bbox = []  # Сброс bbox после сохранения

    elif key == ord('q'):
        break

# Очистка
cap.release()
cv2.destroyAllWindows()
print(f"Создано {img_count} изображений. Готово.")
