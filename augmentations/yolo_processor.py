"""
Модуль для обработки YOLO меток: чтение, конвертация, применение аугментаций и фильтрация.
"""
import os
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Класс для представления bounding box в YOLO формате."""
    class_id: int
    x_center: float  # нормализованное (0-1)
    y_center: float  # нормализованное (0-1)
    width: float     # нормализованное (0-1)
    height: float    # нормализованное (0-1)
    
    def to_list(self) -> List[float]:
        """Возвращает bounding box в виде списка [class_id, x_center, y_center, width, height]."""
        return [self.class_id, self.x_center, self.y_center, self.width, self.height]
    
    def to_yolo_string(self) -> str:
        """Возвращает строку в формате YOLO (class_id x_center y_center width height)."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"
    
    def to_pascal_voc(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Конвертирует в формат Pascal VOC (x_min, y_min, x_max, y_max) в пикселях."""
        x_min = int((self.x_center - self.width / 2) * img_width)
        y_min = int((self.y_center - self.height / 2) * img_height)
        x_max = int((self.x_center + self.width / 2) * img_width)
        y_max = int((self.y_center + self.height / 2) * img_height)
        return x_min, y_min, x_max, y_max
    
    @classmethod
    def from_yolo_line(cls, line: str) -> 'BoundingBox':
        """Создает BoundingBox из строки YOLO формата."""
        parts = line.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Некорректный формат строки YOLO: {line}")
        
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        return cls(class_id, x_center, y_center, width, height)
    
    @classmethod
    def from_pascal_voc(cls, class_id: int, x_min: int, y_min: int, x_max: int, y_max: int,
                       img_width: int, img_height: int) -> 'BoundingBox':
        """Создает BoundingBox из координат Pascal VOC."""
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        return cls(class_id, x_center, y_center, width, height)
    
    def is_valid(self, min_area: float = 0.001, min_visibility: float = 0.3) -> bool:
        """
        Проверяет валидность bounding box.
        
        Args:
            min_area: Минимальная относительная площадь
            min_visibility: Минимальная видимость (не используется в базовой проверке)
        
        Returns:
            True если bounding box валиден
        """
        # Проверка диапазонов
        if not (0 <= self.x_center <= 1 and 0 <= self.y_center <= 1):
            return False
        
        if not (0 < self.width <= 1 and 0 < self.height <= 1):
            return False
        
        # Проверка что bounding box внутри изображения
        x_min = self.x_center - self.width / 2
        y_min = self.y_center - self.height / 2
        x_max = self.x_center + self.width / 2
        y_max = self.y_center + self.height / 2
        
        if x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1:
            # Частично вне изображения - вычисляем видимую область
            visible_width = max(0, min(x_max, 1) - max(x_min, 0))
            visible_height = max(0, min(y_max, 1) - max(y_min, 0))
            visible_area = visible_width * visible_height
            total_area = self.width * self.height
            
            if total_area == 0 or visible_area / total_area < min_visibility:
                return False
        
        # Проверка минимальной площади
        if self.width * self.height < min_area:
            return False
        
        return True


class YOLOProcessor:
    """Класс для обработки YOLO меток."""
    
    def __init__(self, min_area: float = 0.001, min_visibility: float = 0.3, 
                 ignore_classes: List[int] = None):
        """
        Инициализация процессора.
        
        Args:
            min_area: Минимальная относительная площадь bounding box
            min_visibility: Минимальная видимая область (доля)
            ignore_classes: Список class_id для игнорирования
        """
        self.min_area = min_area
        self.min_visibility = min_visibility
        self.ignore_classes = ignore_classes or []
        
        logger.debug(f"YOLOProcessor инициализирован: min_area={min_area}, "
                    f"min_visibility={min_visibility}, ignore_classes={ignore_classes}")
    
    def read_labels(self, label_path: str) -> List[BoundingBox]:
        """
        Читает YOLO метки из файла.
        
        Args:
            label_path: Путь к файлу с метками
        
        Returns:
            Список объектов BoundingBox
        """
        if not os.path.exists(label_path):
            logger.warning(f"Файл меток не найден: {label_path}")
            return []
        
        bboxes = []
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        bbox = BoundingBox.from_yolo_line(line)
                        bboxes.append(bbox)
                    except ValueError as e:
                        logger.warning(f"Ошибка в строке {line_num} файла {label_path}: {e}")
        except Exception as e:
            logger.error(f"Ошибка чтения файла {label_path}: {e}")
        
        logger.debug(f"Прочитано {len(bboxes)} bounding boxes из {label_path}")
        return bboxes
    
    def write_labels(self, bboxes: List[BoundingBox], label_path: str):
        """
        Записывает YOLO метки в файл.
        
        Args:
            bboxes: Список объектов BoundingBox
            label_path: Путь для сохранения файла
        """
        try:
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            with open(label_path, 'w', encoding='utf-8') as f:
                for bbox in bboxes:
                    f.write(bbox.to_yolo_string() + '\n')
            logger.debug(f"Записано {len(bboxes)} bounding boxes в {label_path}")
        except Exception as e:
            logger.error(f"Ошибка записи файла {label_path}: {e}")
            raise
    
    def filter_bboxes(self, bboxes: List[BoundingBox]) -> List[BoundingBox]:
        """
        Фильтрует bounding boxes по валидности и игнорируемым классам.
        
        Args:
            bboxes: Список bounding boxes для фильтрации
        
        Returns:
            Отфильтрованный список bounding boxes
        """
        filtered = []
        removed_count = 0
        ignored_count = 0
        
        for bbox in bboxes:
            # Проверка игнорируемых классов
            if bbox.class_id in self.ignore_classes:
                ignored_count += 1
                continue
            
            # Проверка валидности
            if bbox.is_valid(self.min_area, self.min_visibility):
                filtered.append(bbox)
            else:
                removed_count += 1
        
        if removed_count > 0 or ignored_count > 0:
            logger.debug(f"Фильтрация: {len(filtered)} сохранено, "
                        f"{removed_count} удалено, {ignored_count} проигнорировано")
        
        return filtered
    
    def prepare_for_augmentation(self, bboxes: List[BoundingBox]) -> Tuple[List[List[float]], List[int]]:
        """
        Подготавливает bounding boxes для передачи в пайплайн аугментаций.
        
        Args:
            bboxes: Список объектов BoundingBox
        
        Returns:
            Кортеж (bboxes_list, class_labels), где:
            - bboxes_list: список [x_center, y_center, width, height]
            - class_labels: список class_id
        """
        bboxes_list = []
        class_labels = []
        
        for bbox in bboxes:
            bboxes_list.append([bbox.x_center, bbox.y_center, bbox.width, bbox.height])
            class_labels.append(bbox.class_id)
        
        return bboxes_list, class_labels
    
    def process_augmentation_result(self, bboxes_list: List[List[float]], 
                                   class_labels: List[int]) -> List[BoundingBox]:
        """
        Обрабатывает результат аугментации, создавая объекты BoundingBox.
        
        Args:
            bboxes_list: Список bounding boxes в формате [x_center, y_center, width, height]
            class_labels: Список class_id
        
        Returns:
            Список объектов BoundingBox
        """
        if len(bboxes_list) != len(class_labels):
            logger.warning(f"Несоответствие количества bounding boxes ({len(bboxes_list)}) "
                          f"и меток классов ({len(class_labels)})")
            # Используем минимальную длину
            min_len = min(len(bboxes_list), len(class_labels))
            bboxes_list = bboxes_list[:min_len]
            class_labels = class_labels[:min_len]
        
        bboxes = []
        for bbox_data, class_id in zip(bboxes_list, class_labels):
            if len(bbox_data) != 4:
                logger.warning(f"Некорректный формат bounding box: {bbox_data}")
                continue
            
            x_center, y_center, width, height = bbox_data
            bbox = BoundingBox(class_id, x_center, y_center, width, height)
            bboxes.append(bbox)
        
        return bboxes
    
    def get_statistics(self, bboxes: List[BoundingBox]) -> Dict[str, Any]:
        """
        Возвращает статистику по bounding boxes.
        
        Args:
            bboxes: Список bounding boxes
        
        Returns:
            Словарь со статистикой
        """
        if not bboxes:
            return {
                'total': 0,
                'by_class': {},
                'avg_area': 0,
                'min_area': 0,
                'max_area': 0
            }
        
        # Статистика по классам
        class_counts = {}
        for bbox in bboxes:
            class_id = bbox.class_id
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        # Статистика по площадям
        areas = [bbox.width * bbox.height for bbox in bboxes]
        avg_area = sum(areas) / len(areas)
        min_area = min(areas)
        max_area = max(areas)
        
        return {
            'total': len(bboxes),
            'by_class': class_counts,
            'avg_area': avg_area,
            'min_area': min_area,
            'max_area': max_area
        }
    
    def validate_image_labels(self, image_path: str, label_path: str) -> bool:
        """
        Проверяет соответствие изображения и меток.
        
        Args:
            image_path: Путь к изображению
            label_path: Путь к файлу меток
        
        Returns:
            True если проверка пройдена
        """
        if not os.path.exists(image_path):
            logger.warning(f"Изображение не найдено: {image_path}")
            return False
        
        if not os.path.exists(label_path):
            # Пустой файл меток допустим
            return True
        
        try:
            # Проверяем что можем прочитать метки
            bboxes = self.read_labels(label_path)
            
            # Проверяем валидность каждого bounding box
            for bbox in bboxes:
                if not bbox.is_valid(self.min_area, self.min_visibility):
                    logger.warning(f"Невалидный bounding box в {label_path}: {bbox}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Ошибка проверки меток {label_path}: {e}")
            return False


def find_corresponding_label(image_path: str, label_dir: str, 
                            image_extensions: List[str] = None) -> Optional[str]:
    """
    Находит соответствующий файл меток для изображения.
    
    Args:
        image_path: Путь к изображению
        label_dir: Директория с метками
        image_extensions: Список расширений изображений
    
    Returns:
        Путь к файлу меток или None если не найден
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png']
    
    # Получаем имя файла без расширения
    image_name = os.path.basename(image_path)
    for ext in image_extensions:
        if image_name.lower().endswith(ext):
            image_name = image_name[:-len(ext)]
            break
    
    # Пробуем разные расширения для файла меток
    label_extensions = ['.txt']
    for ext in label_extensions:
        label_path = os.path.join(label_dir, image_name + ext)
        if os.path.exists(label_path):
            return label_path
    
    # Если файл не найден, возможно у него другое имя (например, с суффиксами)
    # Ищем файлы с похожим именем
    try:
        for filename in os.listdir(label_dir):
            if filename.startswith(image_name) and filename.endswith('.txt'):
                return os.path.join(label_dir, filename)
    except FileNotFoundError:
        pass
    
    return None


if __name__ == "__main__":
    # Пример использования
    import tempfile
    
    # Настройка логирования
    logging.basicConfig(level=logging.DEBUG)
    
    # Создание тестового процессора
    processor = YOLOProcessor(min_area=0.001, min_visibility=0.3)
    
    # Тестовые данные
    test_bboxes = [
        BoundingBox(0, 0.5, 0.5, 0.2, 0.2),
        BoundingBox(1, 0.3, 0.3, 0.1, 0.1),
        BoundingBox(0, 1.5, 0.5, 0.2, 0.2),  # Невалидный (x_center > 1)
    ]
    
    print("Тестовые bounding boxes:")
    for bbox in test_bboxes:
        print(f"  {bbox.to_yolo_string()} - валиден: {bbox.is_valid()}")
    
    # Фильтрация
    filtered = processor.filter_bboxes(test_bboxes)
    print(f"\nПосле фильтрации: {len(filtered)} bounding boxes")
    
    # Подготовка для аугментации
    bboxes_list, class_labels = processor.prepare_for_augmentation(filtered)
    print(f"\nДля аугментации: {len(bboxes_list)} bounding boxes")
    
    # Статистика
    stats = processor.get_statistics(filtered)
    print(f"\nСтатистика: {stats}")
    
    # Тест записи/чтения
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_path = f.name
        processor.write_labels(filtered, temp_path)
        print(f"\nЗаписано в {temp_path}")
        
        # Чтение обратно
        read_bboxes = processor.read_labels(temp_path)
        print(f"Прочитано обратно: {len(read_bboxes)} bounding boxes")
    
    os.unlink(temp_path)