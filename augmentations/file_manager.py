"""
Модуль для управления файлами и директориями при аугментации.
"""
import os
import shutil
import logging
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class FileManager:
    """Класс для управления файлами и директориями при аугментации."""
    
    def __init__(self, config):
        """
        Инициализация менеджера файлов.
        
        Args:
            config: Конфигурация аугментации (объект AugmentationConfig)
        """
        self.config = config
        self.dataset_cfg = config.dataset
        self.output_cfg = config.output
        
        # Создание необходимых директорий
        self._create_directories()
        
        # Статистика
        self.stats = {
            'total_images': 0,
            'total_augmented': 0,
            'skipped_images': 0,
            'failed_images': 0,
            'total_bboxes_before': 0,
            'total_bboxes_after': 0,
            'by_class_before': {},
            'by_class_after': {},
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'augmentation_types': {}
        }
        
        logger.info(f"FileManager инициализирован. Выходная директория: {self.dataset_cfg.output_dir}")
    
    def _create_directories(self):
        """Создает необходимые директории для выходных данных."""
        # Основная выходная директория
        os.makedirs(self.dataset_cfg.output_dir, exist_ok=True)
        
        # Поддиректории для изображений и меток
        self.output_images_dir = os.path.join(self.dataset_cfg.output_dir, "images", "train")
        self.output_labels_dir = os.path.join(self.dataset_cfg.output_dir, "labels", "train")
        
        os.makedirs(self.output_images_dir, exist_ok=True)
        os.makedirs(self.output_labels_dir, exist_ok=True)
        
        # Директория для отладки (если включена)
        if self.output_cfg.save_debug_visualization:
            os.makedirs(self.output_cfg.debug_dir, exist_ok=True)
        
        logger.debug(f"Созданы директории: {self.output_images_dir}, {self.output_labels_dir}")
    
    def get_image_paths(self) -> List[str]:
        """
        Возвращает список путей к изображениям в исходном датасете.
        
        Returns:
            Список абсолютных путей к изображениям
        """
        image_dir = os.path.join(self.dataset_cfg.source_dir, self.dataset_cfg.image_subdir)
        
        if not os.path.exists(image_dir):
            logger.error(f"Директория с изображениями не найдена: {image_dir}")
            return []
        
        image_paths = []
        for ext in self.dataset_cfg.image_extensions:
            pattern = f"*{ext}"
            for path in Path(image_dir).glob(pattern):
                image_paths.append(str(path))
        
        # Также ищем файлы с другими регистрами расширений
        for ext in self.dataset_cfg.image_extensions:
            pattern = f"*{ext.upper()}"
            for path in Path(image_dir).glob(pattern):
                if str(path) not in image_paths:
                    image_paths.append(str(path))
        
        logger.info(f"Найдено {len(image_paths)} изображений в {image_dir}")
        return sorted(image_paths)
    
    def get_label_path(self, image_path: str) -> Optional[str]:
        """
        Возвращает путь к соответствующему файлу меток для изображения.
        
        Args:
            image_path: Путь к изображению
        
        Returns:
            Путь к файлу меток или None если не найден
        """
        # Получаем относительный путь от директории изображений
        image_dir = os.path.join(self.dataset_cfg.source_dir, self.dataset_cfg.image_subdir)
        try:
            rel_path = os.path.relpath(image_path, image_dir)
        except ValueError:
            # Если изображение не в поддиректории, используем только имя файла
            rel_path = os.path.basename(image_path)
        
        # Меняем расширение на .txt
        name_without_ext = os.path.splitext(rel_path)[0]
        label_rel_path = name_without_ext + ".txt"
        
        # Полный путь к метке
        label_dir = os.path.join(self.dataset_cfg.source_dir, self.dataset_cfg.label_subdir)
        label_path = os.path.join(label_dir, label_rel_path)
        
        # Проверяем существование
        if os.path.exists(label_path):
            return label_path
        
        # Если файл не найден, ищем альтернативные варианты
        # (например, если имя файла содержит дополнительные суффиксы)
        label_name = os.path.basename(label_rel_path)
        if os.path.exists(label_dir):
            for filename in os.listdir(label_dir):
                if filename.startswith(os.path.splitext(label_name)[0]) and filename.endswith('.txt'):
                    return os.path.join(label_dir, filename)
        
        logger.debug(f"Файл меток не найден для {image_path}")
        return None
    
    def generate_output_paths(self, original_image_path: str, aug_index: int) -> Tuple[str, str]:
        """
        Генерирует пути для сохранения аугментированного изображения и меток.
        
        Args:
            original_image_path: Путь к оригинальному изображению
            aug_index: Индекс аугментации (0-based)
        
        Returns:
            Кортеж (output_image_path, output_label_path)
        """
        # Получаем имя оригинального файла без расширения
        original_name = os.path.splitext(os.path.basename(original_image_path))[0]
        
        # Генерируем имя аугментированного файла по шаблону
        output_name = self.output_cfg.naming_template.format(
            original_name=original_name,
            index=aug_index + 1  # 1-based для пользователя
        )
        
        # Определяем расширение изображения
        ext = os.path.splitext(original_image_path)[1].lower()
        if ext not in self.dataset_cfg.image_extensions:
            # Используем первое расширение из списка
            ext = self.dataset_cfg.image_extensions[0] if self.dataset_cfg.image_extensions else '.jpg'
        
        # Формируем пути
        output_image_path = os.path.join(self.output_images_dir, output_name + ext)
        output_label_path = os.path.join(self.output_labels_dir, output_name + ".txt")
        
        return output_image_path, output_label_path
    
    def copy_original_if_needed(self, original_image_path: str, original_label_path: Optional[str]):
        """
        Копирует оригинальные изображения и метки в выходную директорию, если требуется.
        
        Args:
            original_image_path: Путь к оригинальному изображению
            original_label_path: Путь к оригинальным меткам (может быть None)
        """
        if not self.output_cfg.keep_originals:
            return
        
        # Генерируем пути для оригинальных файлов
        original_name = os.path.splitext(os.path.basename(original_image_path))[0]
        ext = os.path.splitext(original_image_path)[1].lower()
        
        output_image_path = os.path.join(self.output_images_dir, original_name + ext)
        output_label_path = os.path.join(self.output_labels_dir, original_name + ".txt")
        
        # Копируем изображение, если еще не скопировано
        if not os.path.exists(output_image_path):
            try:
                shutil.copy2(original_image_path, output_image_path)
                logger.debug(f"Скопировано оригинальное изображение: {output_image_path}")
            except Exception as e:
                logger.error(f"Ошибка копирования изображения {original_image_path}: {e}")
        
        # Копируем метки, если они существуют
        if original_label_path and os.path.exists(original_label_path):
            if not os.path.exists(output_label_path):
                try:
                    shutil.copy2(original_label_path, output_label_path)
                    logger.debug(f"Скопированы оригинальные метки: {output_label_path}")
                except Exception as e:
                    logger.error(f"Ошибка копирования меток {original_label_path}: {e}")
    
    def save_augmented_image(self, image, output_path: str):
        """
        Сохраняет аугментированное изображение.
        
        Args:
            image: Изображение (numpy array, PIL Image или torch.Tensor)
            output_path: Путь для сохранения
        """
        try:
            import cv2
            
            # Логируем тип и форму для диагностики
            logger.debug(f"Сохранение изображения типа {type(image)}, shape: {getattr(image, 'shape', 'N/A')}, dtype: {getattr(image, 'dtype', 'N/A')}")
            
            # Если изображение в формате PIL, сохраняем напрямую
            if hasattr(image, 'save'):
                # Это PIL Image
                image.save(output_path)
                logger.debug(f"Сохранено аугментированное изображение (PIL): {output_path}")
                return True
            
            # Проверяем, является ли изображение torch.Tensor
            if hasattr(image, 'is_cuda') or hasattr(image, 'device'):
                # Это вероятно torch.Tensor
                try:
                    import torch
                    if isinstance(image, torch.Tensor):
                        # Конвертируем в numpy array
                        image = image.detach().cpu().numpy()
                        # Если тензор имеет размерность (C, H, W), меняем на (H, W, C)
                        if len(image.shape) == 3 and image.shape[0] in [1, 3, 4]:
                            image = np.transpose(image, (1, 2, 0))
                except ImportError:
                    pass  # torch не установлен, пропускаем
            
            # Теперь image должен быть numpy array
            if not isinstance(image, np.ndarray):
                logger.error(f"Изображение не является numpy array после преобразований: {type(image)}")
                return False
            
            # Логируем информацию о массиве
            logger.debug(f"После преобразования: shape={image.shape}, dtype={image.dtype}")
            
            # Конвертируем BGR в RGB если нужно
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Проверяем порядок каналов (предполагаем RGB)
                # Если значения в диапазоне [0, 1] (float), масштабируем до [0, 255]
                if image.dtype == np.float32 or image.dtype == np.float64:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                elif image.dtype != np.uint8:
                    # Другие типы приводим к uint8
                    image = image.astype(np.uint8)
                
                # Конвертируем RGB в BGR для OpenCV
                cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            else:
                # Для grayscale или других форматов сохраняем как есть
                cv2.imwrite(output_path, image)
            
            logger.debug(f"Сохранено аугментированное изображение: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения изображения {output_path}: {e}", exc_info=True)
            return False
    
    def update_statistics(self, image_stats: Dict[str, Any]):
        """
        Обновляет общую статистику на основе статистики одного изображения.
        
        Args:
            image_stats: Статистика для одного изображения
        """
        self.stats['total_images'] += 1
        self.stats['total_augmented'] += image_stats.get('augmentations_created', 0)
        
        if image_stats.get('skipped', False):
            self.stats['skipped_images'] += 1
        
        if image_stats.get('failed', False):
            self.stats['failed_images'] += 1
        
        # Статистика по bounding boxes
        bbox_stats_before = image_stats.get('bbox_stats_before', {})
        bbox_stats_after = image_stats.get('bbox_stats_after', {})
        
        self.stats['total_bboxes_before'] += bbox_stats_before.get('total', 0)
        self.stats['total_bboxes_after'] += bbox_stats_after.get('total', 0)
        
        # Обновляем статистику по классам
        for class_id, count in bbox_stats_before.get('by_class', {}).items():
            self.stats['by_class_before'][class_id] = self.stats['by_class_before'].get(class_id, 0) + count
        
        for class_id, count in bbox_stats_after.get('by_class', {}).items():
            self.stats['by_class_after'][class_id] = self.stats['by_class_after'].get(class_id, 0) + count
        
        # Статистика по типам аугментаций
        aug_types = image_stats.get('augmentation_types', {})
        for aug_type, count in aug_types.items():
            self.stats['augmentation_types'][aug_type] = self.stats['augmentation_types'].get(aug_type, 0) + count
    
    def save_report(self):
        """Сохраняет отчет о процессе аугментации."""
        if not self.output_cfg.generate_report:
            return
        
        self.stats['end_time'] = datetime.now().isoformat()
        
        # Вычисляем продолжительность
        start_time = datetime.fromisoformat(self.stats['start_time'])
        end_time = datetime.fromisoformat(self.stats['end_time'])
        duration = end_time - start_time
        self.stats['duration_seconds'] = duration.total_seconds()
        
        # Дополнительная статистика
        if self.stats['total_images'] > 0:
            self.stats['success_rate'] = (
                (self.stats['total_images'] - self.stats['failed_images'] - self.stats['skipped_images']) 
                / self.stats['total_images'] * 100
            )
            self.stats['avg_augmentations_per_image'] = (
                self.stats['total_augmented'] / self.stats['total_images']
            )
        else:
            self.stats['success_rate'] = 0
            self.stats['avg_augmentations_per_image'] = 0
        
        # Сохраняем отчет в JSON
        report_path = os.path.join(self.dataset_cfg.output_dir, self.output_cfg.report_file)
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
            logger.info(f"Отчет сохранен: {report_path}")
            
            # Также выводим сводку в лог
            self._log_summary()
        except Exception as e:
            logger.error(f"Ошибка сохранения отчета {report_path}: {e}")
    
    def _log_summary(self):
        """Выводит сводку статистики в лог."""
        logger.info("=" * 50)
        logger.info("СВОДКА АУГМЕНТАЦИИ")
        logger.info("=" * 50)
        logger.info(f"Всего изображений обработано: {self.stats['total_images']}")
        logger.info(f"  Успешно: {self.stats['total_images'] - self.stats['failed_images'] - self.stats['skipped_images']}")
        logger.info(f"  Пропущено: {self.stats['skipped_images']}")
        logger.info(f"  Ошибок: {self.stats['failed_images']}")
        logger.info(f"Всего аугментированных изображений создано: {self.stats['total_augmented']}")
        logger.info(f"Среднее аугментаций на изображение: {self.stats.get('avg_augmentations_per_image', 0):.2f}")
        logger.info(f"Bounding boxes до аугментации: {self.stats['total_bboxes_before']}")
        logger.info(f"Bounding boxes после аугментации: {self.stats['total_bboxes_after']}")
        
        if self.stats['by_class_before']:
            logger.info("Распределение по классам (до):")
            for class_id, count in sorted(self.stats['by_class_before'].items()):
                logger.info(f"  Класс {class_id}: {count}")
        
        if self.stats['by_class_after']:
            logger.info("Распределение по классам (после):")
            for class_id, count in sorted(self.stats['by_class_after'].items()):
                logger.info(f"  Класс {class_id}: {count}")
        
        if self.stats.get('augmentation_types'):
            logger.info("Использованные типы аугментаций:")
            for aug_type, count in sorted(self.stats['augmentation_types'].items()):
                logger.info(f"  {aug_type}: {count}")
        
        logger.info(f"Продолжительность: {self.stats.get('duration_seconds', 0):.2f} секунд")
        logger.info(f"Успешность: {self.stats.get('success_rate', 0):.2f}%")
        logger.info("=" * 50)
    
    def cleanup(self):
        """Очистка временных файлов и завершение работы."""
        # В текущей реализации нечего очищать
        pass
    
    def get_output_structure(self) -> Dict[str, Any]:
        """
        Возвращает информацию о структуре выходных данных.
        
        Returns:
            Словарь с информацией о директориях
        """
        return {
            'output_dir': self.dataset_cfg.output_dir,
            'images_dir': self.output_images_dir,
            'labels_dir': self.output_labels_dir,
            'debug_dir': self.output_cfg.debug_dir if self.output_cfg.save_debug_visualization else None,
            'report_file': os.path.join(self.dataset_cfg.output_dir, self.output_cfg.report_file)
        }


if __name__ == "__main__":
    # Пример использования
    from config_loader import ConfigLoader
    
    # Загрузка конфигурации
    loader = ConfigLoader()
    config = loader.load()
    
    # Создание менеджера файлов
    file_manager = FileManager(config)
    
    # Получение путей к изображениям
    image_paths = file_manager.get_image_paths()
    print(f"Найдено {len(image_paths)} изображений")
    
    if image_paths:
        # Пример для первого изображения
        image_path = image_paths[0]
        label_path = file_manager.get_label_path(image_path)
        
        print(f"\nПример для изображения: {image_path}")
        print(f"Соответствующие метки: {label_path}")
        
        # Генерация путей для аугментаций
        for i in range(3):
            output_image_path, output_label_path = file_manager.generate_output_paths(image_path, i)
            print(f"Аугментация {i+1}:")
            print(f"  Изображение: {output_image_path}")
            print(f"  Метки: {output_label_path}")
        
        # Структура выходных данных
        structure = file_manager.get_output_structure()
        print(f"\nСтруктура выходных данных: {structure}")