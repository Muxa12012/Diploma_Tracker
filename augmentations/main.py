"""
Основной модуль для запуска аугментации изображений и меток YOLO.
"""
import os
import sys
import cv2
import logging
import traceback
from typing import Dict, Any, Optional
import numpy as np

# Добавляем текущую директорию в путь для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_loader import ConfigLoader, setup_logging
from augmentation_pipeline import AugmentationPipeline, create_bbox_params
from yolo_processor import YOLOProcessor, BoundingBox
from file_manager import FileManager
from visualizer import Visualizer


class AugmentationApp:
    """Основной класс приложения для аугментации."""
    
    def __init__(self, config_path: str = "aug_config.yaml"):
        """
        Инициализация приложения.
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.config_path = config_path
        self.config = None
        self.pipeline = None
        self.processor = None
        self.file_manager = None
        self.visualizer = None
        
        # Создание логгера для экземпляра класса
        self.logger = logging.getLogger(__name__)
        
        # Статистика для текущего изображения
        self.current_image_stats = {}
        
        self.logger.info(f"AugmentationApp инициализирован с конфигом: {config_path}")
    
    def setup(self):
        """Настройка всех компонентов приложения."""
        try:
            # Загрузка конфигурации
            loader = ConfigLoader(self.config_path)
            self.config = loader.load()
            
            # Настройка логирования
            setup_logging(self.config.logging)
            self.logger.info("Логирование настроено")
            
            # Создание компонентов
            self._create_components()
            
            self.logger.info("Приложение настроено успешно")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка настройки приложения: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _create_components(self):
        """Создает все необходимые компоненты."""
        # Создание параметров bounding boxes
        bbox_params = create_bbox_params(
            min_area=self.config.bbox_params.min_area,
            min_visibility=self.config.bbox_params.min_visibility
        )
        
        # Создание пайплайна аугментаций
        self.pipeline = AugmentationPipeline(
            geometric_cfg=self.config.augmentations.geometric,
            color_cfg=self.config.augmentations.color,
            noise_cfg=self.config.augmentations.noise,
            bbox_params=bbox_params
        )
        
        # Создание процессора YOLO
        self.processor = YOLOProcessor(
            min_area=self.config.bbox_params.min_area,
            min_visibility=self.config.bbox_params.min_visibility,
            ignore_classes=self.config.bbox_params.ignore_classes
        )
        
        # Создание менеджера файлов
        self.file_manager = FileManager(self.config)
        
        # Создание визуализатора
        self.visualizer = Visualizer()
        
        self.logger.debug("Все компоненты созданы")
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Обрабатывает одно изображение: читает, аугментирует и сохраняет результаты.
        
        Args:
            image_path: Путь к изображению
        
        Returns:
            Словарь со статистикой обработки изображения
        """
        self.current_image_stats = {
            'image_path': image_path,
            'augmentations_created': 0,
            'skipped': False,
            'failed': False,
            'error_message': None,
            'bbox_stats_before': {},
            'bbox_stats_after': {},
            'augmentation_types': {}
        }
        
        try:
            # Получение пути к меткам
            label_path = self.file_manager.get_label_path(image_path)
            
            # Чтение изображения
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            
            # Чтение меток
            original_bboxes = self.processor.read_labels(label_path) if label_path else []
            
            # Фильтрация меток
            filtered_bboxes = self.processor.filter_bboxes(original_bboxes)
            
            # Статистика до аугментации
            self.current_image_stats['bbox_stats_before'] = self.processor.get_statistics(filtered_bboxes)
            
            # Копирование оригиналов если требуется
            self.file_manager.copy_original_if_needed(image_path, label_path)
            
            # Создание аугментаций
            augmented_images_data = []
            
            for i in range(self.config.dataset.augmentations_per_image):
                # Подготовка данных для аугментации
                bboxes_list, class_labels = self.processor.prepare_for_augmentation(filtered_bboxes)
                
                # Применение аугментации
                result = self.pipeline.apply(
                    image=image,
                    bboxes=bboxes_list if bboxes_list else None,
                    class_labels=class_labels if class_labels else None,
                    for_visualization=False  # Используем тренировочный пайплайн
                )
                
                # Обработка результата
                augmented_bboxes = self.processor.process_augmentation_result(
                    result['bboxes'], result['class_labels']
                )
                
                # Фильтрация после аугментации
                final_bboxes = self.processor.filter_bboxes(augmented_bboxes)
                
                # Генерация путей для сохранения
                output_image_path, output_label_path = self.file_manager.generate_output_paths(image_path, i)
                
                # Сохранение аугментированного изображения
                success = self.file_manager.save_augmented_image(result['image'], output_image_path)
                if not success:
                    self.logger.warning(f"Не удалось сохранить аугментированное изображение: {output_image_path}")
                    continue
                
                # Сохранение меток
                if final_bboxes:
                    self.processor.write_labels(final_bboxes, output_label_path)
                elif os.path.exists(output_label_path):
                    # Удаляем пустой файл меток если он был создан ранее
                    os.remove(output_label_path)
                
                # Сбор данных для визуализации (если включена)
                if self.config.output.save_debug_visualization:
                    augmented_images_data.append({
                        'image': result['image'],
                        'bboxes': final_bboxes
                    })
                
                # Обновление статистики
                self.current_image_stats['augmentations_created'] += 1
                
                # Статистика после аугментации (для последней аугментации)
                if i == self.config.dataset.augmentations_per_image - 1:
                    self.current_image_stats['bbox_stats_after'] = self.processor.get_statistics(final_bboxes)
            
            # Создание визуализации для отладки
            if (self.config.output.save_debug_visualization and 
                augmented_images_data and
                self.config.dataset.augmentations_per_image > 0):
                
                # Генерируем имя для файла визуализации
                original_name = os.path.splitext(os.path.basename(image_path))[0]
                debug_path = os.path.join(
                    self.config.output.debug_dir,
                    f"{original_name}_debug_grid.png"
                )
                
                # Подготавливаем данные для сетки
                aug_images = [data['image'] for data in augmented_images_data]
                aug_bboxes_list = [data['bboxes'] for data in augmented_images_data]
                
                # Создаем сетку визуализации
                self.visualizer.create_debug_grid(
                    original_image=image,
                    augmented_images=aug_images[:3],  # Максимум 3 для визуализации
                    original_bboxes=filtered_bboxes,
                    augmented_bboxes_list=aug_bboxes_list[:3],
                    output_path=debug_path,
                    max_cols=2
                )
            
            self.logger.info(f"Обработано изображение {os.path.basename(image_path)}: "
                       f"{self.current_image_stats['augmentations_created']} аугментаций создано")
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки изображения {image_path}: {e}")
            self.logger.error(traceback.format_exc())
            self.current_image_stats['failed'] = True
            self.current_image_stats['error_message'] = str(e)
        
        # Обновление общей статистики
        self.file_manager.update_statistics(self.current_image_stats)
        
        return self.current_image_stats
    
    def run(self):
        """Основной метод запуска аугментации."""
        if not self.setup():
            self.logger.error("Не удалось настроить приложение. Завершение.")
            return False
        
        try:
            self.logger.info("=" * 60)
            self.logger.info("ЗАПУСК АУГМЕНТАЦИИ ДАННЫХ")
            self.logger.info("=" * 60)
            
            # Получение списка изображений
            image_paths = self.file_manager.get_image_paths()
            
            if not image_paths:
                self.logger.warning("Изображения не найдены. Завершение.")
                return False
            
            self.logger.info(f"Найдено {len(image_paths)} изображений для обработки")
            
            # Обработка каждого изображения
            for i, image_path in enumerate(image_paths, 1):
                self.logger.info(f"[{i}/{len(image_paths)}] Обработка: {os.path.basename(image_path)}")
                
                stats = self.process_image(image_path)
                
                if stats['failed']:
                    self.logger.warning(f"  Ошибка: {stats.get('error_message', 'Неизвестная ошибка')}")
                elif stats['skipped']:
                    self.logger.info("  Пропущено")
                else:
                    self.logger.info(f"  Успешно: создано {stats['augmentations_created']} аугментаций")
            
            # Сохранение отчета
            self.file_manager.save_report()
            
            # Вывод информации о выходных данных
            structure = self.file_manager.get_output_structure()
            self.logger.info("\n" + "=" * 60)
            self.logger.info("АУГМЕНТАЦИЯ ЗАВЕРШЕНА")
            self.logger.info("=" * 60)
            self.logger.info(f"Выходная директория: {structure['output_dir']}")
            self.logger.info(f"Изображения: {structure['images_dir']}")
            self.logger.info(f"Метки: {structure['labels_dir']}")
            
            if structure['debug_dir']:
                self.logger.info(f"Визуализации: {structure['debug_dir']}")
            
            self.logger.info(f"Отчет: {structure['report_file']}")
            self.logger.info("=" * 60)
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("\nАугментация прервана пользователем")
            return False
        except Exception as e:
            self.logger.error(f"Критическая ошибка при выполнении аугментации: {e}")
            self.logger.error(traceback.format_exc())
            return False
        finally:
            # Очистка
            self.file_manager.cleanup()


def main():
    """Точка входа для запуска из командной строки."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Аугментация изображений и меток YOLO с использованием Albumentations'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='aug_config.yaml',
        help='Путь к конфигурационному файлу (по умолчанию: aug_config.yaml)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Включить подробный вывод (DEBUG уровень)'
    )
    
    args = parser.parse_args()
    
    # Настройка базового логирования до загрузки конфига
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Запуск приложения
    app = AugmentationApp(args.config)
    success = app.run()
    
    if success:
        logger.info("Аугментация успешно завершена")
        return 0
    else:
        logger.error("Аугментация завершена с ошибками")
        return 1


if __name__ == "__main__":
    # Глобальный логгер для модуля
    logger = logging.getLogger(__name__)
    
    # Запуск main функции
    sys.exit(main())