#!/usr/bin/env python3
"""
Тест аугментации на одном изображении.
"""
import os
import sys
import cv2
import logging
import numpy as np
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_loader import ConfigLoader
from augmentation_pipeline import AugmentationPipeline, create_bbox_params
from yolo_processor import YOLOProcessor
from file_manager import FileManager
from visualizer import Visualizer

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_single_augmentation():
    """Аугментировать одно изображение и сохранить результат."""
    config_path = 'aug_config.yaml'
    loader = ConfigLoader(config_path)
    config = loader.load()
    
    # Инициализация компонентов
    processor = YOLOProcessor(
        min_area=config.bbox_params.min_area,
        min_visibility=config.bbox_params.min_visibility,
        ignore_classes=config.bbox_params.ignore_classes
    )
    
    bbox_params = create_bbox_params(config.bbox_params.min_area,
                                     config.bbox_params.min_visibility)
    
    pipeline = AugmentationPipeline(
        config.augmentations.geometric,
        config.augmentations.color,
        config.augmentations.noise,
        bbox_params
    )
    
    # Путь к первому изображению
    image_dir = os.path.join(config.dataset.source_dir, config.dataset.image_subdir)
    label_dir = os.path.join(config.dataset.source_dir, config.dataset.label_subdir)
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        logger.error("Нет изображений в директории")
        return
    
    image_name = image_files[0]
    image_path = os.path.join(image_dir, image_name)
    label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')
    
    logger.info(f"Тестируем изображение: {image_path}")
    logger.info(f"Метки: {label_path}")
    
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Не удалось загрузить изображение: {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Загрузка меток
    bboxes = []
    if os.path.exists(label_path):
        bboxes = processor.read_labels(label_path)
        logger.info(f"Загружено {len(bboxes)} bounding boxes")
    else:
        logger.warning(f"Файл меток не найден: {label_path}")
    
    # Подготовка bounding boxes для аугментации
    bboxes_list, class_labels = processor.prepare_for_augmentation(bboxes)
    
    # Применение аугментации
    augmented = pipeline.apply(image_rgb, bboxes_list, class_labels)
    if augmented is None:
        logger.error("Аугментация не применена")
        return
    
    aug_image = augmented['image']
    aug_bboxes = augmented['bboxes']
    
    logger.info(f"После аугментации: {len(aug_bboxes)} bounding boxes")
    
    # Конвертация обратно в формат YOLO
    aug_bboxes_yolo = processor.convert_to_yolo(aug_bboxes, aug_image.shape[1], aug_image.shape[0])
    
    # Визуализация
    visualizer = Visualizer()
    vis_image = visualizer.draw_bboxes(aug_image, aug_bboxes_yolo)
    
    # Сохранение результатов
    output_image_dir = os.path.join(config.dataset.output_dir, 'images', 'train')
    output_label_dir = os.path.join(config.dataset.output_dir, 'labels', 'train')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    output_image_path = os.path.join(output_image_dir, 'test_augmented.jpg')
    output_label_path = os.path.join(output_label_dir, 'test_augmented.txt')
    
    cv2.imwrite(output_image_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    processor.write_labels(aug_bboxes_yolo, output_label_path)
    
    logger.info(f"Аугментированное изображение сохранено: {output_image_path}")
    logger.info(f"Метки сохранены: {output_label_path}")
    
    # Также сохраним оригинал с bounding boxes для сравнения
    orig_bboxes_yolo = processor.convert_to_yolo(bboxes, image.shape[1], image.shape[0])
    orig_vis = visualizer.draw_bboxes(image_rgb, orig_bboxes_yolo)
    orig_output_path = os.path.join(output_image_dir, 'test_original.jpg')
    cv2.imwrite(orig_output_path, cv2.cvtColor(orig_vis, cv2.COLOR_RGB2BGR))
    logger.info(f"Оригинал с bounding boxes сохранён: {orig_output_path}")
    
    return True

if __name__ == '__main__':
    success = test_single_augmentation()
    if success:
        logger.info("Тест аугментации пройден успешно")
        sys.exit(0)
    else:
        logger.error("Тест аугментации не пройден")
        sys.exit(1)