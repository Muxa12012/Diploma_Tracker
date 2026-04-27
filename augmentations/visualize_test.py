#!/usr/bin/env python3
"""
Визуализация оригинального изображения с bounding boxes.
"""
import os
import sys
import cv2
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yolo_processor import YOLOProcessor
from visualizer import Visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_original():
    image_path = 'datasets/camera_dataset/images/train/image_0000.jpg'
    label_path = 'datasets/camera_dataset/labels/train/image_0000.txt'
    
    if not os.path.exists(image_path):
        logger.error(f"Изображение не найдено: {image_path}")
        return
    if not os.path.exists(label_path):
        logger.error(f"Метки не найдены: {label_path}")
        return
    
    # Загрузка изображения
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Загрузка меток
    processor = YOLOProcessor(min_area=0.001, min_visibility=0.3)
    bboxes = processor.read_labels(label_path)
    logger.info(f"Загружено {len(bboxes)} bounding boxes")
    
    # Визуализация (bboxes уже являются объектами BoundingBox)
    visualizer = Visualizer()
    vis_image = visualizer.draw_bboxes(image_rgb, bboxes)
    
    # Сохранение
    output_dir = 'debug_visualization'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'original_with_bboxes.jpg')
    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    logger.info(f"Визуализация сохранена: {output_path}")
    
    # Также выведем координаты bounding boxes
    for i, bbox in enumerate(bboxes):
        logger.info(f"BBox {i}: class={bbox.class_id}, x={bbox.x_center:.4f}, y={bbox.y_center:.4f}, w={bbox.width:.4f}, h={bbox.height:.4f}")
    
    return True

if __name__ == '__main__':
    success = visualize_original()
    if success:
        logger.info("Визуализация успешно выполнена")
        sys.exit(0)
    else:
        logger.error("Визуализация не удалась")
        sys.exit(1)