#!/usr/bin/env python3
"""
Тест сохранения изображений с различными типами данных.
"""
import sys
import os
import numpy as np
import torch
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from file_manager import FileManager
from config_loader import ConfigLoader

def test_save_with_tensor():
    """Тест сохранения torch.Tensor."""
    print("=== Тест сохранения torch.Tensor ===")
    
    # Загружаем конфиг для создания FileManager
    config_path = 'aug_config.yaml'
    loader = ConfigLoader(config_path)
    config = loader.load()
    
    # Создаем временную директорию для выходных данных
    temp_dir = tempfile.mkdtemp()
    config.dataset.output_dir = temp_dir
    file_manager = FileManager(config)
    
    # Создаем тестовый тензор (симулируем выход из пайплайна)
    # Формат (C, H, W) = (3, 256, 256), значения [0, 1]
    tensor = torch.rand(3, 256, 256, dtype=torch.float32)
    print(f"Создан тензор: shape={tensor.shape}, dtype={tensor.dtype}")
    
    output_path = os.path.join(temp_dir, "test_tensor.jpg")
    success = file_manager.save_augmented_image(tensor, output_path)
    print(f"Сохранение тензора: {'Успех' if success else 'Ошибка'}")
    if os.path.exists(output_path):
        print(f"Файл создан: {output_path}, размер: {os.path.getsize(output_path)} байт")
    else:
        print("Файл не создан")
    
    # Тест с тензором в формате (H, W, C)
    tensor_hwc = tensor.permute(1, 2, 0)  # (256, 256, 3)
    output_path2 = os.path.join(temp_dir, "test_tensor_hwc.jpg")
    success2 = file_manager.save_augmented_image(tensor_hwc, output_path2)
    print(f"Сохранение тензора HWC: {'Успех' if success2 else 'Ошибка'}")
    
    # Тест с numpy array float
    np_float = np.random.rand(256, 256, 3).astype(np.float32)
    output_path3 = os.path.join(temp_dir, "test_np_float.jpg")
    success3 = file_manager.save_augmented_image(np_float, output_path3)
    print(f"Сохранение numpy float: {'Успех' if success3 else 'Ошибка'}")
    
    # Тест с numpy array uint8
    np_uint8 = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    output_path4 = os.path.join(temp_dir, "test_np_uint8.jpg")
    success4 = file_manager.save_augmented_image(np_uint8, output_path4)
    print(f"Сохранение numpy uint8: {'Успех' if success4 else 'Ошибка'}")
    
    # Тест с PIL Image (если установлен PIL)
    try:
        from PIL import Image
        pil_image = Image.fromarray(np_uint8)
        output_path5 = os.path.join(temp_dir, "test_pil.jpg")
        success5 = file_manager.save_augmented_image(pil_image, output_path5)
        print(f"Сохранение PIL Image: {'Успех' if success5 else 'Ошибка'}")
    except ImportError:
        print("PIL не установлен, пропускаем тест")
    
    # Очистка
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Временная директория удалена")
    print()

def test_save_with_real_augmentation():
    """Тест сохранения реального аугментированного изображения из пайплайна."""
    print("=== Тест с реальной аугментацией ===")
    from augmentation_pipeline import AugmentationPipeline, create_bbox_params
    from yolo_processor import YOLOProcessor
    
    config_path = 'aug_config.yaml'
    loader = ConfigLoader(config_path)
    config = loader.load()
    
    # Временная директория
    temp_dir = tempfile.mkdtemp()
    config.dataset.output_dir = temp_dir
    file_manager = FileManager(config)
    
    # Загружаем тестовое изображение
    image_path = 'datasets/camera_dataset/images/train/image_0000.jpg'
    if not os.path.exists(image_path):
        print(f"Тестовое изображение не найдено: {image_path}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return
    
    import cv2
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Загружено изображение: {image_path}, shape={image_rgb.shape}")
    
    # Инициализация пайплайна
    bbox_params = create_bbox_params(config.bbox_params.min_area,
                                     config.bbox_params.min_visibility)
    pipeline = AugmentationPipeline(
        config.augmentations.geometric,
        config.augmentations.color,
        config.augmentations.noise,
        bbox_params
    )
    
    # Применяем аугментацию (for_visualization=False, чтобы получить тензор)
    result = pipeline.apply(image_rgb, for_visualization=False)
    aug_image = result['image']
    print(f"Аугментированное изображение тип: {type(aug_image)}, shape={getattr(aug_image, 'shape', 'N/A')}")
    
    # Сохраняем
    output_path = os.path.join(temp_dir, "real_augmented.jpg")
    success = file_manager.save_augmented_image(aug_image, output_path)
    print(f"Сохранение реального аугментированного изображения: {'Успех' if success else 'Ошибка'}")
    if success and os.path.exists(output_path):
        print(f"Файл создан: {output_path}")
    
    # Также тестируем с for_visualization=True (должен вернуть numpy array)
    result2 = pipeline.apply(image_rgb, for_visualization=True)
    aug_image2 = result2['image']
    print(f"Аугментированное изображение (visualization) тип: {type(aug_image2)}, shape={aug_image2.shape if hasattr(aug_image2, 'shape') else 'N/A'}")
    
    output_path2 = os.path.join(temp_dir, "real_augmented_vis.jpg")
    success2 = file_manager.save_augmented_image(aug_image2, output_path2)
    print(f"Сохранение (visualization): {'Успех' if success2 else 'Ошибка'}")
    
    # Очистка
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Временная директория удалена")
    print()

if __name__ == '__main__':
    test_save_with_tensor()
    test_save_with_real_augmentation()
    print("Все тесты завершены.")