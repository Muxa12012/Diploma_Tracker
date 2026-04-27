"""
Модуль для построения пайплайна аугментаций с использованием Albumentations.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from typing import List, Optional
from dataclasses import dataclass

from config_loader import GeometricConfig, ColorConfig, NoiseConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Конфигурация пайплайна аугментаций."""
    geometric_prob: float
    color_prob: float
    noise_prob: float
    bbox_params: Optional[dict] = None


class AugmentationPipeline:
    """Класс для создания и управления пайплайном аугментаций."""

    def __init__(self, geometric_cfg: GeometricConfig, color_cfg: ColorConfig, 
                 noise_cfg: NoiseConfig, bbox_params: dict):
        """
        Инициализация пайплайна аугментаций.
        
        Args:
            geometric_cfg: Конфигурация геометрических аугментаций
            color_cfg: Конфигурация цветовых аугментаций
            noise_cfg: Конфигурация шумовых аугментаций
            bbox_params: Параметры для обработки bounding boxes
        """
        self.geometric_cfg = geometric_cfg
        self.color_cfg = color_cfg
        self.noise_cfg = noise_cfg
        self.bbox_params = bbox_params
        
        self.train_pipeline = self._build_train_pipeline()
        self.visualization_pipeline = self._build_visualization_pipeline()
        
        logger.info("Пайплайн аугментаций создан")

    def _build_train_pipeline(self) -> A.Compose:
        """Создает основной пайплайн для тренировочных аугментаций."""
        transforms = []
        
        # Геометрические аугментации (применяются с вероятностью geometric_prob)
        if self.geometric_cfg.probability > 0:
            geometric_transforms = self._get_geometric_transforms()
            if geometric_transforms:
                transforms.append(
                    A.OneOf(geometric_transforms, p=self.geometric_cfg.probability)
                )
        
        # Цветовые аугментации (применяются с вероятностью color_prob)
        if self.color_cfg.probability > 0:
            color_transforms = self._get_color_transforms()
            if color_transforms:
                transforms.append(
                    A.OneOf(color_transforms, p=self.color_cfg.probability)
                )
        
        # Шумовые аугментации (применяются с вероятностью noise_prob)
        if self.noise_cfg.probability > 0:
            noise_transforms = self._get_noise_transforms()
            if noise_transforms:
                transforms.append(
                    A.OneOf(noise_transforms, p=self.noise_cfg.probability)
                )
        
        # Всегда применяем нормализацию (если нужно) и конвертацию в тензор
        transforms.append(A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]))
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms, bbox_params=self.bbox_params)

    def _build_visualization_pipeline(self) -> A.Compose:
        """Создает упрощенный пайплайн для визуализации (без нормализации и тензора)."""
        transforms = []
        
        # Геометрические аугментации
        if self.geometric_cfg.probability > 0:
            geometric_transforms = self._get_geometric_transforms()
            if geometric_transforms:
                transforms.append(
                    A.OneOf(geometric_transforms, p=self.geometric_cfg.probability)
                )
        
        # Цветовые аугментации
        if self.color_cfg.probability > 0:
            color_transforms = self._get_color_transforms()
            if color_transforms:
                transforms.append(
                    A.OneOf(color_transforms, p=self.color_cfg.probability)
                )
        
        # Шумовые аугментации
        if self.noise_cfg.probability > 0:
            noise_transforms = self._get_noise_transforms()
            if noise_transforms:
                transforms.append(
                    A.OneOf(noise_transforms, p=self.noise_cfg.probability)
                )
        
        return A.Compose(transforms, bbox_params=self.bbox_params)

    def _get_geometric_transforms(self) -> List[A.BasicTransform]:
        """Создает список геометрических преобразований."""
        transforms = []
        cfg = self.geometric_cfg
        
        # Поворот
        if cfg.rotate:
            transforms.append(
                A.Rotate(
                    limit=cfg.rotate.get('limit', 15),
                    border_mode=cfg.rotate.get('border_mode', 0),
                    fill_value=cfg.rotate.get('fill_value', 0),
                    p=1.0
                )
            )
        
        # Сдвиг, масштабирование и поворот
        if cfg.shift_scale_rotate:
            transforms.append(
                A.ShiftScaleRotate(
                    shift_limit=cfg.shift_scale_rotate.get('shift_limit', 0.1),
                    scale_limit=cfg.shift_scale_rotate.get('scale_limit', 0.1),
                    rotate_limit=cfg.shift_scale_rotate.get('rotate_limit', 10),
                    border_mode=cfg.shift_scale_rotate.get('border_mode', 0),
                    fill_value=cfg.shift_scale_rotate.get('fill_value', 0),
                    p=1.0
                )
            )
        
        # Горизонтальное отражение
        if cfg.horizontal_flip:
            transforms.append(A.HorizontalFlip(p=1.0))
        
        # Вертикальное отражение
        if cfg.vertical_flip:
            transforms.append(A.VerticalFlip(p=1.0))
        
        # Перспективное искажение
        if cfg.perspective:
            transforms.append(
                A.Perspective(
                    scale=cfg.perspective.get('scale', 0.1),
                    keep_size=cfg.perspective.get('keep_size', True),
                    p=1.0
                )
            )
        
        # Случайная обрезка
        if cfg.random_crop and cfg.random_crop.get('enabled', False):
            transforms.append(
                A.RandomCrop(
                    height=int(cfg.random_crop.get('min_height', 0.7) * 100),  # в пикселях
                    width=int(cfg.random_crop.get('min_width', 0.7) * 100),
                    p=1.0
                )
            )
        
        return transforms

    def _get_color_transforms(self) -> List[A.BasicTransform]:
        """Создает список цветовых преобразований."""
        transforms = []
        cfg = self.color_cfg
        
        # Яркость и контраст
        if cfg.random_brightness_contrast:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=cfg.random_brightness_contrast.get('brightness_limit', 0.2),
                    contrast_limit=cfg.random_brightness_contrast.get('contrast_limit', 0.2),
                    p=1.0
                )
            )
        
        # Гамма-коррекция
        if cfg.gamma:
            gamma_limit = cfg.gamma.get('limit', (80, 120))
            if isinstance(gamma_limit, tuple) and len(gamma_limit) == 2:
                # Преобразуем проценты в коэффициенты гаммы
                gamma_min = gamma_limit[0] / 100.0
                gamma_max = gamma_limit[1] / 100.0
                transforms.append(
                    A.RandomGamma(gamma_limit=(gamma_min, gamma_max), p=1.0)
                )
        
        # Оттенок, насыщенность, значение
        if cfg.hue_saturation_value:
            transforms.append(
                A.HueSaturationValue(
                    hue_shift_limit=cfg.hue_saturation_value.get('hue_shift_limit', 20),
                    sat_shift_limit=cfg.hue_saturation_value.get('sat_shift_limit', 30),
                    val_shift_limit=cfg.hue_saturation_value.get('val_shift_limit', 20),
                    p=1.0
                )
            )
        
        # Размытие
        if cfg.blur:
            transforms.append(
                A.Blur(
                    blur_limit=cfg.blur.get('blur_limit', 7),
                    p=1.0
                )
            )
        
        # Гауссов шум
        if cfg.gauss_noise:
            var_limit = cfg.gauss_noise.get('var_limit', (10.0, 50.0))
            transforms.append(
                A.GaussNoise(var_limit=var_limit, p=1.0)
            )
        
        # Случайные тени (эмулируем через RandomShadow)
        if cfg.random_shadow:
            transforms.append(
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    shadow_dimension=5,
                    p=1.0
                )
            )
        
        return transforms

    def _get_noise_transforms(self) -> List[A.BasicTransform]:
        """Создает список шумовых преобразований."""
        transforms = []
        cfg = self.noise_cfg
        
        # Гауссов шум (дополнительный)
        if cfg.gaussian_noise:
            var_limit = cfg.gaussian_noise.get('var_limit', (10.0, 50.0))
            transforms.append(
                A.GaussNoise(var_limit=var_limit, p=1.0)
            )
        
        # Соль-перец (эмулируем через SaltAndPepper)
        if cfg.salt_and_pepper:
            amount = cfg.salt_and_pepper.get('amount', 0.01)
            transforms.append(
                A.SaltAndPepper(
                    amount=amount,
                    p=1.0
                )
            )
        
        # Dropout (выпадение пикселей)
        if cfg.dropout:
            transforms.append(
                A.CoarseDropout(
                    max_holes=cfg.dropout.get('max_holes', 8),
                    max_height=cfg.dropout.get('max_height', 8),
                    max_width=cfg.dropout.get('max_width', 8),
                    min_holes=1,
                    min_height=4,
                    min_width=4,
                    fill_value=0,
                    p=1.0
                )
            )
        
        return transforms

    def apply(self, image, bboxes=None, class_labels=None, for_visualization=False):
        """
        Применяет аугментации к изображению и bounding boxes.
        
        Args:
            image: Входное изображение (numpy array)
            bboxes: Список bounding boxes в формате [x_center, y_center, width, height]
            class_labels: Список меток классов
            for_visualization: Если True, использует pipeline для визуализации
        
        Returns:
            Словарь с аугментированным изображением и bounding boxes
        """
        if for_visualization:
            pipeline = self.visualization_pipeline
        else:
            pipeline = self.train_pipeline
        
        # Подготовка bounding boxes для Albumentations
        albumentations_bboxes = []
        if bboxes is not None:
            for idx, bbox in enumerate(bboxes):
                # Конвертируем из YOLO формата (нормализованный центр) в Pascal VOC (x_min, y_min, x_max, y_max)
                x_center, y_center, width, height = bbox
                x_min = (x_center - width / 2)
                y_min = (y_center - height / 2)
                x_max = (x_center + width / 2)
                y_max = (y_center + height / 2)
                
                # Отладочный вывод
                logger.debug(f"BBox {idx}: x_center={x_center}, y_center={y_center}, width={width}, height={height}")
                logger.debug(f"  до обрезки: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
                
                # Обрезаем координаты до диапазона [0, 1]
                x_min = max(0.0, min(1.0, x_min))
                y_min = max(0.0, min(1.0, y_min))
                x_max = max(0.0, min(1.0, x_max))
                y_max = max(0.0, min(1.0, y_max))
                
                logger.debug(f"  после обрезки: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
                
                # Проверяем валидность bounding box после обрезки
                if x_max <= x_min or y_max <= y_min:
                    logger.warning(f"Bounding box {idx} стал невалидным после обрезки: "
                                   f"x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
                    continue
                
                # Логируем предупреждение если исходные координаты выходили за пределы
                if (x_center - width / 2) < 0 or (x_center + width / 2) > 1 or \
                   (y_center - height / 2) < 0 or (y_center + height / 2) > 1:
                    logger.debug(f"Bounding box {idx} выходит за границы изображения: "
                                 f"x_center={x_center}, y_center={y_center}, width={width}, height={height}")
                
                albumentations_bboxes.append([x_min, y_min, x_max, y_max])
        
        # Применение аугментаций
        if bboxes is not None and class_labels is not None:
            transformed = pipeline(
                image=image,
                bboxes=albumentations_bboxes,
                class_labels=class_labels
            )
        elif bboxes is not None:
            transformed = pipeline(
                image=image,
                bboxes=albumentations_bboxes
            )
        else:
            transformed = pipeline(image=image)
        
        # Конвертация обратно в YOLO формат
        result_bboxes = []
        result_classes = []
        
        if 'bboxes' in transformed:
            for bbox in transformed['bboxes']:
                x_min, y_min, x_max, y_max = bbox
                width = x_max - x_min
                height = y_max - y_min
                x_center = x_min + width / 2
                y_center = y_min + height / 2
                result_bboxes.append([x_center, y_center, width, height])
        
        if 'class_labels' in transformed:
            result_classes = transformed['class_labels']
        
        return {
            'image': transformed['image'],
            'bboxes': result_bboxes,
            'class_labels': result_classes
        }

    def get_pipeline_info(self) -> dict:
        """Возвращает информацию о пайплайне."""
        return {
            'geometric_probability': self.geometric_cfg.probability,
            'color_probability': self.color_cfg.probability,
            'noise_probability': self.noise_cfg.probability,
            'num_geometric_transforms': len(self._get_geometric_transforms()),
            'num_color_transforms': len(self._get_color_transforms()),
            'num_noise_transforms': len(self._get_noise_transforms())
        }


def create_bbox_params(min_area: float = 0.001, min_visibility: float = 0.3) -> dict:
    """
    Создает параметры для обработки bounding boxes в Albumentations.
    
    Args:
        min_area: Минимальная площадь bounding box (относительная)
        min_visibility: Минимальная видимая область (доля)
    
    Returns:
        Словарь с параметрами для bbox_params
    """
    return A.BboxParams(
        format='albumentations',  # Нормализованные координаты x_min, y_min, x_max, y_max
        min_area=min_area,
        min_visibility=min_visibility,
        label_fields=['class_labels']
    )


if __name__ == "__main__":
    # Пример использования
    import numpy as np
    from config_loader import ConfigLoader
    
    # Загрузка конфигурации
    loader = ConfigLoader()
    config = loader.load()
    
    # Создание параметров bounding boxes
    bbox_params = create_bbox_params(
        min_area=config.bbox_params.min_area,
        min_visibility=config.bbox_params.min_visibility
    )
    
    # Создание пайплайна
    pipeline = AugmentationPipeline(
        geometric_cfg=config.augmentations.geometric,
        color_cfg=config.augmentations.color,
        noise_cfg=config.augmentations.noise,
        bbox_params=bbox_params
    )
    
    print("Пайплайн создан успешно")
    print(f"Информация о пайплайне: {pipeline.get_pipeline_info()}")
    
    # Тестовое изображение
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_bboxes = [[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]
    test_classes = [0, 1]
    
    # Применение аугментаций
    result = pipeline.apply(test_image, test_bboxes, test_classes, for_visualization=True)
    print(f"Результат: изображение shape={result['image'].shape}, bboxes={len(result['bboxes'])}")