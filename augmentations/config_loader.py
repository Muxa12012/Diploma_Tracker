"""
Модуль для загрузки и валидации конфигурации аугментации.
"""
import os
import yaml
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Конфигурация датасета."""
    source_dir: str
    image_subdir: str
    label_subdir: str
    output_dir: str
    image_extensions: list[str]
    augmentations_per_image: int

    def __post_init__(self):
        # Нормализация путей
        self.source_dir = os.path.normpath(self.source_dir)
        self.output_dir = os.path.normpath(self.output_dir)
        # Проверка существования исходной директории
        if not os.path.exists(self.source_dir):
            logger.warning(f"Исходная директория не существует: {self.source_dir}")
        # Создание выходной директории
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class GeometricConfig:
    """Конфигурация геометрических аугментаций."""
    probability: float
    rotate: Dict[str, Any]
    shift_scale_rotate: Dict[str, Any]
    horizontal_flip: bool
    vertical_flip: bool
    perspective: Dict[str, Any]
    random_crop: Dict[str, Any]


@dataclass
class ColorConfig:
    """Конфигурация цветовых аугментаций."""
    probability: float
    random_brightness_contrast: Dict[str, Any]
    gamma: Dict[str, Any]
    hue_saturation_value: Dict[str, Any]
    blur: Dict[str, Any]
    gauss_noise: Dict[str, Any]
    random_shadow: bool


@dataclass
class NoiseConfig:
    """Конфигурация шумовых аугментаций."""
    probability: float
    gaussian_noise: Dict[str, Any]
    salt_and_pepper: Dict[str, Any]
    dropout: Dict[str, Any]


@dataclass
class AugmentationsConfig:
    """Конфигурация всех аугментаций."""
    geometric: GeometricConfig
    color: ColorConfig
    noise: NoiseConfig


@dataclass
class BBoxParamsConfig:
    """Параметры обработки bounding boxes."""
    format: str
    min_area: float
    min_visibility: float
    ignore_classes: list[int]


@dataclass
class OutputConfig:
    """Конфигурация вывода."""
    keep_originals: bool
    naming_template: str
    save_debug_visualization: bool
    debug_dir: str
    generate_report: bool
    report_file: str

    def __post_init__(self):
        if self.save_debug_visualization:
            os.makedirs(self.debug_dir, exist_ok=True)


@dataclass
class LoggingConfig:
    """Конфигурация логирования."""
    level: str
    log_file: str
    console_output: bool


@dataclass
class PerformanceConfig:
    """Конфигурация производительности."""
    num_workers: int
    buffer_size: int
    cache_images: bool


@dataclass
class AugmentationConfig:
    """Полная конфигурация аугментации."""
    dataset: DatasetConfig
    augmentations: AugmentationsConfig
    bbox_params: BBoxParamsConfig
    output: OutputConfig
    logging: LoggingConfig
    performance: PerformanceConfig


class ConfigLoader:
    """Загрузчик и валидатор конфигурации."""

    def __init__(self, config_path: str = "aug_config.yaml"):
        self.config_path = config_path
        self.raw_config: Dict[str, Any] = {}
        self.config: Optional[AugmentationConfig] = None

    def _resolve_config_path(self) -> str:
        """
        Пытается найти конфигурационный файл в нескольких возможных местах.
        
        Возвращает:
            Абсолютный путь к найденному файлу.
        
        Исключения:
            FileNotFoundError: если файл не найден ни в одном из мест.
        """
        original_path = self.config_path
        possible_paths = []
        
        # 1. Как есть (относительно текущей рабочей директории)
        possible_paths.append(original_path)
        
        # 2. Относительно директории этого файла (config_loader.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths.append(os.path.join(script_dir, original_path))
        
        # 3. В поддиректории augmentations/ (если запуск из корня проекта)
        possible_paths.append(os.path.join(script_dir, 'aug_config.yaml'))
        
        # 4. В родительской директории (если config_path это просто имя файла)
        if not os.path.dirname(original_path):
            possible_paths.append(os.path.join(os.path.dirname(script_dir), original_path))
        
        # Проверяем каждый путь
        for path in possible_paths:
            if os.path.exists(path):
                logger.debug(f"Конфигурационный файл найден по пути: {path}")
                return os.path.abspath(path)
        
        # Если файл не найден, формируем информативное сообщение об ошибке
        error_msg = f"Конфигурационный файл не найден: {original_path}\n"
        error_msg += "Искали в следующих местах:\n"
        for i, path in enumerate(possible_paths, 1):
            error_msg += f"  {i}. {os.path.abspath(path)}\n"
        error_msg += "Убедитесь, что файл существует или укажите правильный путь с помощью --config."
        raise FileNotFoundError(error_msg)
    
    def load(self) -> AugmentationConfig:
        """Загружает и валидирует конфигурацию из YAML файла."""
        logger.info(f"Загрузка конфигурации из {self.config_path}")
        
        # Разрешаем путь к конфигурационному файлу
        resolved_path = self._resolve_config_path()
        self.config_path = resolved_path
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.raw_config = yaml.safe_load(f)
        
        self._validate_config()
        self.config = self._parse_config()
        logger.info("Конфигурация успешно загружена")
        return self.config

    def _validate_config(self):
        """Проверяет наличие обязательных полей и их типы."""
        required_sections = ['dataset', 'augmentations', 'bbox_params', 'output', 'logging', 'performance']
        for section in required_sections:
            if section not in self.raw_config:
                raise ValueError(f"Отсутствует обязательная секция: {section}")
        
        # Проверка dataset
        dataset = self.raw_config['dataset']
        required_dataset = ['source_dir', 'image_subdir', 'label_subdir', 'output_dir']
        for field in required_dataset:
            if field not in dataset:
                raise ValueError(f"В секции dataset отсутствует поле: {field}")
        
        # Проверка augmentations
        aug = self.raw_config['augmentations']
        required_aug_sections = ['geometric', 'color', 'noise']
        for section in required_aug_sections:
            if section not in aug:
                raise ValueError(f"В секции augmentations отсутствует подсекция: {section}")
        
        # Проверка вероятностей
        for section in required_aug_sections:
            if 'probability' not in aug[section]:
                raise ValueError(f"В {section} отсутствует поле probability")
            prob = aug[section]['probability']
            if not 0 <= prob <= 1:
                raise ValueError(f"Вероятность в {section} должна быть между 0 и 1, получено {prob}")
        
        # Проверка bbox_params
        bbox = self.raw_config['bbox_params']
        if 'format' not in bbox or bbox['format'] != 'yolo':
            raise ValueError("Формат bounding boxes должен быть 'yolo'")
        
        logger.debug("Валидация конфигурации пройдена")

    def _parse_config(self) -> AugmentationConfig:
        """Парсит raw config в объекты конфигурации."""
        # Dataset
        dataset_cfg = self.raw_config['dataset']
        dataset = DatasetConfig(
            source_dir=dataset_cfg['source_dir'],
            image_subdir=dataset_cfg.get('image_subdir', 'images/train'),
            label_subdir=dataset_cfg.get('label_subdir', 'labels/train'),
            output_dir=dataset_cfg.get('output_dir', 'datasets/camera_dataset_augmented'),
            image_extensions=dataset_cfg.get('image_extensions', ['.jpg', '.jpeg', '.png']),
            augmentations_per_image=dataset_cfg.get('augmentations_per_image', 3)
        )

        # Augmentations
        aug_cfg = self.raw_config['augmentations']
        
        geometric_cfg = aug_cfg['geometric']
        geometric = GeometricConfig(
            probability=geometric_cfg['probability'],
            rotate=geometric_cfg.get('rotate', {}),
            shift_scale_rotate=geometric_cfg.get('shift_scale_rotate', {}),
            horizontal_flip=geometric_cfg.get('horizontal_flip', True),
            vertical_flip=geometric_cfg.get('vertical_flip', False),
            perspective=geometric_cfg.get('perspective', {}),
            random_crop=geometric_cfg.get('random_crop', {})
        )
        
        color_cfg = aug_cfg['color']
        color = ColorConfig(
            probability=color_cfg['probability'],
            random_brightness_contrast=color_cfg.get('random_brightness_contrast', {}),
            gamma=color_cfg.get('gamma', {}),
            hue_saturation_value=color_cfg.get('hue_saturation_value', {}),
            blur=color_cfg.get('blur', {}),
            gauss_noise=color_cfg.get('gauss_noise', {}),
            random_shadow=color_cfg.get('random_shadow', False)
        )
        
        noise_cfg = aug_cfg['noise']
        noise = NoiseConfig(
            probability=noise_cfg['probability'],
            gaussian_noise=noise_cfg.get('gaussian_noise', {}),
            salt_and_pepper=noise_cfg.get('salt_and_pepper', {}),
            dropout=noise_cfg.get('dropout', {})
        )
        
        augmentations = AugmentationsConfig(
            geometric=geometric,
            color=color,
            noise=noise
        )

        # BBox params
        bbox_cfg = self.raw_config['bbox_params']
        bbox_params = BBoxParamsConfig(
            format=bbox_cfg.get('format', 'yolo'),
            min_area=bbox_cfg.get('min_area', 0.001),
            min_visibility=bbox_cfg.get('min_visibility', 0.3),
            ignore_classes=bbox_cfg.get('ignore_classes', [])
        )

        # Output
        output_cfg = self.raw_config['output']
        output = OutputConfig(
            keep_originals=output_cfg.get('keep_originals', True),
            naming_template=output_cfg.get('naming_template', '{original_name}_aug{index:03d}'),
            save_debug_visualization=output_cfg.get('save_debug_visualization', False),
            debug_dir=output_cfg.get('debug_dir', 'debug_visualization'),
            generate_report=output_cfg.get('generate_report', True),
            report_file=output_cfg.get('report_file', 'augmentation_report.json')
        )

        # Logging
        logging_cfg = self.raw_config['logging']
        logging_config = LoggingConfig(
            level=logging_cfg.get('level', 'INFO'),
            log_file=logging_cfg.get('log_file', 'augmentation.log'),
            console_output=logging_cfg.get('console_output', True)
        )

        # Performance
        perf_cfg = self.raw_config['performance']
        performance = PerformanceConfig(
            num_workers=perf_cfg.get('num_workers', 4),
            buffer_size=perf_cfg.get('buffer_size', 100),
            cache_images=perf_cfg.get('cache_images', False)
        )

        return AugmentationConfig(
            dataset=dataset,
            augmentations=augmentations,
            bbox_params=bbox_params,
            output=output,
            logging=logging_config,
            performance=performance
        )

    def get_config(self) -> AugmentationConfig:
        """Возвращает загруженную конфигурацию."""
        if self.config is None:
            raise RuntimeError("Конфигурация не загружена. Сначала вызовите load().")
        return self.config


def setup_logging(config: LoggingConfig):
    """Настраивает логирование на основе конфигурации."""
    log_level = getattr(logging, config.level.upper(), logging.INFO)
    
    handlers = []
    if config.console_output:
        handlers.append(logging.StreamHandler())
    if config.log_file:
        # Создаем директорию для лог-файла, если указан путь с директорией
        log_dir = os.path.dirname(config.log_file)
        if log_dir:  # Если путь содержит директорию (не пустая строка)
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(config.log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


if __name__ == "__main__":
    # Пример использования
    loader = ConfigLoader()
    try:
        config = loader.load()
        print("Конфигурация успешно загружена:")
        print(f"  Исходный датасет: {config.dataset.source_dir}")
        print(f"  Выходная директория: {config.dataset.output_dir}")
        print(f"  Аугментаций на изображение: {config.dataset.augmentations_per_image}")
    except Exception as e:
        print(f"Ошибка загрузки конфигурации: {e}")