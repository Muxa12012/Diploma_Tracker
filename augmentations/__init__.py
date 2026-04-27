"""
Пакет для аугментации изображений и меток YOLO.
"""

__version__ = "1.0.0"
__author__ = "Practice_Diploma_Tracker"

# Импорты основных модулей для удобства
from .config_loader import ConfigLoader
from .augmentation_pipeline import AugmentationPipeline
from .yolo_processor import YOLOProcessor
from .file_manager import FileManager
from .visualizer import Visualizer

__all__ = [
    "ConfigLoader",
    "AugmentationPipeline",
    "YOLOProcessor",
    "FileManager",
    "Visualizer",
]