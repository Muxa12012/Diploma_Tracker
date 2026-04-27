"""
Модуль для визуализации результатов аугментации.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from yolo_processor import BoundingBox

logger = logging.getLogger(__name__)


class Visualizer:
    """Класс для визуализации изображений и bounding boxes."""
    
    # Цвета для разных классов (BGR формат для OpenCV)
    COLOR_PALETTE = [
        (255, 0, 0),    # Красный
        (0, 255, 0),    # Зеленый
        (0, 0, 255),    # Синий
        (255, 255, 0),  # Голубой
        (255, 0, 255),  # Пурпурный
        (0, 255, 255),  # Желтый
        (128, 0, 0),    # Темно-красный
        (0, 128, 0),    # Темно-зеленый
        (0, 0, 128),    # Темно-синий
        (128, 128, 0),  # Оливковый
    ]
    
    def __init__(self, class_names: Optional[Dict[int, str]] = None):
        """
        Инициализация визуализатора.
        
        Args:
            class_names: Словарь соответствия class_id -> имя класса
        """
        self.class_names = class_names or {}
        
        # Создаем палитру цветов с запасом
        self.colors = self.COLOR_PALETTE * 10  # Повторяем палитру для многих классов
        
        logger.debug("Visualizer инициализирован")
    
    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Возвращает цвет для заданного class_id."""
        return self.colors[class_id % len(self.colors)]
    
    def get_class_name(self, class_id: int) -> str:
        """Возвращает имя класса или строку с class_id."""
        return self.class_names.get(class_id, f"Class {class_id}")
    
    def draw_bboxes(self, image: np.ndarray, bboxes: List[BoundingBox], 
                   thickness: int = 2, font_scale: float = 0.5) -> np.ndarray:
        """
        Рисует bounding boxes на изображении.
        
        Args:
            image: Изображение (numpy array в формате BGR или RGB)
            bboxes: Список объектов BoundingBox
            thickness: Толщина линий
            font_scale: Масштаб шрифта для подписей
        
        Returns:
            Изображение с нарисованными bounding boxes
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Копируем изображение чтобы не изменять оригинал
            result = image.copy()
            height, width = result.shape[:2]
        else:
            # Градации серого или другое
            if len(image.shape) == 2:
                result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                result = image.copy()
            height, width = result.shape[:2]
        
        for bbox in bboxes:
            # Конвертируем нормализованные координаты в пиксели
            x_min, y_min, x_max, y_max = bbox.to_pascal_voc(width, height)
            
            # Получаем цвет и имя класса
            color = self.get_color(bbox.class_id)
            class_name = self.get_class_name(bbox.class_id)
            
            # Рисуем прямоугольник
            cv2.rectangle(result, (x_min, y_min), (x_max, y_max), color, thickness)
            
            # Рисуем подпись с классом
            label = f"{class_name}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            # Фон для подписи
            cv2.rectangle(
                result, 
                (x_min, y_min - label_height - baseline - 5),
                (x_min + label_width, y_min),
                color,
                -1  # Заполненный прямоугольник
            )
            
            # Текст подписи
            cv2.putText(
                result,
                label,
                (x_min, y_min - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # Белый текст
                1
            )
        
        return result
    
    def visualize_comparison(self, original_image: np.ndarray, augmented_image: np.ndarray,
                           original_bboxes: List[BoundingBox], augmented_bboxes: List[BoundingBox],
                           title: str = "Сравнение аугментации") -> plt.Figure:
        """
        Создает фигуру matplotlib для сравнения оригинального и аугментированного изображений.
        
        Args:
            original_image: Оригинальное изображение
            augmented_image: Аугментированное изображение
            original_bboxes: Bounding boxes оригинального изображения
            augmented_bboxes: Bounding boxes аугментированного изображения
            title: Заголовок фигуры
        
        Returns:
            Объект matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Конвертируем BGR в RGB для matplotlib если нужно
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_rgb = original_image
        
        if len(augmented_image.shape) == 3 and augmented_image.shape[2] == 3:
            augmented_rgb = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
        else:
            augmented_rgb = augmented_image
        
        # Отображаем оригинальное изображение
        axes[0].imshow(original_rgb)
        axes[0].set_title(f"Оригинал ({len(original_bboxes)} bboxes)")
        axes[0].axis('off')
        
        # Рисуем bounding boxes на оригинале
        self._draw_bboxes_matplotlib(axes[0], original_bboxes, original_image.shape)
        
        # Отображаем аугментированное изображение
        axes[1].imshow(augmented_rgb)
        axes[1].set_title(f"Аугментировано ({len(augmented_bboxes)} bboxes)")
        axes[1].axis('off')
        
        # Рисуем bounding boxes на аугментированном
        self._draw_bboxes_matplotlib(axes[1], augmented_bboxes, augmented_image.shape)
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def _draw_bboxes_matplotlib(self, ax, bboxes: List[BoundingBox], image_shape: Tuple[int, ...]):
        """
        Рисует bounding boxes на axes matplotlib.
        
        Args:
            ax: Объект axes matplotlib
            bboxes: Список bounding boxes
            image_shape: Форма изображения (height, width, channels)
        """
        height, width = image_shape[:2]
        
        for bbox in bboxes:
            # Конвертируем нормализованные координаты в пиксели
            x_min, y_min, x_max, y_max = bbox.to_pascal_voc(width, height)
            
            # Получаем цвет и имя класса
            color = self.get_color(bbox.class_id)
            # Конвертируем BGR в RGB для matplotlib
            color_rgb = (color[2]/255, color[1]/255, color[0]/255)
            
            class_name = self.get_class_name(bbox.class_id)
            
            # Создаем прямоугольник
            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor=color_rgb,
                facecolor='none',
                label=class_name
            )
            
            # Добавляем прямоугольник на axes
            ax.add_patch(rect)
            
            # Добавляем подпись
            ax.text(
                x_min, y_min - 5,
                class_name,
                color=color_rgb,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
            )
    
    def save_visualization(self, image: np.ndarray, bboxes: List[BoundingBox], 
                          output_path: str, title: Optional[str] = None):
        """
        Сохраняет визуализацию с bounding boxes в файл.
        
        Args:
            image: Изображение
            bboxes: Список bounding boxes
            output_path: Путь для сохранения
            title: Заголовок (необязательно)
        """
        # Рисуем bounding boxes
        visualized = self.draw_bboxes(image, bboxes)
        
        # Добавляем заголовок если указан
        if title:
            height, width = visualized.shape[:2]
            cv2.putText(
                visualized,
                title,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )
        
        # Сохраняем
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, visualized)
        logger.debug(f"Визуализация сохранена: {output_path}")
    
    def create_debug_grid(self, original_image: np.ndarray, augmented_images: List[np.ndarray],
                         original_bboxes: List[BoundingBox], augmented_bboxes_list: List[List[BoundingBox]],
                         output_path: str, max_cols: int = 3):
        """
        Создает сетку с оригинальным и аугментированными изображениями.
        
        Args:
            original_image: Оригинальное изображение
            augmented_images: Список аугментированных изображений
            original_bboxes: Bounding boxes оригинального изображения
            augmented_bboxes_list: Список списков bounding boxes для каждого аугментированного изображения
            output_path: Путь для сохранения сетки
            max_cols: Максимальное количество колонок в сетке
        """
        if len(augmented_images) != len(augmented_bboxes_list):
            logger.warning("Количество изображений и списков bounding boxes не совпадает")
            return
        
        # Определяем размер сетки
        num_images = 1 + len(augmented_images)  # оригинал + аугментированные
        num_cols = min(max_cols, num_images)
        num_rows = (num_images + num_cols - 1) // num_cols
        
        # Создаем фигуру
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
        
        # Преобразуем axes в плоский список для удобства
        if num_rows == 1 and num_cols == 1:
            axes = [[axes]]
        elif num_rows == 1:
            axes = [axes]
        elif num_cols == 1:
            axes = [[ax] for ax in axes]
        
        axes_flat = [ax for row in axes for ax in row]
        
        # Отображаем оригинальное изображение
        ax = axes_flat[0]
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_rgb = original_image
        
        ax.imshow(original_rgb)
        ax.set_title(f"Оригинал ({len(original_bboxes)} bboxes)")
        ax.axis('off')
        self._draw_bboxes_matplotlib(ax, original_bboxes, original_image.shape)
        
        # Отображаем аугментированные изображения
        for i, (aug_image, aug_bboxes) in enumerate(zip(augmented_images, augmented_bboxes_list), 1):
            if i >= len(axes_flat):
                break
                
            ax = axes_flat[i]
            if len(aug_image.shape) == 3 and aug_image.shape[2] == 3:
                aug_rgb = cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB)
            else:
                aug_rgb = aug_image
            
            ax.imshow(aug_rgb)
            ax.set_title(f"Аугментация {i} ({len(aug_bboxes)} bboxes)")
            ax.axis('off')
            self._draw_bboxes_matplotlib(ax, aug_bboxes, aug_image.shape)
        
        # Скрываем пустые axes
        for i in range(num_images, len(axes_flat)):
            axes_flat[i].axis('off')
        
        plt.suptitle("Сравнение аугментаций", fontsize=16)
        plt.tight_layout()
        
        # Сохраняем
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.debug(f"Сетка визуализации сохранена: {output_path}")
    
    def plot_bbox_statistics(self, bbox_stats: Dict[str, Any], output_path: Optional[str] = None):
        """
        Создает графики статистики bounding boxes.
        
        Args:
            bbox_stats: Статистика bounding boxes
            output_path: Путь для сохранения графика (если None, отображает на экране)
        """
        if not bbox_stats.get('by_class'):
            logger.warning("Нет данных для построения статистики")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # График распределения по классам
        classes = list(bbox_stats['by_class'].keys())
        counts = list(bbox_stats['by_class'].values())
        
        bars = axes[0].bar(classes, counts)
        axes[0].set_xlabel('Class ID')
        axes[0].set_ylabel('Количество bounding boxes')
        axes[0].set_title('Распределение по классам')
        
        # Добавляем значения на столбцы
        for bar, count in zip(bars, counts):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        str(count), ha='center', va='bottom')
        
        # График площадей
        if 'areas' in bbox_stats and bbox_stats['areas']:
            areas = bbox_stats['areas']
            axes[1].hist(areas, bins=20, edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('Относительная площадь')
            axes[1].set_ylabel('Частота')
            axes[1].set_title('Распределение площадей bounding boxes')
            
            # Добавляем среднее значение
            avg_area = bbox_stats.get('avg_area', 0)
            axes[1].axvline(avg_area, color='red', linestyle='--', 
                           label=f'Среднее: {avg_area:.4f}')
            axes[1].legend()
        
        plt.suptitle(f"Статистика bounding boxes (всего: {bbox_stats.get('total', 0)})", fontsize=14)
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.debug(f"График статистики сохранен: {output_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # Пример использования
    import tempfile
    
    # Настройка логирования
    logging.basicConfig(level=logging.DEBUG)
    
    # Создание тестовых данных
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_bboxes = [
        BoundingBox(0, 0.5, 0.5, 0.2, 0.2),
        BoundingBox(1, 0.3, 0.3, 0.1, 0.1),
        BoundingBox(2, 0.7, 0.7, 0.15, 0.15),
    ]
    
    # Создание визуализатора с именами классов
    class_names = {0: "Person", 1: "Car", 2: "Dog"}
    visualizer = Visualizer(class_names)
    
    # Тест рисования bounding boxes
    visualized = visualizer.draw_bboxes(test_image, test_bboxes)
    
    # Сохранение визуализации
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        temp_path = f.name
        visualizer.save_visualization(test_image, test_bboxes, temp_path, "Тест визуализации")
        print(f"Визуализация сохранена: {temp_path}")
    
    # Тест сравнения (используем то же изображение для простоты)
    fig = visualizer.visualize_comparison(
        test_image, test_image, 
        test_bboxes, test_bboxes,
        "Тест сравнения"
    )
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_fig_path = f.name
        fig.savefig(temp_fig_path, dpi=150, bbox_inches='tight')
        print(f"Фигура сравнения сохранена: {temp_fig_path}")
        plt.close(fig)
    
    # Тест статистики
    stats = {
        'total': 3,
        'by_class': {0: 1, 1: 1, 2: 1},
        'areas': [0.04, 0.01, 0.0225],
        'avg_area': 0.0242
    }
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_stats_path = f.name
        visualizer.plot_bbox_statistics(stats, temp_stats_path)
        print(f"График статистики сохранен: {temp_stats_path}")
    
    # Очистка
    os.unlink(temp_path)
    os.unlink(temp_fig_path)
    os.unlink(temp_stats_path)