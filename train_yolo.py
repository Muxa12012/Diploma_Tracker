from ultralytics import YOLO
import os

# Путь к данным
data_yaml = "datasets/Pet Detection.v4i.yolo26/data.yaml"

# Загрузка предобученной модели YOLO26
model = YOLO('yolo26n.pt')  # можно заменить на 'yolov8s.pt', 'yolov8m.pt' и т.д.

if __name__ == '__main__':
    # Обучение модели
    results = model.train(
        data=data_yaml,
        epochs=50,        # количество эпох
        imgsz=640,        # размер изображения
        batch=16,         # размер батча
        name='face_detection_yolo26'  # имя модели для сохранения
    )

    print("Обучение завершено.")