import cv2
from ultralytics import YOLO
import time

# Загрузка обученной модели
model = YOLO("best_one_yet/weights/best.pt")  # путь к лучшей модели после обучения

model2 = YOLO("yolo26n.pt")

# Захват с веб-камеры
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: не удалось открыть камеру.")
    exit()

# Инициализация переменных для расчета FPS
prev_frame_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось захватить кадр.")
        break

    # Выполнение детекции
    results = model(frame)

    # Отрисовка результатов
    annotated_frame = results[0].plot()  # отрисовка ограничивающих рамок и меток
    print(results)

    # Расчет FPS
    current_frame_time = time.time()
    fps = 1 / (current_frame_time - prev_frame_time)
    prev_frame_time = current_frame_time

    # Округление FPS до 2 знаков после запятой
    fps_text = f'FPS: {fps:.2f}'

    # Добавляем текст с FPS в левый верхний угол
    cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Отображение кадра
    cv2.imshow('YOLO26 Face Detection', annotated_frame)

    # Получаем код нажатой клавиши
    key = cv2.waitKey(1) & 0xFF
    # Выход по нажатию 'q' или кириллической 'й'
    if key == ord('q') or key == ord('й'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()