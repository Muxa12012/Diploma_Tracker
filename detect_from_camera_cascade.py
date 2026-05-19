#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Каскадное обнаружение объектов с использованием двух моделей YOLO:
1. Targeted модель обнаруживает целевой объект (например, конкретное лицо).
2. General модель обнаруживает все остальные объекты, исключая области, уже обнаруженные targeted моделью.
Обнаружения не пересекаются.
"""
import sys
import locale
import io
import cv2
from ultralytics import YOLO
import time

# Настройка кодировки вывода для поддержки кириллицы в Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    # Установка локали для корректного отображения
    locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')
import numpy as np

def calculate_iou(box1, box2):
    """
    Вычисляет Intersection over Union (IoU) между двумя bounding boxes.
    box1, box2: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    if union == 0:
        return 0.0
    return intersection / union

def filter_overlapping_boxes(target_boxes, other_boxes, iou_threshold=0.3):
    """
    Фильтрует other_boxes, удаляя те, которые пересекаются с target_boxes (IoU > threshold).
    Возвращает индексы боксов, которые не пересекаются.
    """
    keep_indices = []
    for i, other_box in enumerate(other_boxes):
        overlap = False
        for target_box in target_boxes:
            iou = calculate_iou(target_box, other_box)
            if iou > iou_threshold:
                overlap = True
                break
        if not overlap:
            keep_indices.append(i)
    return keep_indices

def main():
    # Пути к моделям
    targeted_model_path = "runs/detect/targeted_yolo/targeted_yolo_model/weights/best.pt"
    general_model_path = "best_one_yet/weights/best.pt"
    
    # Пороги уверенности
    targeted_conf = 0.5
    general_conf = 0.5
    
    # Загрузка моделей
    print("Загрузка targeted модели...")
    targeted_model = YOLO(targeted_model_path)
    print("Загрузка general модели...")
    general_model = YOLO(general_model_path)
    print("Модели загружены.")
    
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
        
        # 1. Детекция targeted моделью
        targeted_results = targeted_model(frame, conf=targeted_conf, verbose=False)
        targeted_boxes = []
        targeted_classes = []
        targeted_confidences = []
        if len(targeted_results) > 0:
            boxes = targeted_results[0].boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    targeted_boxes.append([x1, y1, x2, y2])
                    targeted_classes.append(int(box.cls[0].cpu().numpy()))
                    targeted_confidences.append(float(box.conf[0].cpu().numpy()))
        
        # 2. Детекция general моделью
        general_results = general_model(frame, conf=general_conf, verbose=False)
        general_boxes = []
        general_classes = []
        general_confidences = []
        if len(general_results) > 0:
            boxes = general_results[0].boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    general_boxes.append([x1, y1, x2, y2])
                    general_classes.append(int(box.cls[0].cpu().numpy()))
                    general_confidences.append(float(box.conf[0].cpu().numpy()))
        
        # 3. Фильтрация пересекающихся боксов
        keep_indices = filter_overlapping_boxes(targeted_boxes, general_boxes)
        filtered_boxes = [general_boxes[i] for i in keep_indices]
        filtered_classes = [general_classes[i] for i in keep_indices]
        filtered_confidences = [general_confidences[i] for i in keep_indices]
        
        # 4. Отрисовка targeted детекций (синим цветом)
        for box, cls, conf in zip(targeted_boxes, targeted_classes, targeted_confidences):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # синий
            class_name = targeted_model.names[cls]
            label = f"targeted: {class_name}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 5. Отрисовка filtered general детекций (зеленым цветом)
        for box, cls, conf in zip(filtered_boxes, filtered_classes, filtered_confidences):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # зеленый
            class_name = general_model.names[cls]
            label = f"general: {class_name}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Расчет FPS
        current_frame_time = time.time()
        fps = 1 / (current_frame_time - prev_frame_time)
        prev_frame_time = current_frame_time
        fps_text = f'FPS: {fps:.2f}'
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Отображение кадра
        cv2.imshow('Cascade Detection: Targeted (Blue) + General (Green)', frame)
        
        # Выход по нажатию 'q' или 'й'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('й'):
            break
    
    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()