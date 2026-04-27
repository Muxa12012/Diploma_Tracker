import os

# Укажите путь к папке с файлами .txt
folder_paths = ["datasets/Face_Detection_v25i_yolo26/test/labels","datasets/Face_Detection_v25i_yolo26/valid/labels","datasets/Face_Detection_v25i_yolo26/train/labels"]

# Проходим по всем файлам в папках
for folder_path in folder_paths:
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            # Читаем содержимое файла
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            # Обрабатываем строки: заменяем "0 " на "11 " только в начале строки
            new_lines = []
            for line in lines:
                if line.startswith("11 "):
                    new_lines.append("10" + line[2:])  # Заменяем только первый символ '0' на '11'
                else:
                    new_lines.append(line)
            
            # Перезаписываем файл с изменениями
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(new_lines)
            
            print(f"Обработан файл: {filename}")

print("Замена завершена для всех файлов.")