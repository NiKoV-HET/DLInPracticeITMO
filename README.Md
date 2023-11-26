# Deep Learning in practice ITMO
Проект по распознаванию дорожных знаков, предмет "Глубокое обучение на практике".

# Клонирование репозитория и модулей
```bash
git clone --recurse-submodules https://github.com/NiKoV-HET/DLInPracticeITMO
```

# Установка и настройка окружения
```bash
python -m venv venv
. .\venv\Scripts\activate
pip install -r TrafficSignRecognition\requirements.txt
pip install -r requirements.txt
```

# Запуск 
```bash
python main.py
```
# Интерфейс
![interface](examples\interface.jpg)

# Пример работы
Исходный файл в "example_video.mp4"

Результат обработки:

![example_result](examples\example_result.gif)
