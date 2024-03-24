# Используем базовый образ Python
FROM python:3.12-bookworm

# Установка зависимостей системных библиотек
RUN apt-get update && \
    apt-get install -y libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Копируем файл requirements.txt и устанавливаем зависимости
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
# Копируем все файлы из текущего каталога в контейнер
COPY . /app

# Задаем рабочую директорию
WORKDIR /app
EXPOSE 8501
# Команда для запуска приложения Streamlit
CMD ["streamlit", "run", "voxy.py"]

