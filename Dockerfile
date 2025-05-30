# Используем официальный образ Python
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Устанавливаем Python 3.10, pip и системные зависимости для обработки PDF и OpenGL
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-dev \
    build-essential libpq-dev curl \
    libgl1-mesa-glx libglib2.0-0 \
    poppler-utils tesseract-ocr \
    ghostscript \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Делаем python3.10 и pip3 доступными как python и pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Обновляем pip и устанавливаем зависимости Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "unstructured[pdf]" pdf2image opencv-python-headless

# Загружаем языковую модель для spaCy
RUN python -m spacy download en_core_web_sm

# Копируем директорию со скриптами и API в контейнер
COPY ./scripts /app/scripts
COPY api_main.py /app/api_main.py

ENV PYTHONPATH "${PYTHONPATH:-}:${PYTHONPATH:+:}/app/scripts"

# Создаем директории для монтирования данных и моделей
RUN mkdir -p /data/raw_pdfs && \
    mkdir -p /data/processed_data && \
    mkdir -p /data/results && \
    mkdir -p /app/models/section_classifier_deberta_v3 && \
    mkdir -p /app/config \
    mkdir -p /data/uploaded_pdfs

# Устанавливаем переменные окружения для кэша Hugging Face (опционально, но полезно)
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_CACHE=/app/.cache/huggingface/hub
RUN mkdir -p /app/.cache/huggingface/hub

# Открываем порт, на котором будет работать API
EXPOSE 8100

# Команда для запуска API-сервера Uvicorn
CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8100", "--reload"]