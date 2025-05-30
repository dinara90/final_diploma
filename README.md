# DDR Analyzer

Система автоматического анализа буровых отчетов (Daily Drilling Reports) с использованием машинного обучения и обработки естественного языка.

## Возможности

- 📄 Автоматическая обработка PDF документов
- 🧠 Анализ проблем с помощью LLM
- 🔍 Извлечение информации о скважинах
- 🗂️ Классификация секций документов
- 🌐 REST API для интеграции

## Технологии

- FastAPI
- PyTorch & Transformers
- FAISS Vector Search
- Docker & Docker Compose
- spaCy NLP

## Быстрый старт

```bash
docker-compose up --build
```

API будет доступен по адресу: http://localhost:8100
