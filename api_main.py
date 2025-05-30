import json
import logging
import os
import shutil
import uuid
import time
import torch
import pandas as pd
from datetime import datetime
from fastapi import BackgroundTasks
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form
from pydantic import BaseModel, Field
import uvicorn

from scripts import process_documents_pipeline
from scripts import main_pipeline
from scripts.problem_analyzer import DEFAULT_PROBLEM_CATEGORIES, DEFAULT_DETAIL_EXTRACTION_PROMPT_TEMPLATE, DEFAULT_PROMPT_TEMPLATE # Импортируем константы для main_pipeline config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PipelineAPI")

app = FastAPI(title="Document Processing API")

class PreprocessRequest(BaseModel):
    input_dir_container: str = Field(..., example="/data/raw_pdfs", description="Путь внутри контейнера к директории с PDF для предобработки.")
    output_dir_container: str = Field(..., example="/data/processed_data", description="Путь внутри контейнера для сохранения результатов предобработки.")
    embedding_model_name: str = "all-mpnet-base-v2"
    chunking_strategy: str = "title"
    max_chunk_chars: Optional[int] = 2000 # Было 2500 в process_documents, но в DEFAULT_CONFIG main_pipeline 2000
    combine_under_n_chars: Optional[int] = 150 # Было 150 в process_documents, но в DEFAULT_CONFIG main_pipeline 500, потом 150
    new_after_n_chars: Optional[int] = 1500 # Было 2500 в process_documents, но в DEFAULT_CONFIG main_pipeline 1500
    embedding_batch_size: int = 32
    
class PreprocessResponse(BaseModel):
    status: str
    message: str
    processed_files_count: Optional[int] = None
    output_path: Optional[str] = None

class AnalyzeRequest(BaseModel):
    input_dir_container: str = Field(..., example="/data/raw_pdfs", description="Путь внутри контейнера к директории с исходными файлами (PDF, DOCX).")
    output_file_container: str = Field(..., example="/data/results/analysis.csv", description="Путь внутри контейнера для сохранения итогового CSV.")
    # Путь к директории с предобработанными данными (FAISS, чанки)
    processed_data_dir_container: str = Field(..., example="/data/processed_data", description="Путь внутри контейнера к предобработанным данным (FAISS индексы, чанки).")
    
    # Остальные параметры конфигурации для main_pipeline
    embedding_model: str = "all-mpnet-base-v2"
    section_classifier_path_container: str = "/app/models/section_classifier_deberta_v3" # Фиксированный путь внутри контейнера
    llm_model_name: str = "Qwen/Qwen2.5-32B-Instruct"
    llm_use_4bit: bool = True
    llm_use_8bit: bool = False
    # llm_prompt_template: str = main_pipeline.DEFAULT_PROMPT_TEMPLATE # Можно взять из импорта
    # llm_max_new_tokens: int = 100
    # llm_detail_extraction_prompt_template: str = main_pipeline.DEFAULT_DETAIL_EXTRACTION_PROMPT_TEMPLATE
    # llm_detail_extraction_max_tokens: int = 100
    # problem_categories: list = main_pipeline.DEFAULT_PROBLEM_CATEGORIES
    target_sections: Optional[Dict] = None # Если None, будет использован DEFAULT_CONFIG из main_pipeline
    retriever_k: int = 100
    min_classifier_score: float = 0.5
    chunking_strategy_main: str = Field("title", alias="chunking_strategy") # псевдоним для соответствия ключу в DEFAULT_CONFIG
    max_chunk_chars_main: int = Field(2000, alias="max_chunk_chars")
    combine_under_n_chars_main: int = Field(150, alias="combine_under_n_chars")
    new_after_n_chars_main: int = Field(1500, alias="new_after_n_chars")
    partition_kwargs_main: Dict = Field({"strategy": "hi_res", "pdf_infer_table_structure": True}, alias="partition_kwargs")
    detail_extractor_use_spacy: bool = True
    spacy_model_name: str = "en_core_web_sm"
    file_pattern: str = "*.pdf"
    device: Optional[str] = None # 'cpu' or 'cuda'
    
class AnalyzeResponse(BaseModel):
    status: str
    message: str
    output_file_path: Optional[str] = None
    
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    try:
        logger.info(f"Начало загрузки файла: {file.filename}")
        base_dir = Path("/data/uploaded_pdfs")
        logger.info(f"Базовая директория для загрузки: {base_dir}")
        
        today = datetime.now()
        date_dir = base_dir / f"{today.year}{today.month:02d}/{today.day:02d}"
        logger.info(f"Создание директории по дате: {date_dir}")
        date_dir.mkdir(parents=True, exist_ok=True)
        
        process_id = str(uuid.uuid4())
        
        session_specific_dir = date_dir / process_id
        session_specific_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Создана уникальная директория для сессии: {session_specific_dir}")
        
        target_file_path = session_specific_dir / file.filename
        logger.info(f"Сохранение файла по пути: {target_file_path}")
        
        with open(target_file_path, 'wb') as buffer:
            logger.debug("Копирование содержимого файла...")
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Файл {file.filename} успешно загружен в {target_file_path}")
        
        target_processed_data_dir = session_specific_dir / "processed_data"
        target_processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        target_results_file_path = session_specific_dir / f"results_{file.filename.split('.')[0]}_{process_id}.json"
        target_status_file_path = session_specific_dir / f"status_{process_id}.json"
        
        with open(target_status_file_path, 'w', encoding='utf-8') as f:
            json.dump({
                "process_id": process_id,
                "status": "processing",
                "file": file.filename,
                "original_path": str(target_file_path),
                "upload_time": today.isoformat(),
                "processing_complete": False,
                "results_file": None,
                "error": None
            }, f, ensure_ascii=False, indent=2)
            
        background_tasks.add_task(
            process_document_background,
            file_path_bg=str(target_file_path),
            session_dir_bg=str(session_specific_dir),
            processed_data_dir_bg=str(target_processed_data_dir),
            results_file_bg=str(target_results_file_path),
            status_file_bg=str(target_status_file_path),
            process_id_bg=process_id
        )
        
        return {
            'status': "uploaded", 
            'message': f"File uploaded successfully. Processing started in background.",
            'file_path': str(target_file_path),
            'directory': str(session_specific_dir),
            'process_id': process_id,
            'status_file': str(target_status_file_path)
        }
    
    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        if 'target_status_file_path' in locals() and Path(target_status_file_path).exists():
            try:
                update_status(str(target_status_file_path), {
                    "status": "error",
                    "processing_complete": True, # Обработка завершена (с ошибкой)
                    "error": f"Error during upload: {str(e)}"
                })
            except Exception as ue:
                logger.error(f"Failed to update status file with upload error: {ue}")
        raise HTTPException(status_code=500, detail=str(e))
    
async def process_document_background(
    file_path_bg: str,
    session_dir_bg: str,
    processed_data_dir_bg: str,
    results_file_bg: str,
    status_file_bg: str,
    process_id_bg: str
):
    try:
        logger.info(f"[Process {process_id_bg}] Начало фоновой обработки для {file_path_bg}")
        
        update_status(status_file_bg, {"status": "preprocessing"})
        
        success_count = process_documents_pipeline.process_documents(
            input_dir=session_dir_bg,
            output_dir=processed_data_dir_bg,
            embedding_model_name="all-mpnet-base-v2",
            chunking_strategy="title",
            max_chunk_chars=2000,
            combine_under_n_chars=150,
            new_after_n_chars=1500,
            embedding_batch_size=32,
            partition_kwargs={
                "strategy": "hi_res",
                "pdf_infer_table_structure": True    
            },
            # file_pattern=Path(file_path_bg).name # Чтобы обработать только этот файл
        )
        
        logger.info(f"[Process {process_id_bg}] Предобработка завершена. Обработано файлов: {success_count}")
        
        if success_count > 0:
            update_status(status_file_bg, {"status": "analyzing"})
            
            logger.info(f"[Process {process_id_bg}] Запуск анализа документов")
            
            current_pipeline_config = {
                "embedding_model": "all-mpnet-base-v2",
                "section_classifier_path": "/app/models/section_classifier_deberta_v3",
                "llm_model_name": "Qwen/Qwen2.5-32B-Instruct",
                "llm_use_4bit": True,
                "llm_use_8bit": False,
                "llm_prompt_template": DEFAULT_PROMPT_TEMPLATE,
                "llm_max_new_tokens": 100,
                "llm_detail_extraction_prompt_template": DEFAULT_DETAIL_EXTRACTION_PROMPT_TEMPLATE,
                "llm_detail_extraction_max_tokens": 100,
                "problem_categories": DEFAULT_PROBLEM_CATEGORIES,
                "target_sections": {
                    "comments": [
                        "Operations Summary", "Management Summary", "Current Operations",
                        "Planned Operations", "Comments:", "Summary of Issues:", "Problems Encountered:"
                    ],
                    "well_information": [
                        "Well:", "Well ID:", "Well Name:", "Well Data:",
                        "REPORT", "REPORT N°", "DAILY DRILLING REPORT"
                    ]
                },
                "retriever_k": 100,
                "min_classifier_score": 0.5,
                "chunking_strategy": "title",
                "max_chunk_chars": 2000,
                "combine_under_n_chars": 150,
                "new_after_n_chars": 1500,
                "partition_kwargs": {
                    "strategy": "hi_res",
                    "pdf_infer_table_structure": True
                },
                "detail_extractor_use_spacy": True,
                "spacy_model_name": "en_core_web_sm",
                "file_pattern": Path(file_path_bg).name # Анализируем только загруженный файл
            }
            
            # Временный CSV файл сохраняем также в директорию сессии
            temp_csv = Path(session_dir_bg) / f"temp_analysis_{process_id_bg}.csv"
            main_pipeline.run_pipeline(
                input_dir=session_dir_bg, # Директория, где лежит исходный PDF
                output_file=str(temp_csv), # Путь для временного CSV
                processed_data_path=processed_data_dir_bg, # Путь к предобработанным данным (FAISS, чанки)
                config=current_pipeline_config
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"[Process {process_id_bg}] GPU cache cleared after analysis.")
            
            if temp_csv.exists():
                df = pd.read_csv(temp_csv)
                # Сохраняем в JSON файл по пути results_file_bg
                with open(results_file_bg, 'w', encoding='utf-8') as f:
                    df.to_json(f, orient="records", indent=2)
                
                logger.info(f"[Process {process_id_bg}] Результаты анализа сохранены в JSON: {results_file_bg}")
                
                update_status(status_file_bg, {
                    "status": "completed",
                    "processing_complete": True,
                    "results_file": results_file_bg, # Обновляем на полный путь
                    "completion_time": datetime.now().isoformat()
                })
            else:
                logger.warning(f"[Process {process_id_bg}] CSV файл с результатами анализа не был создан: {temp_csv}")
                update_status(status_file_bg, {
                    "status": "error",
                    "processing_complete": True,
                    "error": f"CSV результат не был создан после анализа ({temp_csv})"
                })
            
        else:
            logger.warning(f"[Process {process_id_bg}] Предобработка не выполнена успешно (файлов: {success_count}), анализ пропущен")
            update_status(status_file_bg, {
                "status": "error",
                "processing_complete": True,
                "error": "Предобработка не выполнена успешно или файл не найден/не обработан"
            })
            
    except Exception as e:
        logger.error(f"[Process {process_id_bg}] Ошибка при фоновой обработке: {e}", exc_info=True)
        update_status(status_file_bg, {
            "status": "error",
            "processing_complete": True,
            "error": str(e)
        })
        
        
def update_status(status_file_path, updates):
    """Update the status file with new information"""
    try:
        with open(status_file_path, 'r', encoding='utf-8') as f:
            status_data = json.load(f)
            
        status_data.update(updates)
        
        if updates.get('processing_complete') and updates.get('results_file'):
            try:
                results_file_path = updates['results_file']
                with open(results_file_path, 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
                
                status_data['results'] = results_data
            except Exception as e:
                logger.error(f"Error loading results file content: {e}")
                status_data["results_error"] = str(e)
        
        with open(status_file_path, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"Ошибка при обновлении статуса: {e}", exc_info=True)
        
        
@app.get("/processing_status/{process_id}")
async def get_processing_status(process_id: str):
    """
    Check the status of a document processing job
    """
    try:
        base_dir = Path("/data/uploaded_pdfs")
        status_files = list(base_dir.glob(f"**/status_{process_id}.json"))
        
        if not status_files:
            raise HTTPException(status_code=404, detail=f"Processing job with ID {process_id} not found")
        
        status_file = status_files[0]
        
        with open(status_file, 'r', encoding='utf-8') as f:
            status_data = json.load(f)
            
        if status_data.get("processing_complete") and status_data.get("results_file"):
            try:
                with open(status_data["results_file"], 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
                status_data['results'] = results_data
                logger.info(f"Results file loaded successfully: {status_data['results']}")
                logger.info(f"Results file path: {status_data}")
            except Exception as e:
                logger.error(f"Error reading results file: {e}")
                status_data["results_error"] = str(e)
                
        return status_data
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking processing status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/results/{process_id}")
async def get_process_results(process_id: str):
    """
    Получить только результаты обработки документа без метаданных о статусе обработки
    """
    try:
        base_dir = Path("/data/uploaded_pdfs")
        status_files = list(base_dir.glob(f"**/status_{process_id}.json"))
        
        if not status_files:
            raise HTTPException(status_code=404, detail=f"Processing job with ID {process_id} not found")
        
        status_file = status_files[0]
        
        with open(status_file, 'r', encoding='utf-8') as f:
            status_data = json.load(f)
            
        if not status_data.get("processing_complete"):
            return {"status": "processing", "message": "Processing is not complete yet"}
        
        if not status_data.get("results_file"):
            return {"status": "error", "message": "No results file available"}
        
        try:
            with open(status_data["results_file"], 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            return results_data
        except Exception as e:
            logger.error(f"Error reading results file: {e}")
            raise HTTPException(status_code=500, detail=f"Error reading results file: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving process results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    

# Добавить в существующий middleware для более детального логирования
@app.middleware("http")
async def log_requests(request, call_next):
    request_id = str(uuid.uuid4())[:8]  # Короткий ID для идентификации запроса
    
    # Логирование заголовков запроса
    headers_str = "\n".join([f"    {k}: {v}" for k, v in request.headers.items()])
    body = None
    
    # Попытка получить тело запроса (если есть)
    try:
        if request.method in ["POST", "PUT"]:
            body = await request.body()
            if body:
                try:
                    # Пробуем форматировать как JSON для лучшей читаемости
                    body_str = json.loads(body)
                    body = json.dumps(body_str, indent=2, ensure_ascii=False)
                except:
                    body = str(body)
    except Exception as e:
        logger.warning(f"Не удалось прочитать тело запроса: {e}")
    
    logger.info(f"[{request_id}] Запрос: {request.method} {request.url}")
    logger.info(f"[{request_id}] Заголовки:\n{headers_str}")
    if body:
        logger.info(f"[{request_id}] Тело запроса:\n{body}")
    
    # Выполнение запроса
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Логирование ответа
    logger.info(f"[{request_id}] Ответ: {response.status_code}, Время обработки: {process_time:.4f} сек")
    
    # Добавление заголовка X-Process-Time в ответ
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id
    
    return response

# Добавить новый эндпоинт для тестирования и проверки работоспособности API
@app.get("/health")
async def health_check():
    """
    Простой эндпоинт для проверки работоспособности API.
    Возвращает статус 200 OK, если API работает корректно.
    """
    logger.info("Запрос к /health")
    return {"status": "ok", "message": "API работает корректно"}

# Дополнительный эндпоинт, который возвращает основную информацию об API
@app.get("/")
async def root():
    """
    Корневой эндпоинт - возвращает основную информацию об API.
    """
    logger.info("Запрос к корневому эндпоинту /")
    return {
        "api_name": "Document Processing API",
        "version": "1.0",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Основная информация об API"},
            {"path": "/health", "method": "GET", "description": "Проверка работоспособности API"},
            {"path": "/upload_pdf", "method": "POST", "description": "Загрузка PDF-файла"},
            {"path": "/preprocess_documents", "method": "POST", "description": "Предобработка документов"},
            {"path": "/analyze_documents", "method": "POST", "description": "Анализ документов"}
        ],
        "docs_url": "/docs",  # URL для OpenAPI документации (Swagger UI)
        "redoc_url": "/redoc"  # URL для ReDoc документации
    }
    
    
@app.post("/preprocess_documents", response_model=PreprocessResponse)
async def preprocess_document_endpoint(req: PreprocessRequest = Body(...)):
    logger.info(f"Запрос на /preprocess_documents: {req.model_dump_json(indent=2)}")
    try:
        Path(req.input_dir_container).mkdir(parents=True, exist_ok=True)
        Path(req.output_dir_container).mkdir(parents=True, exist_ok=True)
        
        partition_opts = {
            # "strategy": "hi_res",
            # "pdf_infer_table_structure": True    
        }
        
        success_count = process_documents_pipeline.process_documents(
            input_dir=req.input_dir_container,
            output_dir=req.output_dir_container,
            embedding_model_name=req.embedding_model_name,
            chunking_strategy=req.chunking_strategy,
            max_chunk_chars=req.max_chunk_chars,
            combine_under_n_chars=req.combine_under_n_chars,
            new_after_n_chars=req.new_after_n_chars,
            embedding_batch_size=req.embedding_batch_size,
            device=req.device,
            partition_kwargs=partition_opts,
            # save_intermediate_raw_chunks=req.save_intermediate_raw_chunks
        )
        
        if success_count > 0:
            return PreprocessResponse(
                status='success',
                message=f"Предобработка завершена. Успешно обработано файлов: {success_count}",
                processed_files_count=success_count,
                output_path=req.output_dir_container
            )
        else:
            return PreprocessResponse(
                status='partial_failure_or_no_files',
                message="Предобработка завершена, но не все файлы могли быть успешно обработаны или файлы не найдены.",
                processed_files_count=success_count if isinstance(success_count, int) else 0,
                output_path=req.output_dir_container
            )
            
    except Exception as e:
        logger.error(f"Ошибка в /preprocess_documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/analyze_documents", response_model=AnalyzeResponse)
async def analyze_documents_endpoint(req: AnalyzeRequest = Body(...)):
    logger.info(f"Запрос на /analyze_documents: {req.model_dump_json(indent=2)}")
    try:
        current_pipeline_config = {
            "embedding_model": req.embedding_model,
            # "processed_data_dir": Это теперь отдельный аргумент run_pipeline
            "section_classifier_path": req.section_classifier_path_container, # Фиксированный путь внутри контейнера
            "llm_model_name": req.llm_model_name,
            "llm_use_4bit": req.llm_use_4bit,
            "llm_use_8bit": req.llm_use_8bit,
            "llm_prompt_template": DEFAULT_PROMPT_TEMPLATE, # main_pipeline.DEFAULT_PROMPT_TEMPLATE,
            "llm_max_new_tokens": 100, # или req.llm_max_new_tokens если хотите сделать параметром
            "llm_detail_extraction_prompt_template": DEFAULT_DETAIL_EXTRACTION_PROMPT_TEMPLATE, # main_pipeline.DEFAULT_DETAIL_EXTRACTION_PROMPT_TEMPLATE,
            "llm_detail_extraction_max_tokens": 100, # или req.llm_detail_extraction_max_tokens
            "problem_categories": DEFAULT_PROBLEM_CATEGORIES, # main_pipeline.DEFAULT_PROBLEM_CATEGORIES,
            "target_sections": req.target_sections if req.target_sections else { # Используем из запроса или дефолтные
                "comments": [
                    "Operations Summary", "Management Summary", "Current Operations",
                    "Planned Operations", "Comments:", "Summary of Issues:", "Problems Encountered:"
                ],
                "well_information": [
                    "Well:", "Well ID:", "Well Name:", "Well Data:",
                    "REPORT", "REPORT N°", "DAILY DRILLING REPORT"
                ]
            },
            "retriever_k": req.retriever_k,
            "min_classifier_score": req.min_classifier_score,
            "chunking_strategy": req.chunking_strategy_main,
            "max_chunk_chars": req.max_chunk_chars_main,
            "combine_under_n_chars": req.combine_under_n_chars_main,
            "new_after_n_chars": req.new_after_n_chars_main,
            "partition_kwargs": req.partition_kwargs_main,
            "detail_extractor_use_spacy": req.detail_extractor_use_spacy,
            "spacy_model_name": req.spacy_model_name,
            "file_pattern": req.file_pattern,
            "device": req.device
        }
        
        Path(req.input_dir_container).mkdir(parents=True, exist_ok=True)
        Path(req.output_file_container).parent.mkdir(parents=True, exist_ok=True)
        Path(req.processed_data_dir_container).mkdir(parents=True, exist_ok=True)
        Path(req.section_classifier_path_container).parent.mkdir(parents=True, exist_ok=True)
        
        main_pipeline.run_pipeline(
            input_dir=req.input_dir_container,
            output_file=req.output_file_container,
            processed_data_path=req.processed_data_dir_container,
            config=current_pipeline_config
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"GPU cache cleared after analysis in analyze_documents_endpoint for input {req.input_dir_container}.")
            
        return AnalyzeResponse(
            status='success',
            message=f"Анализ документов завершен. Результаты в {req.output_file_container}",
            output_file_path=req.output_file_container
        )
    except Exception as e:
        logger.error(f"Ошибка в /analyze_documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8100))
    uvicorn.run(app, host="0.0.0.0", port=port)