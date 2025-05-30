import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from document_processor import load_and_chunk
    from embedding_generator import generate_embeddings, save_embeddings, load_embeddings
    from vector_store import build_and_save_faiss_index
except ImportError as e:
    print(f"Ошибка импорта: {e}. Убедитесь, что файлы document_processor.py, "
          "embedding_generator.py и vector_store.py доступны.")
    exit(1)
    
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger("DocumentProcessingPipeline")

def process_documents(
    input_dir: str,
    output_dir: str,
    embedding_model_name: str='all-mpnet-base-v2',
    chunking_strategy: str="title",
    max_chunk_chars: Optional[int]=2500,
    combine_under_n_chars: Optional[int]=150,
    new_after_n_chars: Optional[int]=2500,
    embedding_batch_size: int=32,
    device: Optional[str]=None,
    partition_kwargs: Optional[Dict]=None,
    save_intermediate_raw_chunks: bool=False,
):
    """
    Обрабатывает все PDF-файлы в input_dir, генерирует эмбеддинги и
    создает единый FAISS индекс и файл с ID чанков в output_dir.

    Args:
        input_dir: Путь к папке с входными PDF файлами.
        output_dir: Путь к папке для сохранения результатов (индекс, ID чанков).
        embedding_model_name: Название модели Sentence Transformer.
        chunking_strategy: Стратегия чанкинга для load_and_chunk.
        max_chunk_chars: Параметр для load_and_chunk.
        combine_under_n_chars: Параметр для load_and_chunk.
        new_after_n_chars: Параметр для load_and_chunk.
        embedding_batch_size: Размер батча для generate_embeddings.
        device: Устройство для generate_embeddings ('cuda', 'cpu', None).
        partition_kwargs: Дополнительные аргументы для unstructured.partition.
        save_intermediate: Если True, сохраняет чанки и эмбеддинги для каждого PDF отдельно.

    Returns:
        True, если пайплайн успешно завершился и индекс создан, иначе False.
    """
    logger.info(f"--- Начало пайплайна --- {max_chunk_chars=}, {combine_under_n_chars=}, {new_after_n_chars=}")
    start_time = time.time()
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if partition_kwargs is None:
        partition_kwargs = {}
        
    pdf_files = list(input_path.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"В папке '{input_dir}' не найдено PDF файлов.")
        return False
    
    logger.info(f"Найдено {len(pdf_files)} PDF файлов для обработки в '{input_dir}'.")
    
    processed_files_count = 0
    successful_files_count = 0
    
    for pdf_path in pdf_files:
        logger.info(f"Обработка файла: {pdf_path.name}")
        
        doc_base_name = pdf_path.stem
        doc_output_index_path = output_path / f"{doc_base_name}.index"
        doc_output_chunks_path = output_path / f"{doc_base_name}_chunks.json"
        
        logger.info(f"1. Загрузка и чанкинг...")
        try:
            chunks = load_and_chunk(
                file_path=str(pdf_path),
                chunking_strategy=chunking_strategy,
                max_chunk_chars=max_chunk_chars,
                combine_under_n_chars=combine_under_n_chars,
                new_after_n_chars=new_after_n_chars,
                **partition_kwargs
            )
        except Exception as e:
            logger.error(f"Ошибка при загрузке/чанкинге файла {pdf_path.name}: {e}", exc_info=True)
            continue 
        
        if not chunks:
            logger.warning(f"Не удалось извлечь или создать чанки для {pdf_path.name}. Файл пропущен.")
            continue
        
        logger.info(f"   Получено {len(chunks)} чанков.")
        
        try:
            with open(doc_output_chunks_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
            logger.info(f"   Файл с чанками (ID+текст) сохранен в {doc_output_chunks_path}")
        except Exception as e:
            logger.error(f"   Не удалось сохранить файл чанков для {pdf_path.name}: {e}")
            continue
        
        if save_intermediate_raw_chunks:
            intermediate_chunks_dir = output_path / "intermediate_raw_chunks"
            intermediate_chunks_dir.mkdir(parents=True, exist_ok=True)
            chunk_file_path = intermediate_chunks_dir / f"{doc_base_name}_raw_chunks.json"
            try:
                with open(chunk_file_path, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=4)
                logger.info(f"   Промежуточные сырые чанки сохранены в {chunk_file_path}")
            except Exception as e:
                logger.error(f"   Не удалось сохранить промежуточные сырые чанки: {e}")
            
        logger.info(f"2. Генерация эмбеддингов (модель: {embedding_model_name})...")
        try:
            embeddings, chunk_ids = generate_embeddings(
                chunks=chunks,
                model_name=embedding_model_name,
                device=device,
                batch_size=embedding_batch_size
            )
        except Exception as e:
             logger.error(f"Ошибка при генерации эмбеддингов для {pdf_path.name}: {e}", exc_info=True)
             continue
        
        if embeddings is None or chunk_ids is None:
            logger.warning(f"Не удалось сгенерировать эмбеддинги для {pdf_path.name}. Файл пропущен.")
            continue
        
        logger.info(f"   Сгенерировано {embeddings.shape[0]} эмбеддингов размерности {embeddings.shape[1]}.")
        
        original_chunk_ids_from_chunks = [c['chunk_id'] for c in chunks]
        if set(chunk_ids) != set(original_chunk_ids_from_chunks):
             logger.error(f"Критическое несоответствие ID чанков из generate_embeddings и load_and_chunk для {pdf_path.name}. Пропускаем.")
             logger.debug(f"IDs из generate_embeddings: {chunk_ids}")
             logger.debug(f"IDs из load_and_chunk: {original_chunk_ids_from_chunks}")
             continue
         
        if len(chunk_ids) != embeddings.shape[0]:
            logger.error(f"Несоответствие числа эмбеддингов ({embeddings.shape[0]}) и ID чанков ({len(chunk_ids)}) для {pdf_path.name}. Пропускаем.")
            continue
        
        logger.info(f"3. Создание и сохранение индекса FAISS в: {doc_output_index_path}...")
        try:
            success = build_and_save_faiss_index(embeddings, str(doc_output_index_path))
            if not success:
                logger.error(f"   Не удалось создать или сохранить индекс FAISS для {pdf_path.name}.")
                continue
            else:
                logger.info(f"   Индекс FAISS успешно сохранен.")
                successful_files_count += 1
        except Exception as e:
             logger.error(f"   Ошибка при создании/сохранении FAISS индекса для {pdf_path.name}: {e}", exc_info=True)
             continue
         
        processed_files_count += 1
        logger.info(f"--- Обработка файла {pdf_path.name} завершена ---")

        
    end_time = time.time()
    logger.info(f"--- Пайплайн завершен за {end_time - start_time:.2f} секунд ---")
    logger.info(f"Всего найдено PDF файлов: {len(pdf_files)}")
    # logger.info(f"Попыток обработки: {processed_files_count}") # processed_files_count не отражает успешность
    logger.info(f"Успешно создано индексов/файлов чанков: {successful_files_count}")
    logger.info(f"Результаты сохранены в папке: {output_path}")
    
    return successful_files_count > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Пайплайн обработки PDF документов: чанкинг, эмбеддинги, FAISS индекс.")
    parser.add_argument("input_dir", help="Путь к папке с входными PDF файлами.")
    parser.add_argument("output_dir", help="Путь к папке для сохранения результатов (индекс, ID чанков).")
    parser.add_argument("--model", default="all-mpnet-base-v2", help="Название модели Sentence Transformer.")
    parser.add_argument("--chunking_strategy", default="title", choices=["title", "basic", "none"], help="Стратегия чанкинга.")
    parser.add_argument("--max_chars", type=int, default=2000, help="Макс. символов в чанке (для 'title').")
    parser.add_argument("--combine_chars", type=int, default=500, help="Порог объединения символов (для 'title').")
    parser.add_argument("--new_after_chars", type=int, default=1500, help="Новый чанк после N символов (для 'title').")
    parser.add_argument("--batch_size", type=int, default=32, help="Размер батча для генерации эмбеддингов.")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"], help="Устройство для эмбеддингов (cuda/cpu).")
    parser.add_argument("--save_intermediate_raw_chunks", action='store_true', help="Сохранять промежуточные чанки и эмбеддинги для каждого файла.")
    # parser.add_argument("--strategy", default="fast", help="Стратегия парсинга unstructured (например, 'fast', 'hi_res')")

    args = parser.parse_args()

    # Сборка partition_kwargs (пример, если бы добавили --strategy)
    partition_options = {
        #  "strategy": "hi_res",
        #  "pdf_infer_table_structure": True
    }
    # if args.strategy:
    #     partition_options['strategy'] = args.strategy
    
    process_documents(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        embedding_model_name=args.model,
        chunking_strategy=args.chunking_strategy,
        max_chunk_chars=args.max_chars,
        combine_under_n_chars=args.combine_chars,
        new_after_n_chars=args.new_after_chars,
        embedding_batch_size=args.batch_size,
        device=args.device,
        partition_kwargs=partition_options,
        save_intermediate_raw_chunks=args.save_intermediate_raw_chunks
    )