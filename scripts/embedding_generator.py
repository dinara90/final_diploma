import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_embeddings(chunks: List[Dict],
                        model_name: str='all-mpnet-base-v2',
                        device: Optional[str]=None,
                        batch_size: int=32) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """
    Генерирует эмбеддинги для списка текстовых чанков с помощью модели Sentence Transformer.

    Args:
        chunks: Список словарей, где каждый словарь представляет чанк и должен
                содержать как минимум ключи 'chunk_id' и 'text'.
        model_name: Название или путь к модели Sentence Transformer (например, 'all-mpnet-base-v2').
        device: Устройство для вычислений ('cuda', 'cpu', или None для автоопределения).
        batch_size: Размер батча для обработки моделью.

    Returns:
        Кортеж из двух элементов:
        - Numpy массив с эмбеддингами (размер N x D, где N - кол-во чанков, D - размерность эмбеддинга).
          None в случае ошибки.
        - Список соответствующих 'chunk_id'. None в случае ошибки.
    """
    if not chunks:
        logger.warning("Список чанков пуст. Эмбеддинги не сгенерированы.")
        return None, None
    
    if not all('chunk_id' in chunk and 'text' in chunk for chunk in chunks):
        logger.error("Некоторые чанки не содержат обязательных ключей 'chunk_id' или 'text'.")
        return None, None
    
    texts = [chunk['text'] for chunk in chunks]
    chunk_ids = [chunk['chunk_id'] for chunk in chunks]
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Используемое устройство для генерации эмбеддингов: {device}")
    
    try:
        logger.info(f"Загрузка модели Sentence Transformer: {model_name}")
        model = SentenceTransformer(model_name, device=device)
        logger.info(f"Модель {model_name} успешно загружена.")
        
        logger.info(f"Начало генерации эмбеддингов для {len(texts)} чанков (batch_size={batch_size})...")
        embeddings = model.encode(texts,
                                  batch_size=batch_size,
                                  show_progress_bar=True,
                                  convert_to_numpy=True)
        logger.info(f"Генерация эмбеддингов завершена. Размерность: {embeddings.shape}")
        
        return embeddings, chunk_ids
    except Exception as e:
        logger.error(f"Ошибка при генерации эмбеддингов: {e}", exc_info=True)
        return None, None
    
def save_embeddings(embeddings: np.ndarray,
                    chunk_ids: List[str],
                    output_dir: str,
                    base_filename: str):
    """
    Сохраняет массив эмбеддингов и список ID чанков в указанную директорию.

    Args:
        embeddings: Numpy массив эмбеддингов.
        chunk_ids: Список соответствующих ID чанков.
        output_dir: Директория для сохранения файлов.
        base_filename: Базовое имя файла (обычно ID документа), без расширения.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    embeddings_file = output_path / f"{base_filename}_embeddings.npy"
    ids_file = output_path / f"{base_filename}_chunk_ids.json"
    
    try:
        np.save(embeddings_file, embeddings)
        logger.info(f"Эмбеддинги сохранены в: {embeddings_file}")
        
        with open(ids_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_ids, f)
        logger.info(f"ID чанков сохранены в: {ids_file}")
    except IOError as e:
        logger.error(f"Ошибка при сохранении эмбеддингов или ID для {base_filename}: {e}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении эмбеддингов или ID для {base_filename}: {e}")

def load_embeddings(output_dir: str,
                    base_filename: str) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """
    Загружает массив эмбеддингов и список ID чанков из указанной директории.

    Args:
        output_dir: Директория, из которой загружать файлы.
        base_filename: Базовое имя файла (обычно ID документа), без расширения.

    Returns:
        Кортеж (embeddings, chunk_ids). None для соответствующего элемента, если файл не найден или произошла ошибка.
    """
    output_path = Path(output_dir)
    embeddings_file = output_path / f"{base_filename}_embeddings.npy"
    ids_file = output_path / f"{base_filename}_chunk_ids.json"
    
    loaded_embeddings = None
    loaded_ids = None
    
    if embeddings_file.is_file:
        try:
            loaded_embeddings = np.load(embeddings_file)
            logger.info(f"Эмбеддинги загружены из: {embeddings_file} (shape: {loaded_embeddings.shape})")
        except Exception as e:
            logger.error(f"Ошибка при загрузке эмбеддингов из {embeddings_file}: {e}")
    else:
        logger.warning(f"Файл эмбеддингов не найден: {embeddings_file}")
        
    if ids_file.is_file():
        try:
            with open(ids_file, 'r', encoding='utf-8') as f:
                loaded_ids = json.load(f)
            logger.info(f"ID чанков загружены из: {ids_file} (count: {len(loaded_ids)})")
        except Exception as e:
            logger.error(f"Ошибка при загрузке ID чанков из {ids_file}: {e}")
    else:
        logger.warning(f"Файл ID чанков не найден: {ids_file}")
        
    if loaded_embeddings is not None and loaded_ids is not None:
        if len(loaded_embeddings) != len(loaded_ids):
            logger.error(f"Несоответствие количества эмбеддингов ({len(loaded_embeddings)}) и ID чанков ({len(loaded_ids)}) для {base_filename}.")
            return None, None
        
    return loaded_embeddings, loaded_ids

if __name__ == "__main__":
    chunks_file_path = "/home/azureuser/vllm-transfer/dp_code/pdf_to_test_title_chunks.json"
    chunks_input = []
    doc_name_for_output = "unknown_doc"
    
    try:
        with open(chunks_file_path, 'r', encoding='utf-8') as f:
            chunks_input = json.load(f)
        logger.info(f"Успешно загружено {len(chunks_input)} чанков из {chunks_file_path}")
        if chunks_input:
            doc_name_for_output = Path(chunks_input[0].get('doc_id', 'unknown_doc')).stem
    except FileNotFoundError:
        logger.error(f"Файл с чанками не найден: {chunks_file_path}. Невозможно выполнить тест.")
        chunks_input = []
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка декодирования JSON из файла {chunks_file_path}: {e}")
        chunks_input = []
    except Exception as e:
        logger.error(f"Неожиданная ошибка при загрузке чанков: {e}")
        chunks_input = []
    
    if chunks_input:
        embeddings_array, generated_chunk_ids = generate_embeddings(chunks_input,
                                                                    model_name='all-mpnet-base-v2',
                                                                    batch_size=16)
        
        if embeddings_array is not None and generated_chunk_ids is not None:
            print(f"\n--- Результаты генерации эмбеддингов ---")
            print(f"Форма массива эмбеддингов: {embeddings_array.shape}")
            print(f"Количество ID чанков: {len(generated_chunk_ids)}")
            print(f"Первый ID чанка: {generated_chunk_ids[0]}")
            print(f"Пример эмбеддинга (первые 10 значений): {embeddings_array[0, :10]}...")
            
            output_directory = "output/embeddings"
            save_embeddings(embeddings_array, generated_chunk_ids, output_directory, doc_name_for_output)
            
            print(f"\n--- Проверка загрузки сохраненных данных ---")
            loaded_embeddings_array, loaded_chunk_ids = load_embeddings(output_directory,
                                                                        doc_name_for_output)
            if loaded_embeddings_array is not None and loaded_chunk_ids is not None:
                if loaded_embeddings_array.shape == embeddings_array.shape and \
                    len(loaded_chunk_ids) == len(generated_chunk_ids) and \
                        loaded_chunk_ids[0] == generated_chunk_ids[0] and \
                            np.allclose(loaded_embeddings_array[0], embeddings_array[0]):
                    print("Загруженные данные успешно совпадают с сгенерированными (базовая проверка).")
                else:
                    print("ОШИБКА: Загруженные данные не совпадают с сгенерированными!")
            else:
                 print("Не удалось загрузить сохраненные данные для проверки.")
        else:
            print("Генерация эмбеддингов не удалась.")