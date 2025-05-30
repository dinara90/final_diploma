import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import faiss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
try:
    from embedding_generator import load_embeddings
except ImportError:
    logger.warning("Не удалось импортировать load_embeddings. Убедитесь, что embedding_generator.py находится в той же директории или доступен в PYTHONPATH.")
    def load_embeddings(output_dir: str, base_filename: str) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        return None, None

def build_and_save_faiss_index(embeddings: np.ndarray, index_path: str):
    """
    Создает индекс FAISS на основе эмбеддингов и сохраняет его на диск.

    Args:
        embeddings: Numpy массив эмбеддингов (N x D).
        index_path: Путь для сохранения файла индекса FAISS.
    """
    if embeddings is None or embeddings.size == 0:
        logger.error("Массив эмбеддингов пуст или None. Индекс не создан.")
        return False
    
    if embeddings.ndim != 2:
        logger.error(f"Массив эмбеддингов должен быть 2D, получено {embeddings.ndim}D. Индекс не создан.")
        return False
    
    num_vectors, dimension = embeddings.shape
    logger.info(f"Создание индекса FAISS для {num_vectors} векторов размерности {dimension}.")
    
    try:
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype(np.float32))
        
        logger.info(f"Векторы успешно добавлены в индекс. Общее количество векторов в индексе: {index.ntotal}")
        
        index_path_obj = Path(index_path)
        index_path_obj.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path_obj))
        logger.info(f"Индекс успешно сохранен в {index_path_obj}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при создании или сохранении индекса FAISS: {e}", exc_info=True)
        return False
        
def load_faiss_index(index_path: str) -> Optional[faiss.Index]:
    """
    Загружает индекс FAISS с диска.

    Args:
        index_path: Путь к файлу индекса FAISS.

    Returns:
        Загруженный объект faiss.Index или None в случае ошибки.
    """
    index_path_obj = Path(index_path)
    if not index_path_obj.is_file():
        logger.error(f"Файл индекса FAISS не найден: {index_path}")
        return None
    
    try:
        index = faiss.read_index(str(index_path_obj))
        logger.info(f"Индекс FAISS успешно загружен из: {index_path}. "
                    f"Количество векторов: {index.ntotal}, Размерность: {index.d}")
        return index
    except Exception as e:
        logger.error(f"Ошибка при загрузке индекса FAISS из {index_path}: {e}", exc_info=True)
        return None
    
def search_faiss_index(query_embeddings: np.ndarray,
                       index: faiss.Index, 
                       k: int=5) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Выполняет поиск ближайших соседей в индексе FAISS для одного или нескольких запросов.

    Args:
        query_embeddings: Numpy массив с эмбеддингами запросов (M x D), где M - кол-во запросов.
                          Должен быть 2D и типа float32.
        index: Загруженный объект faiss.Index.
        k: Количество ближайших соседей для поиска.

    Returns:
        Кортеж (distances, indices) или None в случае ошибки.
        distances: Numpy массив (M x k) со значениями схожести (или расстояний).
                   Для IndexFlatIP - скалярное произведение (выше = лучше).
        indices: Numpy массив (M x k) с индексами найденных соседей в оригинальном массиве эмбеддингов.
    """
    if query_embeddings is None or query_embeddings.size == 0:
        logger.error("Массив эмбеддингов запроса пуст или None.")
        return None
    
    if index is None:
        logger.error("Объект индекса FAISS не предоставлен (None).")
        return None

    if query_embeddings.shape[1] != index.d:
        logger.error(f"Размерность эмбеддингов запроса ({query_embeddings.shape[1]}) "
                      f"не совпадает с размерностью индекса ({index.d}).")
        return None
    
    if k > index.ntotal:
        logger.warning(f"Запрошено k={k} соседей, но в индексе всего {index.ntotal} векторов. Установлено k={index.ntotal}")
        k = index.ntotal
        
    if k <= 0:
        logger.error(f"Значение k должно быть положительным, получено {k}.")
        return None
    
    try:
        distances, indices = index.search(query_embeddings.astype(np.float32), k)
        logger.info(f"Поиск для {query_embeddings.shape[0]} запросов завершен. Найдено топ-{k} соседей.")
        return distances, indices
    except Exception as e:
        logger.error(f"Ошибка при поиске в индексе FAISS: {e}", exc_info=True)
        return None
    

if __name__ == "__main__": 
    # --- Исправленные пути --- 
    embeddings_dir = "output/embeddings"  # Директория с эмбеддингами
    doc_base_name = "pdf_to_test"        # Базовое имя документа
    
    print(f"--- Загрузка данных для документа: {doc_base_name} ---")
    embeddings_array, chunk_ids = load_embeddings(embeddings_dir, doc_base_name)
    
    if embeddings_array is not None and chunk_ids is not None:
        # Используем базовое имя для пути к индексу
        index_file_path = f"output/faiss_indices/{doc_base_name}.index"
        print(f"\n--- Создание и сохранение индекса FAISS в {index_file_path} ---")
        
        success = build_and_save_faiss_index(embeddings_array, index_file_path)
        
        if success:
            print(f"\n--- Загрузка индекса FAISS из {index_file_path} ---")
            loaded_index = load_faiss_index(index_file_path)
            
            if loaded_index:
                print(f"\n--- Тестовый поиск в индексе ---")
                
                query_vector = embeddings_array[0:1]
                num_neighbors = 5
                
                search_results = search_faiss_index(query_vector, loaded_index, k=num_neighbors)
                
                if search_results:
                    distances, indices = search_results
                    print(f"Результаты поиска для первого чанка (ID: {chunk_ids[0]}):")
                    print(f"  Найденные индексы: {indices[0]}")
                    print(f"  Значения схожести (Inner Product): {distances[0]}")
                    
                    print(f"  Найденные Chunk IDs:")
                    for idx in indices[0]:
                        if 0 <= idx < len(chunk_ids):
                            print(f"    - {chunk_ids[idx]} (индекс: {idx})")
                        else:
                            print(f"    - Неверный индекс: {idx}")
                    
                    if indices[0][0] == 0:
                        print("Самопроверка: Первый результат совпадает с запросом, как и ожидалось.")
                    else:
                        print("ПРЕДУПРЕЖДЕНИЕ: Первый результат НЕ совпадает с запросом!")
                else:
                    print("Не удалось выполнить поиск.")
                    
    else:
        print(f"Не удалось загрузить эмбеддинги или ID чанков для {doc_base_name} из {embeddings_dir}. Тестирование vector_store не выполнено.")