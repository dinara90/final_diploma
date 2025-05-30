import logging
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
    Pipeline
)

try:
    from vector_store import load_faiss_index, search_faiss_index
    from embedding_generator import load_embeddings
except ImportError:
    logging.error("Не удалось импортировать функции из vector_store.py или embedding_generator.py. "
                  "Убедитесь, что файлы находятся рядом или в PYTHONPATH.")
    def load_faiss_index(p: str) -> Optional[faiss.Index]: return None
    def search_faiss_index(q: np.ndarray, i: faiss.Index, k: int) -> Optional[Tuple[np.ndarray, np.ndarray]]: return None
    def load_embeddings(d: str, b: str) -> Tuple[Optional[np.ndarray], Optional[List[str]]]: return None, None
    
    
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_retriever_components(
    embedding_model_name: str='all-mpnet-base-v2',
    classifier_model_path: str="models/section_classifier_deberta_v3",
    device: Optional[str]=None
) -> Dict:
    """Загружает все необходимые компоненты для гибридного ретривера."""
    logger.info("Загрузка компонентов для гибридного ретривера...")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Используемое устройство: {device}")
    
    components = {}
    try:
        logger.info(f"Загрузка Sentence Transformer модели: {embedding_model_name}...")
        components['embedding_model'] = SentenceTransformer(embedding_model_name, device=device)
        logger.info("Sentence Transformer модель загружена.")
        
        logger.info(f"Загрузка классификатора секций из: {classifier_model_path}...")
        components['classifier_tokenizer'] = AutoTokenizer.from_pretrained(classifier_model_path)
        components['classifier_model'] = AutoModelForSequenceClassification.from_pretrained(classifier_model_path)
        components['classifier_model'].to(device)
        logger.info("Классификатор секций загружен.")
        
        components['classifier_pipeline'] = TextClassificationPipeline(
            model=components['classifier_model'],
            tokenizer=components['classifier_tokenizer'],
            device=0 if device == "cuda" else -1,
        )
        logger.info("Пайплайн классификации создан.")
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке компонентов: {e}", exc_info=True)
        return {}
    
    logger.info("Все компоненты успешно загружены.")
    return components

def find_sections_hybrid(
    faiss_index: faiss.Index,
    chunk_text_map: Dict[str, str],
    ordered_chunk_ids: List[str],
    embedding_model: SentenceTransformer,
    classifier_model: AutoModelForSequenceClassification,
    classifier_tokenizer: AutoTokenizer,
    target_labels: Dict[str, List[str]],
    k: int=20,
    classifier_batch_size: int=16,
    min_classifier_score: float=0.7,
    device: Optional[str]=None
) -> Dict[str, Optional[str]]:
    """
    Выполняет гибридный поиск секций: сначала FAISS, затем классификатор.

    Args:
        doc_chunks: Список чанков текущего документа [{'doc_id':..., 'chunk_id':..., 'text':...}].
        components: Словарь с загруженными компонентами (st_model, faiss_index, all_chunk_ids, 
                    classifier_pipeline, classifier_tokenizer).
        target_labels: Словарь, где ключ - искомая метка секции (напр., 'comments'),
                       а значение - список примеров текста для этой секции.
        k: Количество кандидатов для извлечения из FAISS для каждой метки.
        classifier_batch_size: Размер батча для классификатора.
        min_classifier_score: Минимальный score от классификатора для принятия чанка.

    Returns:
        Словарь, где ключ - искомая метка, значение - объединенный текст найденных
        и подтвержденных чанков (или None, если не найдено).
    """
    
    st_model = embedding_model
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else 'cpu'
    
    classifier_pipeline = TextClassificationPipeline(
        model=classifier_model,
        tokenizer=classifier_tokenizer,
        device=0 if device == "cuda" else -1,
    )
    
    final_texts: Dict[str, List[str]] = {label: [] for label in target_labels}
    
    logger.info("Генерация эталонных эмбеддингов для целевых меток...")
    query_embeddings_map = {}
    for label, ref_texts in target_labels.items():
        if not ref_texts:
            logger.warning(f"Нет референсных текстов для метки '{label}'. Поиск для этой метки невозможен.")
            continue
        try:
            ref_embeddings = st_model.encode(ref_texts, convert_to_numpy=True)
            
            query_embeddings_map[label] = np.mean(ref_embeddings, axis=0, keepdims=True).astype(np.float32)
            logger.info(f"Сгенерирован эмбеддинг для запроса по метке '{label}'")
        except Exception as e:
            logger.error(f"Ошибка при генерации эмбеддинга для метки '{label}': {e}", exc_info=True)
            
    candidate_chunks_for_classification: Dict[str, List[str]] = {label: [] for label in target_labels}
    
    for label, query_embedding in query_embeddings_map.items():
        logger.info(f"Поиск кандидатов в FAISS для метки '{label}' (k={k})...")
        search_results = search_faiss_index(query_embedding, faiss_index, k=k)
        logger.info(f"search_results: {search_results}")
        if search_results:
            distances, indices = search_results
            
            retrieved_faiss_indices = indices[0]
            
            candidate_ids_for_label: Set[str] = set()
            for faiss_idx in retrieved_faiss_indices:
                if 0 <= faiss_idx < len(ordered_chunk_ids):
                    original_chunk_id = ordered_chunk_ids[faiss_idx]
                    candidate_ids_for_label.add(original_chunk_id)
                    
                    logger.debug(f"  FAISS idx {faiss_idx} -> Chunk ID: {original_chunk_id}")
                else:
                    logger.warning(f"Индекс FAISS {faiss_idx} вне допустимого диапазона.")
            
            logger.info(f"Найдено {len(candidate_ids_for_label)} кандидатов из FAISS для метки '{label}'.")
            
            for chunk_id in candidate_ids_for_label:
                chunk_text = chunk_text_map.get(chunk_id)
                if chunk_text is not None:
                    candidate_chunks_for_classification[label].append({
                        'chunk_id': chunk_id,
                        'text': chunk_text
                    })
                else:
                    logger.warning(f"Chunk ID '{chunk_id}' найден в FAISS, но отсутствует в chunk_text_map.")
        else: 
            logger.warning(f"Поиск в FAISS не дал результатов для метки '{label}'.")
            
    added_to_comments_ids: Set[str] = set()
    added_to_well_info_ids: Set[str] = set()
            
    logger.info(f"Классификация чанков-кандидатов... с min_classifier_score={min_classifier_score}")
    # logger.info(f"candidate_chunks_for_classification: {candidate_chunks_for_classification}")
    for label, candidate_chunks in candidate_chunks_for_classification.items():
        if not candidate_chunks:
            logger.warning(f"Нет кандидатов для классификации для метки '{label}'.")
            continue
        
        candidate_texts = [chunk['text'] for chunk in candidate_chunks]
        candidate_ids = [chunk['chunk_id'] for chunk in candidate_chunks]
        
        logger.info(f"Классификация {len(candidate_texts)} кандидатов для метки '{label}'...")
        # logger.info(f"candidate_texts: {candidate_texts}")
        try:
            predictions = classifier_pipeline(candidate_texts, batch_size=classifier_batch_size, truncation=True)
            
            for i, prediction in enumerate(predictions):
                predicted_label = prediction['label']
                score = prediction['score']
                
                current_chunk_id = candidate_ids[i]
                current_chunk_text = candidate_texts[i]
                
                logger.info(f"Кандидат {candidate_ids[i]}: Предсказано '{predicted_label}' с score={score:.4f}") # <-- Новая строка
                
                if score >= min_classifier_score:
                    # logger.info(f"  -> Условие пройдено для {candidate_ids[i]}. Перед append: len(final_texts['{label}'])={len(final_texts[label])}") # <-- Добавить 1
                    if predicted_label == 'well_information':
                        if current_chunk_id not in added_to_well_info_ids:
                            try:
                                final_texts['well_information'].append(current_chunk_text)
                                added_to_well_info_ids.add(current_chunk_id)
                                logger.info(f"  -> Успешно добавлен чанк {current_chunk_id} в список well_information c размером {len(current_chunk_text)}. После append: len(final_texts['well_information'])={len(final_texts['well_information'])}")
                            except Exception as append_err:
                                logger.error(f"  -> ОШИБКА при добавлении чанка {current_chunk_id} в список well_information: {append_err}", exc_info=True)
                        else:
                            logger.debug(f"  -> Чанк {current_chunk_id} уже был добавлен в 'well_information'.")
                    else:
                        if current_chunk_id not in added_to_comments_ids:
                            try:
                                final_texts['comments'].append(current_chunk_text)
                                added_to_comments_ids.add(current_chunk_id)
                                logger.info(f"  -> Успешно добавлен чанк {current_chunk_id} в список comments c размером {len(current_chunk_text)}. После append: len(final_texts['comments'])={len(final_texts['comments'])}")
                            except Exception as append_err:
                                logger.error(f"  -> ОШИБКА при добавлении чанка {current_chunk_id} в 'comments': {append_err}", exc_info=True)
                        else:
                            logger.debug(f"  -> Чанк {current_chunk_id} уже был добавлен в 'comments'.")
                else:
                    logger.debug(f"Чанк {current_chunk_id} отклонен для любой категории из-за низкого score ({score:.4f}). Предсказано: '{predicted_label}'")
                    
        except Exception as e:
            logger.error(f"Ошибка во время классификации для метки '{label}': {e}", exc_info=True)
            
    
    final_result = {}
    well_info_source = 'not_found'
    if 'comments' in final_texts and final_texts['comments']:
        final_result['comments'] = "\n\n".join(final_texts['comments'])
        logger.info(f"Найдена и подтверждена секция 'comments' (состоит из {len(final_texts['comments'])} чанков).")
    else:
        final_result['comments'] = None
        logger.info("Секция 'comments' не найдена или не подтверждена.")
        
    well_info_text_result = None
    if 'well_information' in final_texts and final_texts['well_information']:
        well_info_text_result = "\n\n".join(final_texts['well_information'])
        well_info_source = 'found'
        logger.info(f"Найдена и подтверждена секция 'well_information' (состоит из {len(final_texts['well_information'])} чанков). Источник: {well_info_source}")
    else:
        logger.warning("Секция 'well_information' не найдена гибридным поиском. Применяется запасная логика: взятие первых 2 чанков.")
        fallback_well_info_texts = []
        num_chunks_to_take = min(2, len(ordered_chunk_ids))
        
        if num_chunks_to_take > 0:
            for i in range(num_chunks_to_take):
                chunk_id = ordered_chunk_ids[i]
                chunk_text = chunk_text_map.get(chunk_id)
                if chunk_text:
                    fallback_well_info_texts.append(chunk_text)
                    logger.info(f"  -> Взят чанк {i+1} (ID: {chunk_id}) для fallback 'well_information'.")
                else:
                    logger.warning(f"  -> Не удалось получить текст для чанка {i+1} (ID: {chunk_id}) для fallback.")
            
            if fallback_well_info_texts:
                well_info_text_result = "\n\n".join(fallback_well_info_texts)
                well_info_source = 'fallback'
                logger.info(f"Запасная логика успешно применилась. 'well_information' содержит текст из {len(fallback_well_info_texts)} чанков. Источник: {well_info_source}")
            else:
                 logger.warning("Запасная логика не смогла собрать текст для 'well_information' (чанки не найдены или пусты).")
                 well_info_source = 'not_found'
        else:
            logger.warning("Запасная логика не может быть применена: в документе нет чанков (ordered_chunk_ids пуст).")
            well_info_source = 'not_found'
    
    final_result['well_information'] = {
        'text': well_info_text_result,
        'source': 'fallback'
    }
    logger.info(f"final_texts: {final_result}")
    # logger.info(f"Итоговый результат find_sections_hybrid: { {k: (v if k=='comments' else v.get('source')) for k,v in final_result.items()} }") # Логируем источники
    return final_result