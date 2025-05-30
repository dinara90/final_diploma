import argparse
import logging
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import time

import pandas as pd
from tqdm import tqdm
import torch

try:
    from vector_store import load_faiss_index
    from section_retriever import load_retriever_components, find_sections_hybrid
    from problem_analyzer import load_llm_pipeline, analyze_comment_problem, extract_details_llm, DEFAULT_PROBLEM_CATEGORIES, DEFAULT_DETAIL_EXTRACTION_PROMPT_TEMPLATE
    from detail_extractor import extract_well_details, load_spacy_model
except ImportError as e:
    logging.error(f"Ошибка импорта модулей. Убедитесь, что все скрипты "
                  f"(document_processor.py, section_retriever.py, problem_analyzer.py, detail_extractor.py, "
                  f"vector_store.py, embedding_generator.py) находятся в директории scripts "
                  f"или доступны в PYTHONPATH. Ошибка: {e}", exc_info=True)
    exit(1)
    
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

DEFAULT_PROBLEM_CATEGORIES = ["stuck_pipe", "lost_circulation", "well_control", "equipment_failure", "other", "none"]

DEFAULT_PROMPT_TEMPLATE = """Analyze the following comment from a drilling report.
Determine if a problem is described. If yes, classify the problem into ONE of the following categories: {categories_str}

Respond ONLY with a valid JSON object in the following format:
{{
"problem_detected": boolean, // true if a problem is described, false otherwise
"category": "string"       // one of the categories listed above, or "none" if no problem detected
}}

Comment:
{comment_text}

JSON Response:
"""

def run_pipeline(
    input_dir: str,
    output_file: str,
    processed_data_path: str,
    config: Dict,
):
    """
    Запускает полный пайплайн обработки DDR файлов.

    Args:
        input_dir: Директория с входными DDR файлами (PDF, DOCX...).
        output_file: Путь для сохранения итогового CSV файла.
        config: Словарь с конфигурацией пайплайна (пути к моделям, параметры и т.д.).
    """
    start_time_pipeline = time.time()
    logger.info("--- Запуск Пайплайна Обработки DDR ---")
    logger.info(f"Входная директория: {input_dir}")
    logger.info(f"Выходной файл: {output_file}")
    logger.info(f"Конфигурация: {json.dumps(config, indent=2)}")
    
    logger.info("--- Шаг 1: Загрузка моделей и компонентов ---")
    load_start_time = time.time()
    components = {}
    llm_model = None
    llm_tokenizer = None
    
    try:
        components = load_retriever_components(
            embedding_model_name=config['embedding_model'],
            classifier_model_path=config['section_classifier_path']
        )
        if not components:
            raise RuntimeError("Не удалось загрузить компоненты ретривера.")    
        
        llm_model, llm_tokenizer = load_llm_pipeline(
            model_name=config['llm_model_name'],
            use_4bit=config.get('use_4bit', True),
            use_8bit=config.get('use_8bit', False)
        )
        if not llm_model or not llm_tokenizer:
            raise RuntimeError("Не удалось загрузить LLM модель или токенизатор.")
        
        if config.get('detail_extractor_use_spacy', True):
            if not load_spacy_model(config.get('spacy_model_name', 'en_core_web_sm')):
                logger.warning("Не удалось загрузить spaCy модель, извлечение деталей будет работать только с regex.")
                config['detail_extractor_use_spacy'] = False
                
        load_end_time = time.time()
        logger.info(f"Все компоненты загружены за {load_end_time - load_start_time:.2f} сек.")
    
    except Exception as e:
        logger.error(f"Критическая ошибка при загрузке компонентов: {e}. Пайплайн остановлен.", exc_info=True)
        return
    
    logger.info(f"--- Шаг 2: Поиск и обработка файлов в {input_dir} ---")
    input_path = Path(input_dir)
    
    file_pattern = config.get('file_pattern', '*.pdf')
    logger.info(f"Используемый паттерн для поиска файлов: {file_pattern}")
    
    ddr_files = list(input_path.glob(file_pattern))
    if file_pattern == '.*pdf':
        ddr_files.extend(list(input_path.glob('*.docx')))
        
    if not ddr_files:
        logger.error(f"Не найдено файлов по указанному пути и паттерну: {input_dir}/{file_pattern}")
        return

    logger.info(f"Найдено {len(ddr_files)} файлов для обработки.")
    
    results_list = []
    
    for file_path in tqdm(ddr_files, desc="Обработка файлов"):
        file_start_time = time.time()
        logger.info(f"--- Обработка файла: {file_path.name} ---")
        
        file_result = {
            'file_name': file_path.name,
            'processing_status': 'pending',
            'error_message': None,
            'comments_text': None,
            'well_info_text': None,
            'problem_detected': None,
            'problem_category': None,
            'well_identifier': None,
            'latitude': None,
            'longitude': None,
            'llm_raw_output': None,
            'llm_parse_error': None
        }
        
        try:
            logger.debug(f"Шаг 2.1: Загрузка предобработанных данных для {file_path.name}...")
            try:
                processed_data_dir = Path(processed_data_path)
                doc_base_name = file_path.stem
                current_index_path = processed_data_dir / f"{doc_base_name}.index"
                current_chunks_path = processed_data_dir / f"{doc_base_name}_chunks.json"
                
                if not current_index_path.exists():
                    raise FileNotFoundError(f"Файл индекса не найден: {current_index_path}")
                if not current_chunks_path.exists():
                    raise FileNotFoundError(f"Файл чанков не найден: {current_chunks_path}")
                
                logger.debug(f"Загрузка индекса из {current_index_path}...")
                current_faiss_index = load_faiss_index(str(current_index_path))
                if not current_faiss_index:
                    raise ValueError("Не удалось загрузить FAISS индекс (load_faiss_index вернул None).") 
                logger.info(f"   Индекс для {doc_base_name} загружен, векторов: {current_faiss_index.ntotal}")
                
                logger.debug(f"Загрузка чанков из {current_chunks_path}...")
                with open(current_chunks_path, 'r', encoding='utf-8') as f:
                    loaded_chunks_list = json.load(f)
                
                if not loaded_chunks_list:
                    raise ValueError("Файл чанков пуст.")
                
                chunk_text_map = {}
                ordered_chunk_ids = []
                for chunk in loaded_chunks_list:
                    chunk_id = chunk['chunk_id']
                    chunk_text = chunk.get('text', '')
                    chunk_text_map[chunk_id] = chunk_text
                    ordered_chunk_ids.append(chunk_id)
                
                logger.info(f"   Данные чанков для {doc_base_name} загружены ({len(chunk_text_map)} чанков).")
            except Exception as e:
                logger.error(f"Ошибка при загрузке предобработанных данных для {file_path.name}: {e}", exc_info=True)
                file_result['processing_status'] = 'error'
                file_result['error_message'] = f"Ошибка загрузки данных: {e}"
                results_list.append(file_result)
                continue
                
            logger.debug(f"Шаг 2.2: Гибридный поиск секций...")
            found_sections = find_sections_hybrid(
                faiss_index=current_faiss_index,
                chunk_text_map=chunk_text_map,
                ordered_chunk_ids=ordered_chunk_ids,
                embedding_model=components['embedding_model'],
                classifier_model=components['classifier_model'],
                classifier_tokenizer=components['classifier_tokenizer'],
                target_labels=config['target_sections'],
                k=config.get('retriever_k', 20),
                classifier_batch_size=config.get('classifier_batch_size', 16),
                min_classifier_score=config.get('min_classifier_score', 0.5),
                device=config.get('device', 'gpu' if torch.cuda.is_available() else 'cpu')
            )
            comments_text = found_sections.get('comments', None)
            well_info_data = found_sections.get('well_information', {'text': None, 'source': 'not_found'})
            well_info_text = well_info_data.get('text')
            well_info_source = well_info_data.get('source', 'not_found')
            
            file_result['comments_text'] = comments_text
            
            # logger.info(f"--- Текст комментариев для LLM анализа ---:\n{file_result}\n------------------------------------")
            if comments_text:
                logger.debug(f"Шаг 2.3: Анализ найденных комментариев...")
                analysis_result = analyze_comment_problem(
                    comment_text=file_result['comments_text'],
                    model=llm_model,
                    tokenizer=llm_tokenizer,
                    problem_categories=config.get('problem_categories', DEFAULT_PROBLEM_CATEGORIES),
                    prompt_template=config.get('prompt_template', DEFAULT_PROMPT_TEMPLATE),
                    max_new_tokens=config.get('max_new_tokens', 100),
                )
                file_result['problem_detected'] = analysis_result.get('problem_detected')
                file_result['problem_category'] = analysis_result.get('category')
                file_result['llm_raw_output'] = analysis_result.get('raw_output')
                file_result['llm_parse_error'] = analysis_result.get('error')
                if analysis_result.get('error'):
                    logger.warning(f"Ошибка при анализе комментариев: {analysis_result.get('error')}")
            else:
                logger.info("Секция 'comments' не найдена, анализ проблем пропущен.")
            
            if well_info_text:
                logger.debug(f"Шаг 2.4: Извлечение деталей скважины...")
                
                well_details = {}
                combined_text = ""
                if file_result['well_info_text'] is not None:
                    combined_text += file_result['well_info_text']
                if file_result['comments_text'] is not None:
                    if combined_text:
                        combined_text += '\n'
                    combined_text += file_result['comments_text']
                
                if well_info_source == 'found':
                    logger.info("Извлечение деталей с помощью Regex/SpaCy...")
                    
                    well_details = extract_well_details(
                        well_info_text=combined_text,
                        use_spacy=config.get('detail_extractor_use_spacy', True)
                    )    
                    logger.info(f"Успешно извлечены детали скважины: {well_details}")
                    
                elif well_info_source == 'fallback':
                    logger.info("Извлечение деталей с помощью LLM...")
                    llm_detail_result = extract_details_llm(
                        text_to_analyze=combined_text,
                        model=llm_model,
                        tokenizer=llm_tokenizer,
                        prompt_template=config.get('llm_detail_extraction_prompt_template', DEFAULT_DETAIL_EXTRACTION_PROMPT_TEMPLATE),
                        max_new_tokens=config.get('llm_detail_extraction_max_tokens', 100)
                    )
                    well_details['well_identifier'] = llm_detail_result.get('well_identifier')
                    well_details['latitude'] = llm_detail_result.get('latitude')
                    well_details['longitude'] = llm_detail_result.get('longitude')
                    
                    if llm_detail_result.get('error'):
                        logger.warning(f"Ошибка при LLM-извлечении деталей: {llm_detail_result['error']}")
                        # file_result['llm_detail_parse_error'] = llm_detail_result.get('error')
                else:
                    logger.warning(f"Текст well_information есть, но источник '{well_info_source}' не 'found' или 'fallback'. Пропуск извлечения деталей.")
                    
                    
                file_result['well_identifier'] = well_details.get('well_identifier')
                file_result['latitude'] = well_details.get('latitude')
                file_result['longitude'] = well_details.get('longitude')
                logger.info(f"Извлеченные детали: ID={file_result['well_identifier']}, "
                            f"Lat={file_result['latitude']}, Lon={file_result['longitude']}")
            else:
                logger.info("Секция 'well_information' не найдена, извлечение деталей пропущено.")
                
            file_result['processing_status'] = 'success'
            logger.info(f"Файл {file_path.name} успешно обработан.")
        
        except Exception as e:
            file_result['processing_status'] = 'error'
            file_result['error_message'] = str(e)
            logger.error(f"Ошибка при обработке файла {file_path.name}: {e}", exc_info=True) # Логируем traceback
            
        finally:
            file_result.pop('comments_text', None) 
            file_result.pop('well_info_text', None)
            file_result.pop('llm_raw_output', None)
            
            results_list.append(file_result)
            file_end_time = time.time()
            logger.info(f"Обработка {file_path.name} заняла {file_end_time - file_start_time:.2f} сек.")
        
    
    logger.info(f"--- Шаг 3: Сохранение результатов в {output_file} ---")
    if results_list:
        try:
            columns_order = [
                'file_name', 'processing_status', 'error_message', 
                'well_identifier', 'latitude', 'longitude', 
                'problem_detected', 'problem_category', 'llm_parse_error'
            ]
            for res in results_list:
                for col in columns_order:
                    res.setdefault(col, None)
                    
            df_results = pd.DataFrame(results_list, columns=columns_order)
            
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df_results.to_csv(output_path, index=False, encoding='utf-8')
            
            logger.info(f"Результаты успешно сохранены в {output_file}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении результатов в CSV: {e}", exc_info=True)
    else:
        logger.warning("Список результатов пуст. Файл CSV не создан.")
        
    end_time_pipeline = time.time()
    logger.info(f"--- Пайплайн завершен. Общее время выполнения: {end_time_pipeline - start_time_pipeline:.2f} сек. ---")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DDR processing pipeline.")
    
    parser.add_argument("--input_dir", type=str, required=True, help="Директория с входными DDR файлами.")
    parser.add_argument("--output_file", type=str, required=True, help="Путь для сохранения итогового CSV файла.")
    parser.add_argument("--processed_data_path_cli", type=str, default="/home/azureuser/vllm-transfer/dp_code/output/pipeline_results_short_chunks_500_not_image", help="Путь к предобработанным данным для локального запуска.")
    
    parser.add_argument("--config_file", type=str, default="config.json", help="Путь к JSON файлу конфигурации пайплайна.")
    
    # parser.add_argument("--embedding_model", type=str, default='all-mpnet-base-v2')
    # parser.add_argument("--faiss_index_path", type=str, default="output/faiss_indices/global.index")
    # parser.add_argument("--chunk_ids_path", type=str, default="output/embeddings/global_chunk_ids.json")
    # parser.add_argument("--section_classifier_path", type=str, default="models/section_classifier_deberta_v3")
    # parser.add_argument("--llm_model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    # parser.add_argument("--llm_use_4bit", action='store_true', default=True)
    # parser.add_argument("--llm_use_8bit", action='store_true', default=False)

    args = parser.parse_args()
    
    pipeline_config = {}
    
    DEFAULT_CONFIG = {
        "embedding_model": "all-mpnet-base-v2",
        # "processed_data_dir": "/home/azureuser/vllm-transfer/dp_code/output/pipeline_results_short_chunks_500_not_image",
        "section_classifier_path": "models/section_classifier_deberta_v3",
        "llm_model_name": "Qwen/Qwen2.5-7B-Instruct",
        "llm_use_4bit": True,
        "llm_use_8bit": False,
        "llm_prompt_template": DEFAULT_PROMPT_TEMPLATE,
        "llm_max_new_tokens": 100,
        "llm_detail_extraction_prompt_template": DEFAULT_DETAIL_EXTRACTION_PROMPT_TEMPLATE,
        "llm_detail_extraction_max_tokens": 100,
        "problem_categories": DEFAULT_PROBLEM_CATEGORIES,
        "target_sections": {
            "comments": [
        "Operations Summary",
      "Management Summary",
      "Current Operations",
      "Planned Operations",
      "Comments:",
      "Summary of Issues:",
      "Problems Encountered:"
    #   "Waited on Day Light",
    #   "Conducted Pre-Spud Rig Inspection",
    #   "Strapped, OD, ID and Fishing Neck of HWDP",
    #   "Picked up 114 joints of 5\" drill pipe",
    #   "Cleaned and strapped 16 in Casing",
    #   "Made up Directional tools",
    #   "Finished drilling 22 in hole to 421 ft",
    #   "POH and run 16 in casing and cement in place",
    #   "Nipple up and test 20 in BOP",
    #   "Move company man quarters, tool pusher quarters",
    #   "MSE training with all contractors",
    #   "Tear down and move everything off 56-32 location",
    #   "Hold JSA with all crews involved with rig move",
    #   "Spoke to Jim Goddard with Department of Water Resources",
    #   "Safety Summary",
    #   "No incidents or events reported",
    #   "0 days since LTI",
    #   "Conducted Safety Meeting",
    #   "Safety: Mixing chemicals, pinch points, running casing / BHA",
    #   "Clean up site and release Wyoming Casing",
    #   "Removed monkey board from derrick",
    #   "Welder completed modifying racking board",
    #   "Received 1st load of diesel",
    #   "Jim had received messages of our start casing, cementing and BOP testing schedule",
    #   "Waived witnessing of the 20\" BOP test provided testing be done by a 3rd party"
                ],
            "well_information": [
                "Well:",
                "Well ID:",
                "Well Name:",
                "Well Data:",
                "REPORT",
                "REPORT N°",
                "DAILY DRILLING REPORT"
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
        "file_pattern": "*.pdf"
    }
    
    for key, value in DEFAULT_CONFIG.items():
        pipeline_config.setdefault(key, value)
        
    if pipeline_config.get('llm_use_4bit') and pipeline_config.get('llm_use_8bit'):
        logger.warning("В конфигурации указаны и llm_use_4bit=True, и llm_use_8bit=True. Приоритет у 4bit.")
        pipeline_config['llm_use_8bit'] = False
    if not pipeline_config.get('llm_use_4bit') and not pipeline_config.get('llm_use_8bit'):
         logger.info("Квантизация LLM не включена в конфигурации. Используется float16/bfloat16.")
         
    log_dir = Path(args.output_file).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"pipeline_run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(logging.StreamHandler())
    root_logger.addHandler(file_handler)
    logger.info(f"Логи будут сохранены в: {log_file_path}")
    
    logger.info(f"pipeline_config: {pipeline_config}")
    
    run_pipeline(args.input_dir, args.output_file, args.processed_data_path_cli, pipeline_config)