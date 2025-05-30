import logging
import json
from pathlib import Path
from typing import List, Dict, Optional

from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Text as UnstructuredText

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_chunk(
    file_path:str, chunking_strategy: str="title",
    max_chunk_chars: Optional[int]=1500,
    combine_under_n_chars: Optional[int] = 200,
    new_after_n_chars: Optional[int] = 1500,
    **partition_kwargs
) -> List[Dict]:
    """
    Загружает документ (PDF, DOCX и др.), извлекает контент с помощью unstructured,
    разбивает на чанки и возвращает структурированный список чанков.

    Args:
        file_path: Путь к файлу документа.
        chunking_strategy: Стратегия чанкинга ('title', 'basic', 'none').
                           'title' использует chunk_by_title.
                           'basic' - весь текст как один чанк.
                           'none' - каждый элемент unstructured как чанк.
        max_chunk_chars: Максимальное количество символов для чанка (для 'title').
        combine_under_n_chars: Объединяет короткие блоки текста под заголовком (для 'title').
        new_after_n_chars: Начинать новый чанк, если текущий достиг этого размера (для 'title').
        **partition_kwargs: Дополнительные аргументы для unstructured.partition.auto.partition
                           (например, strategy='hi_res', pdf_infer_table_structure=True).

    Returns:
        Список словарей, где каждый словарь представляет чанк:
        [{'doc_id': ..., 'chunk_id': ..., 'text': ...}, ...]
        Возвращает пустой список в случае ошибки.
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.is_file():
        logger.error(f"File not found: {file_path}")
        return []
    
    doc_id = file_path_obj.name
    logger.info(f"Processing document: {doc_id}")
    
    try:
        elements = partition(filename=str(file_path_obj), **partition_kwargs)
        logger.info(f"Extracted {len(elements)} elements from {doc_id}")
    except Exception as e:
        logger.error(f"Error processing {doc_id}: {e}", exc_info=True)
        return []
    
    if not elements:
        logger.warning(f"No elements extracted from {doc_id}")
        return []

    chunks_elements = []
    applied_strategy_log_name = chunking_strategy
    
    if chunking_strategy == "title":
        try:
            chunks_elements = chunk_by_title(
                elements, max_characters=max_chunk_chars,
                combine_text_under_n_chars=combine_under_n_chars,
                new_after_n_chars=new_after_n_chars
            )
            logger.info(f"Применена стратегия чанкинга 'title', получено {len(chunks_elements)} чанков для {doc_id}")
        except Exception as e:
            logger.error(f"Ошибка при чанкинге ('title') для {doc_id}: {e}. Возврат к базовому объединению.", exc_info=True)
            full_text = "\n\n".join([el.text for el in elements if hasattr(el, 'text') and el.text and el.text.strip()])
            if full_text:
                chunks_elements = [UnstructuredText(text=full_text)]
            else:
                chunks_elements = []
            applied_strategy_log_name = "basic (fallback)"
    elif chunking_strategy == "basic":
        full_text = "\n\n".join([el.text for el in elements if hasattr(el, 'text') and el.text and el.text.strip()])
        if full_text:
            chunks_elements = [UnstructuredText(text=full_text)]
        else:
            chunks_elements = []
        logger.info(f"Применена стратегия чанкинга 'basic' (весь текст как один чанк) для {doc_id}")
    elif chunking_strategy == "none":
        chunks_elements = [el for el in elements if hasattr(el, 'text') and el.text and el.text.strip()]
        logger.info(f"Стратегия чанкинга 'none', используется {len(chunks_elements)} исходных элементов как чанки для {doc_id}")
    else:
        logger.warning(f"Неизвестная стратегия чанкинга: {chunking_strategy}. Чанкинг не выполнен.")
        return []
        
    output_chunks = []
    for i, chunk in enumerate(chunks_elements):
        if hasattr(chunk, 'text') and chunk.text:
            chunk_text = chunk.text.strip()
            if chunk_text:
                chunk_id = f"{doc_id}-{i}"
                output_chunks.append({
                    'doc_id': doc_id, 
                    'chunk_id': chunk_id,
                    'text': chunk_text
                })
            else:
                logger.debug(f"Пропущен пустой чанк {i} (после strip) для {doc_id}")
        else:
            logger.debug(f"Пропущен элемент чанка {i} без атрибута 'text' или с пустым текстом для {doc_id}")
            
    if not output_chunks and len(elements) > 0:
        logger.warning(f"Не удалось создать непустые текстовые чанки для {doc_id} (стратегия: {applied_strategy_log_name}), хотя элементы были извлечены.")
    elif output_chunks:
        logger.info(f"Успешно создано {len(output_chunks)} текстовых чанков для {doc_id} (стратегия: {applied_strategy_log_name})")
    
    return output_chunks
    
if __name__ == "__main__":
    test_file_path = "/home/azureuser/vllm-transfer/dp_code/pdf_to_test.pdf"
        
    if Path(test_file_path).exists():
        print(f"\n--- Тест файла: {test_file_path} ---")
        
        print("\n--- Стратегия чанкинга: 'title' ---")
        
        partition_opts = {}
        chunk_title = load_and_chunk(
            test_file_path, chunking_strategy="title", max_chunk_chars=500,
            new_after_n_chars=500, combine_under_n_chars=100,
            **partition_opts
        )
        
        if chunk_title:
            output_filename = Path(test_file_path).stem + "_title_chunks.json"
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(chunk_title, f, ensure_ascii=False, indent=4)
                print(f"Результаты сохранены в файл: {output_filename}")
            except IOError as e:
                logger.error(f"Не удалось сохранить результат в файл {output_filename}: {e}")
        else:
            print("Чанки ('title') не были созданы или не содержат текста.")
    else:
        logger.warning(f"Тестовый файл {test_file_path} не найден или не был создан. Пропустили тестирование.")