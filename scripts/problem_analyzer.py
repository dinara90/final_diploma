import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
    StoppingCriteriaList,
    StoppingCriteria
)

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

def load_llm_pipeline(
    model_name: str,
    use_4bit: bool=True,
    use_8bit: bool=False,
    device_map: str='auto'
) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
    """Загружает LLM и токенизатор с опциональной квантизацией."""
    logger.info(f"Загрузка LLM: {model_name} (4bit={use_4bit}, 8bit={use_8bit}, device_map='{device_map}')")
    
    if use_4bit and use_8bit:
        logger.warning("Одновременно выбраны 4-bit и 8-bit квантизация. Используется 4-bit.")
        use_8bit = False
        
    quantization_config = None
    torch_dtype = torch.float16
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        torch_dtype = torch.bfloat16
        logger.info("Настроена 4-bit квантизация (NF4, BFloat16 compute).")
    elif use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        logger.info("Настроена 8-bit квантизация.")
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Установлен pad_token = eos_token")
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        logger.info(f"Модель {model_name} и токенизатор успешно загружены.")
        return model, tokenizer
    except ImportError as e:
         if "bitsandbytes" in str(e):
              logger.error("Ошибка: библиотека bitsandbytes не установлена. "
                           "Установите ее: pip install bitsandbytes")
         elif "flash_attn" in str(e):
              logger.error("Ошибка: библиотека flash-attn не установлена или несовместима.")
         else:
              logger.error(f"Ошибка импорта при загрузке модели: {e}", exc_info=True)
         return None, None
    except Exception as e:
        logger.error(f"Неожиданная ошибка при загрузке модели или токенизатора: {e}", exc_info=True)
        return None, None
    

def analyze_comment_problem(
    comment_text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problem_categories: List[str] = DEFAULT_PROBLEM_CATEGORIES,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    max_new_tokens: int = 100,
) -> Dict:
    """
    Анализирует текст комментария с помощью LLM для детекции и классификации проблем.

    Args:
        comment_text: Текст комментария для анализа.
        model: Загруженная модель LLM.
        tokenizer: Загруженный токенизатор.
        problem_categories: Список допустимых категорий проблем.
        prompt_template: Шаблон промпта.
        max_new_tokens: Максимальное количество токенов для генерации ответа.

    Returns:
        Словарь с результатами анализа:
        {
            'problem_detected': bool | None, # None в случае ошибки парсинга
            'category': str | None,        # None в случае ошибки парсинга/валидации
            'raw_output': str,             # Полный сырой вывод модели
            'parsed_json': dict | None,    # Распарсенный JSON, если успешно
            'error': str | None            # Сообщение об ошибке, если была
        }
    """
    if not comment_text or not comment_text.strip():
        return { 'problem_detected': None, 'category': None, 'raw_output': "", 'parsed_json': None, 'error': "Пустой входной текст комментария."}
    
    categories_str = ", ".join(f"'{cat}'" for cat in problem_categories)
    prompt = prompt_template.format(categories_str=categories_str, comment_text=comment_text)
    
    result = {
        "problem_detected": None,
        "category": None,
        "raw_output": "",
        "parsed_json": None,
        "error": None
    }
    
    try:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=False).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        raw_output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        logger.info(f"Сырой вывод модели: {raw_output_text}")
        result['raw_output'] = raw_output_text
        
        try: 
            json_start = raw_output_text.find('{')
            json_end = -1
            parsed_json = None
            
            if json_start != -1:
                brace_level = 0
                for i, char in enumerate(raw_output_text[json_start:]):
                    if char == '{':
                        brace_level += 1
                    elif char == '}':
                        brace_level -= 1
                        if brace_level == 0:
                            json_end = json_start + i + 1
                            break

                if json_end != -1:
                    json_string = raw_output_text[json_start:json_end]
                    try:
                        parsed_json = json.loads(json_string)
                        result['parsed_json'] = parsed_json
                        logger.info(f"Успешно извлечен JSON: {json_string}")

                        if not isinstance(parsed_json, dict):
                            raise ValueError("Распарсенный JSON не является словарем.")
                        
                        problem_detected = parsed_json.get('problem_detected', None)
                        category = parsed_json.get('category', None)
                        
                        result['problem_detected'] = problem_detected
                        result['category'] = category
                        logger.info(f"JSON успешно распарсен и валидирован: {result}")


                    except json.JSONDecodeError as e:
                        result['error'] = f"Ошибка декодирования извлеченного JSON: {e}. Извлеченная строка: {json_string}. Полный вывод: {raw_output_text}"
                        logger.error(result['error'])
                        parsed_json = None 
                    except ValueError as e:
                         result['error'] = f"Ошибка валидации JSON: {e}. Распарсено: {result.get('parsed_json')}"
                         logger.error(result['error'])
                         parsed_json = None

                else:
                    result['error'] = "Не удалось найти соответствующую закрывающую '}' для JSON объекта."
                    logger.warning(f"{result['error']} Сырой вывод: {raw_output_text}")

            else:
                result['error'] = "Не удалось найти открывающую '{' для JSON объекта в ответе модели."
                logger.warning(f"{result['error']} Сырой вывод: {raw_output_text}")
            
        except json.JSONDecodeError as e:
                result['error'] = f"Ошибка декодирования JSON: {e}. Сырой вывод: {raw_output_text}"
                logger.error(result['error'])
        except ValueError as e:
                result['error'] = f"Ошибка валидации JSON: {e}. Распарсено: {result.get('parsed_json')}"
                logger.error(result['error'])
        except Exception as e:
                result['error'] = f"Неожиданная ошибка при парсинге или валидации JSON: {e}. Сырой вывод: {raw_output_text}"
                logger.error(result['error'], exc_info=True)
                
                
    except Exception as e:
        result['error'] = f"Ошибка во время генерации ответа LLM: {e}"
        logger.error(result['error'], exc_info=True)
        
    return result

DEFAULT_DETAIL_EXTRACTION_PROMPT_TEMPLATE = """Extract the well identifier, latitude, and longitude from the following text from a drilling report.

Respond ONLY with a valid JSON object in the following format:
{{
  "well_identifier": "string or null", // The unique name or ID of the well (e.g., "Well-A1", "78B-32", "HON-GT-01S1")
  "latitude": "float or null",         // Latitude in decimal degrees (e.g., 40.7128)
  "longitude": "float or null"         // Longitude in decimal degrees (e.g., -74.0060)
}}

If a value cannot be found, use null.

Text:
{text_to_analyze}

JSON Response:
"""

def extract_details_llm(
    text_to_analyze: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_template: str = DEFAULT_DETAIL_EXTRACTION_PROMPT_TEMPLATE,
    max_new_tokens: int = 100,
) -> Dict[str, Any]:
    """
    Извлекает детали скважины (ID, lat, lon) из текста с помощью LLM.

    Args:
        text_to_analyze: Текст для анализа (обычно первые чанки документа).
        model: Загруженная модель LLM.
        tokenizer: Загруженный токенизатор.
        prompt_template: Шаблон промпта для извлечения деталей.
        max_new_tokens: Максимальное количество токенов для генерации ответа.

    Returns:
        Словарь с результатами извлечения:
        {
            'well_identifier': str | None,
            'latitude': float | None,
            'longitude': float | None,
            'raw_output': str,
            'parsed_json': dict | None,
            'error': str | None
        }
    """
    if not text_to_analyze or not text_to_analyze.strip():
        return {
            'well_identifier': None, 'latitude': None, 'longitude': None,
            'raw_output': "", 'parsed_json': None, 'error': "Пустой входной текст для анализа."
        }
        
    prompt = prompt_template.format(text_to_analyze=text_to_analyze)
    
    result = {
        "well_identifier": None,
        "latitude": None,
        "longitude": None,
        "raw_output": "",
        "parsed_json": None,
        "error": None
    }
    
    try:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=False).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        raw_output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        logger.info(f"LLM Detail Extraction: Сырой вывод модели: {raw_output_text}")
        result['raw_output'] = raw_output_text
        
        json_start = raw_output_text.find('{')
        json_end = -1
        parsed_json = None

        if json_start != -1:
            brace_level = 0
            for i, char in enumerate(raw_output_text[json_start:]):
                if char == '{':
                    brace_level += 1
                elif char == '}':
                    brace_level -= 1
                    if brace_level == 0:
                        json_end = json_start + i + 1
                        break

            if json_end != -1:
                json_string = raw_output_text[json_start:json_end]
                try:
                    parsed_json = json.loads(json_string)
                    result['parsed_json'] = parsed_json
                    logger.info(f"LLM Detail Extraction: Успешно извлечен JSON: {json_string}")

                    if not isinstance(parsed_json, dict):
                        raise ValueError("Распарсенный JSON не является словарем.")

                    # Извлекаем и валидируем значения
                    well_id = parsed_json.get('well_identifier')
                    lat = parsed_json.get('latitude')
                    lon = parsed_json.get('longitude')

                    result['well_identifier'] = str(well_id) if well_id is not None else None

                    # Пытаемся конвертировать координаты в float
                    try:
                        result['latitude'] = float(lat) if lat is not None else None
                    except (ValueError, TypeError):
                        logger.warning(f"Не удалось конвертировать latitude '{lat}' в float. Установлено None.")
                        result['latitude'] = None
                    try:
                        result['longitude'] = float(lon) if lon is not None else None
                    except (ValueError, TypeError):
                        logger.warning(f"Не удалось конвертировать longitude '{lon}' в float. Установлено None.")
                        result['longitude'] = None

                    logger.info(f"LLM Detail Extraction: JSON успешно распарсен и валидирован: "
                                f"ID={result['well_identifier']}, Lat={result['latitude']}, Lon={result['longitude']}")

                except json.JSONDecodeError as e:
                    result['error'] = f"LLM Detail Extraction: Ошибка декодирования JSON: {e}. Строка: {json_string}. Полный вывод: {raw_output_text}"
                    logger.error(result['error'])
                except ValueError as e:
                     result['error'] = f"LLM Detail Extraction: Ошибка валидации JSON: {e}. Распарсено: {result.get('parsed_json')}"
                     logger.error(result['error'])
                except Exception as e:
                     result['error'] = f"LLM Detail Extraction: Неожиданная ошибка парсинга/валидации: {e}. Вывод: {raw_output_text}"
                     logger.error(result['error'], exc_info=True)

            else: # json_end == -1
                result['error'] = "LLM Detail Extraction: Не найдена закрывающая '}' для JSON."
                logger.warning(f"{result['error']} Вывод: {raw_output_text}")
        else: # json_start == -1
            result['error'] = "LLM Detail Extraction: Не найдена открывающая '{' для JSON."
            logger.warning(f"{result['error']} Вывод: {raw_output_text}")

    except Exception as e:
        result['error'] = f"LLM Detail Extraction: Ошибка во время генерации ответа LLM: {e}"
        logger.error(result['error'], exc_info=True)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze DDR comments for problems using LLM.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Название или путь к LLM модели из Hugging Face Hub.")
    parser.add_argument("--use_4bit", action='store_true', help="Использовать 4-bit квантизацию.")
    parser.add_argument("--use_8bit", action='store_true', help="Использовать 8-bit квантизацию.")
    parser.add_argument("--comment_text", type=str,
                        default="During tripping out, encountered high torque and drag above the casing shoe. Worked pipe and circulated bottoms up, observed metal shavings. Suspect possible collapsed casing or junk in hole.",
                        help="Текст комментария для анализа.")
    parser.add_argument("--categories", nargs='+', default=DEFAULT_PROBLEM_CATEGORIES,
                         help="Список категорий проблем.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Макс. количество новых токенов для генерации.")

    args = parser.parse_args()
    
    use_4bit_flag = args.use_4bit if args.use_4bit or args.use_8bit else True
    use_8bit_flag = args.use_8bit if not use_4bit_flag else False
    
    model, tokenizer = load_llm_pipeline(args.model_name, use_4bit_flag, use_8bit_flag)
    
    if model and tokenizer:
        print(f"\n--- Анализ комментария ---")
        print(f"Модель: {args.model_name}")
        print(f"Текст комментария:\n---\n{args.comment_text}\n---")
        
        analysis_result = analyze_comment_problem(
            comment_text=args.comment_text,
            model=model,
            tokenizer=tokenizer,
            problem_categories=args.categories,
            max_new_tokens=args.max_new_tokens
        )
        
        print("\n--- Результат анализа ---")
        print(f"Проблема обнаружена: {analysis_result.get('problem_detected')}")
        print(f"Категория: {analysis_result.get('category')}")
        print(f"Ошибка: {analysis_result.get('error')}")
        print(f"Распарсенный JSON: {analysis_result.get('parsed_json')}")
        print(f"Сырой вывод модели:\n---\n{analysis_result.get('raw_output')}\n---")
    else:
        print("Не удалось загрузить модель или токенизатор. Анализ не выполнен.")