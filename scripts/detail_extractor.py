import re 
import logging
from typing import List, Dict, Optional, Tuple, Any
import spacy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

NLP = None
SPACY_MODEL_NAME = "en_core_web_sm"

def load_spacy_model(model_name: str=SPACY_MODEL_NAME):
    """Загружает модель spaCy, если она еще не загружена."""
    global NLP
    if NLP is None:
        try:
            logger.info(f"Загрузка spaCy модели: {model_name}...")
            NLP = spacy.load(model_name)
            logger.info(f"spaCy модель {model_name} успешно загружена.")
        except OSError:
            logger.error(f"Модель spaCy '{model_name}' не найдена. "
                         f"Установите ее: python -m spacy download {model_name}")
            NLP = "error" # Помечаем, что была ошибка загрузки
        except Exception as e:
            logger.error(f"Неожиданная ошибка при загрузке spaCy модели: {e}", exc_info=True)
            NLP = "error"
    return NLP if NLP != "error" else None

def extract_well_details(well_info_text: str, use_spacy: bool=True) -> Dict[str, Optional[Any]]:
    """
    Извлекает ID/имя скважины и координаты из предоставленного текста секции well_information.

    Args:
        well_info_text: Текст секции 'well_information'.
        use_spacy: Использовать ли spaCy NER в дополнение к регулярным выражениям (рекомендуется).

    Returns:
        Словарь с извлеченными деталями:
        {
            'well_identifier': str | None, # Найденный ID или имя
            'latitude': float | None,      # Широта в десятичных градусах
            'longitude': float | None      # Долгота в десятичных градусах
        }
    """
    if not well_info_text or not well_info_text.strip():
        logger.warning("Входной текст пуст. Извлечение не выполнено.")
        return {'well_identifier': None, 'latitude': None, 'longitude': None}
    
    results: Dict[str, Optional[Any]] = {
        'well_identifier': None,
        'latitude': None,
        'longitude': None
    }
    
    logger.debug("Поиск деталей с помощью регулярных выражений...")
    
    well_id_patterns = [
        re.compile(r"(?:Well Name|Well ID|Well)\s*[:\s]+\s*([^\n\r]+)", re.IGNORECASE),
        re.compile(r"^\s*([A-Za-z0-9\s\-\.]+)\s*Well\s*$", re.IGNORECASE | re.MULTILINE)   
    ]
    
    for pattern in well_id_patterns:
        match = pattern.search(well_info_text)
        if match:
            identifier = match.group(1).strip()
            identifier = re.sub(r'\s+', ' ', identifier)
            identifier = identifier.replace("Planned Operations", "").replace("Current Operations", "").strip()
            if identifier:
                results['well_identifier'] = identifier
                logger.info(f"Найдено с помощью regex: well_identifier = '{identifier}'")
                break
            
    # coord_pattern_decimal = re.compile(
    #     r"""
    #     (?:Lat(?:itude)?)\s*[:\s]+      # Ищем Lat или Latitude
    #     (-?[\d\.]+)                     # Группа 1: Значение широты (возможно с минусом)
    #     (?:\s*°?\s*([NS])?)?            # Группа 2: Опциональный знак N/S
    #     \s*[,\s/]+\s*                   # Разделитель
    #     (?:Lon(?:gitude)?)\s*[:\s]+     # Ищем Lon или Longitude
    #     (-?[\d\.]+)                     # Группа 3: Значение долготы (возможно с минусом)
    #     (?:\s*°?\s*([EW])?)?            # Группа 4: Опциональный знак E/W
    #     """,
    #     re.IGNORECASE | re.VERBOSE
    # )
    coord_patterns = [
        # Существующий шаблон
        re.compile(r"""
            (?:Lat(?:itude)?)\s*[:\s]+      # Ищем Lat или Latitude
            (-?[\d\.]+)                     # Группа 1: Значение широты
            (?:\s*°?\s*([NS])?)?            # Группа 2: Опциональный знак N/S
            \s*[,\s/]+\s*                   # Разделитель
            (?:Lon(?:gitude)?)\s*[:\s]+     # Ищем Lon или Longitude
            (-?[\d\.]+)                     # Группа 3: Значение долготы
            (?:\s*°?\s*([EW])?)?            # Группа 4: Опциональный знак E/W
            """, re.IGNORECASE | re.VERBOSE),
        
        # Новый шаблон для случаев без разделителя между Lat и Lon
        re.compile(r"""
            (?:Lat(?:itude)?)[:\s]+(-?[\d\.]+)(?:\s*°?\s*([NS])?)?
            (?:\s*(?:Lon(?:gitude)?)[:\s]+(-?[\d\.]+)(?:\s*°?\s*([EW])?)?)
            """, re.IGNORECASE | re.VERBOSE)
    ]
    
    for pattern in coord_patterns:
        coord_match = pattern.search(well_info_text)
        if coord_match:
            try:
                lat_val = float(coord_match.group(1))
                lon_val = float(coord_match.group(3))
                lat_sign = coord_match.group(2)
                lon_sign = coord_match.group(4)
                
                if lat_sign and lat_sign.upper() == 'S':
                    lat_val = -abs(lat_val)
                if lon_sign and lon_sign.upper() == 'W':
                    lon_val = -abs(lon_val)
                    
                results['latitude'] = lat_val
                results['longitude'] = lon_val
                logger.info(f"Найдены с помощью regex: latitude = {lat_val}, longitude = {lon_val}")

            except (ValueError, TypeError) as e:
                logger.warning(f"Ошибка преобразования координат, найденных regex: {coord_match.groups()}. Ошибка: {e}")
            except Exception as e:
                logger.error(f"Неожиданная ошибка при обработке координат из regex: {e}", exc_info=True)
             
    if use_spacy and results['well_identifier'] is None:
        logger.debug("Попытка найти идентификатор скважины с помощью spaCy NER...")
        nlp_model = load_spacy_model()
        
        if nlp_model:
            try:
                doc = nlp_model(well_info_text)
                potential_names = []
                for ent in doc.ents:
                    if ent.label_ in ["FAC", "ORG", "GPE", "LOC"]:
                        text = ent.text.strip()
                        if len(text) > 2 and text.upper() not in ["USA", "UK", "CANADA", "TEXAS", "ALBERTA", "NORTH SEA"] and "Latitude" not in text and "Longitude" not in text:
                            potential_names.append(text)
                
                if potential_names:
                    results['well_identifier'] = potential_names[0]
                    logger.info(f"Найдено с помощью spaCy: potential well_identifier = '{potential_names[0]}'")
                else:
                    logger.info("spaCy NER не нашел подходящих кандидатов для идентификатора скважины.")

            except Exception as e:
                 logger.error(f"Ошибка при обработке текста с помощью spaCy: {e}", exc_info=True)
        else:
             logger.warning("Модель spaCy не загружена, пропуск анализа NER.")
             
    logger.info(f"Итоговый результат извлечения: {results}")
    return results

if __name__ == "__main__":
    sample_text_1 = """
    Well: Example Well 1-FX
    Location: West Texas Basin
    Rig: SuperDrill Rig 5
    Coordinates: Lat: 31.8765, Lon: -102.3456
    Spud Date: 2023-01-15
    Current Depth: 12,500 ft
    """

    sample_text_2 = """
    REPORT FOR Alpha Beta Gamma WELL
    Company: DrillCorp
    Field: Green Field
    Latitude: 45.123 N Longitude: 110.456 W
    API No: 12-345-67890
    """

    sample_text_3 = """
    Project: Deep Horizon Exploration
    Vessel: Ocean Explorer
    Water Depth: 1500m
    Well ID : DH-EXP-003-A
    Pos: LAT 28.7366 ° N / LONG 88.3658 ° W
    """

    sample_text_4 = """
    Rig : Noble Discoverer
    Prospect : Chukchi Sea Prospect A
    Measured Depth : 5000 ft
    No specific coordinates mentioned in this section.
    """
    
    sample_text_empty = ""

    print("--- Тест 1 ---")
    details1 = extract_well_details(sample_text_1)
    print(f"Извлечено: {details1}")

    print("\n--- Тест 2 ---")
    details2 = extract_well_details(sample_text_2)
    print(f"Извлечено: {details2}")

    print("\n--- Тест 3 ---")
    details3 = extract_well_details(sample_text_3)
    print(f"Извлечено: {details3}")
    
    print("\n--- Тест 4 (без координат) ---")
    details4 = extract_well_details(sample_text_4)
    print(f"Извлечено: {details4}")

    print("\n--- Тест 5 (пустой текст) ---")
    details_empty = extract_well_details(sample_text_empty)
    print(f"Извлечено: {details_empty}")

    print("\n--- Тест 6 (без spaCy) ---")
    details_no_spacy = extract_well_details(sample_text_2, use_spacy=False)
    print(f"Извлечено (без spaCy): {details_no_spacy}")