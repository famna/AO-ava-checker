import keyboard
import pyautogui
from PIL import Image, ImageOps 
import time
import json
import sys
import re
import os
from fuzzywuzzy import fuzz
import easyocr 
import numpy as np 

# --- КОНФИГУРАЦИЯ ---
try:
    print("[*] Загрузка EasyOCR Reader. Это может занять некоторое время при первом запуске...")
    reader = easyocr.Reader(['en']) 
    print("[*] EasyOCR Reader успешно загружен.")
except Exception as e:
    print(f"Ошибка при загрузке EasyOCR Reader: {e}")
    print("Убедитесь, что EasyOCR установлен ('pip install easyocr') и есть доступ к интернету для загрузки моделей.")
    sys.exit(1)


# Параметры области захвата вокруг курсора.
SCAN_WIDTH = 800  
SCAN_HEIGHT = 400 

# Порог уверенности для нечеткого сопоставления (fuzzywuzzy).
CONFIDENCE_THRESHOLD = 75 

# Количество вариантов совпадений для вывода
TOP_N_MATCHES = 3 # <-- НОВАЯ КОНСТАНТА

# --- ЗАГРУЗКА ДАННЫХ ПОРТАЛОВ ---
PORTAL_DATA_FILE = "avalon_portal_data.json"
portal_data = {}

try:
    with open(PORTAL_DATA_FILE, "r", encoding="utf-8") as f:
        portal_data = json.load(f)
    print(f"[*] База данных порталов '{PORTAL_DATA_FILE}' успешно загружена.")
except FileNotFoundError:
    print(f"Ошибка: Файл '{PORTAL_DATA_FILE}' не найден. Пожалуйста, создайте его.")
    print("Пример содержимого:")
    print("""
{
  "Hynes-Ieatun": "Сундук T7, Мобы: Элементали, Ресурсы: Руда",
  "Avalonian Rest": "Сундук T5, Ресурсы: Дерево, Руда",
  "Whispering Caves": "Сундук T6, Мобы: Пауки, Големы",
  "Xases-Ataglos": {"tier": "T7", "Синий": 1, "Голд": 0, "БигГолд": 1},
  "Ceritos-Avulsum": "Пример описания для Ceritos-Avulsum"
}
""")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Ошибка: Неправильный формат JSON в файле '{PORTAL_DATA_FILE}': {e}")
    sys.exit(1)


# --- ФУНКЦИЯ ДЛЯ ЗАХВАТА И РАСПОЗНАВАНИЯ ТЕКСТА ---
def get_text_from_screenshot_region():
    x, y = pyautogui.position()

    top_left_x_large = x - (SCAN_WIDTH // 2)
    top_left_y_large = y - (SCAN_HEIGHT // 2)
    large_screenshot_region = (top_left_x_large, top_left_y_large, SCAN_WIDTH, SCAN_HEIGHT)
    
    large_screenshot = pyautogui.screenshot(region=large_screenshot_region)
    large_screenshot.save("debug_large_centered_screenshot.png")
    print(f"Сохранен большой отладочный скриншот: debug_large_centered_screenshot.png (регион: {large_screenshot_region})")

    large_screenshot_np = np.array(large_screenshot) 

    results = reader.readtext(large_screenshot_np) 

    portal_tooltip_text = ""
    found_bbox = None

    all_raw_text_parts = [res[1] for res in results]
    full_raw_text = ' '.join(all_raw_text_parts) 
    print(f"EasyOCR полный сырой вывод: '{full_raw_text}'") 

    best_anchor_bbox = None
    best_anchor_text = ""
    highest_anchor_prob = 0.0

    anchor_keywords = ['road', 'rodd', 'codofatdlan', 'avalon', 'avolonto', 'atdlan', 'to', 't0']

    for (bbox, text, prob) in results:
        normalized_text = text.replace('\n', ' ').replace('\r', '').strip().lower()
        
        contains_keywords = any(keyword in normalized_text for keyword in anchor_keywords)
        
        if contains_keywords and prob > 0.1: 
            if ('avalon' in normalized_text or 'avolonto' in normalized_text or 'atdlan' in normalized_text) and \
               ('to' in normalized_text or 't0' in normalized_text):
                best_anchor_bbox = bbox
                best_anchor_text = text
                break 
            elif prob > highest_anchor_prob: 
                highest_anchor_prob = prob
                best_anchor_bbox = bbox
                best_anchor_text = text

    if best_anchor_bbox:
        print(f"Найден якорь для обрезки: '{best_anchor_text}' (Conf: {highest_anchor_prob:.2f})")
        x_coords = [p[0] for p in best_anchor_bbox]
        y_coords = [p[0] for p in best_anchor_bbox] # bug fix: should be y_coords
        y_coords = [p[1] for p in best_anchor_bbox]
        
        min_x, max_x = int(min(x_coords)), int(max(x_coords))
        min_y, max_y = int(min(y_coords)), int(max(y_coords))

        padding = 15 
        crop_left = max(0, min_x - padding)
        crop_top = max(0, min_y - padding)
        crop_right = min(large_screenshot.width, max_x + padding)
        crop_bottom = min(large_screenshot.height, max_y + padding)

        try:
            cropped_tooltip_final = large_screenshot.crop((crop_left, crop_top, crop_right, crop_bottom))
            cropped_tooltip_final.save("debug_found_tooltip_crop.png")
            print(f"Сохранен обрезанный тултип по найденным координатам: debug_found_tooltip_crop.png")
            
            cropped_tooltip_np = np.array(cropped_tooltip_final)
            re_ocr_results = reader.readtext(cropped_tooltip_np)
            portal_tooltip_text = ' '.join([res[1] for res in re_ocr_results])
            print(f"Повторный OCR на обрезанном тултипе: '{portal_tooltip_text}'")

        except Exception as e:
            print(f"Ошибка при обрезке тултипа по найденным координатам: {e}")
            portal_tooltip_text = full_raw_text
            print("Будет использоваться полный сырой текст для обработки.")
    else:
        print("Ошибка: Не удалось найти подходящий якорь для обрезки тултипа. Будет использоваться полный сырой текст.")
        portal_tooltip_text = full_raw_text


    # --- Этап очистки и нормализации текста ---
    text = portal_tooltip_text.replace('\n', ' ').replace('\r', '').strip() 
    text = ' '.join(text.split()) 
    
    text = re.sub(r'[^A-Za-z0-9\s-]', '', text) 

    text = text.replace("Road of Avolon to", "Road of Avalon to") 
    text = text.replace("Avolon", "Avalon") 
    text = text.replace("HynesIeatun", "Hynes-Ieatun") 
    text = text.replace("Xases Acaglos", "Xases-Ataglos") 
    text = text.replace("Xases-Acaglos", "Xases-Ataglos") 
    text = text.replace("Xases Ataglos", "Xases-Ataglos") 
    text = text.replace("XasesAtaglos", "Xases-Ataglos") 
    text = text.replace("Rodd afAvalon to", "Road of Avalon to")
    text = text.replace("Rodd ofAvalon to", "Road of Avalon to") 
    text = text.replace("t0", "to") 
    text = text.replace("to 2", "to") 
    
    # Новые замены из вашего последнего лога:
    text = text.replace("codofatdlan", "Road of Avalon to")
    text = text.replace("Qlltun-Vletls", "Ceritos-Avulsum") 
    text = text.replace("~>", "") 
    text = text.replace("01321", "") 

    text = re.sub(r'\b\d+\b', '', text).strip() 
    text = re.sub(r'\b[a-zA-Z]\b', '', text).strip() 

    text = re.sub(r'Im\s*Icloses In \d*', '', text, flags=re.IGNORECASE).strip() 
    text = re.sub(r'n/a', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'closes in \d+\s*[hm]\s*\d+s*', '', text, flags=re.IGNORECASE).strip() 
    text = re.sub(r'closes in \d+\s*[m]\s*\d+s*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'Closesin', '', text, flags=re.IGNORECASE).strip() 
    text = re.sub(r'\b\d+\s*/\s*\d+\b', '', text).strip() 
    text = re.sub(r'\s*cd@\s*', '', text).strip() 
    text = re.sub(r'S', '', text).strip() 
    text = re.sub(r'07 m', '', text).strip() 
    text = re.sub(r'Icloses In', '', text).strip()

    text = ' '.join(text.split()) 

    return text

# --- ФУНКЦИЯ ОБРАБОТКИ ГОРЯЧЕЙ КЛАВИШИ ---
def on_hotkey():
    print("\n[*] Обнаружение портала...")
    try:
        portal_text_detected = get_text_from_screenshot_region() 
        print(f"Финальный распознанный текст после очистки: '{portal_text_detected}'")

        if not portal_text_detected:
            print("Текст не распознан. Проверьте отладочные скриншоты и настройки обрезки/очистки.")
            return

        matches = [] # Список для хранения всех совпадений
        
        # Сначала пытаемся извлечь название локации после "Road of Avalon to "
        match_location = re.search(r'Road of Avalon to\s*([A-Za-z0-9\s-]+)', portal_text_detected, re.IGNORECASE)
        
        text_for_matching = portal_text_detected 
        if match_location:
            extracted_location_name = match_location.group(1).strip()
            print(f"Извлеченное название локации для сопоставления: '{extracted_location_name}'")
            text_for_matching = extracted_location_name 
        else:
            print("Не удалось извлечь название локации по шаблону 'Road of Avalon to ...'. Попытка сопоставления со всем текстом.")

        # Собираем все совпадения
        for name_in_db in portal_data:
            score = fuzz.token_set_ratio(name_in_db.lower(), text_for_matching.lower())
            if score >= CONFIDENCE_THRESHOLD: # Добавляем только те, что выше порога
                matches.append((name_in_db, score))

        # Сортируем совпадения по убыванию точности
        matches.sort(key=lambda x: x[1], reverse=True)

        if matches:
            print(f"Найдено {len(matches)} возможных совпадений (выводится топ {TOP_N_MATCHES}):")
            for i, (name, score) in enumerate(matches[:TOP_N_MATCHES]):
                print(f"  {i+1}. {name} (точность: {score}%)")
                # Выводим информацию о портале только для лучшего совпадения или всех?
                # Давайте выведем для лучшего, а для остальных только название и точность.
                if i == 0: # Для лучшего совпадения выводим полную информацию
                    if isinstance(portal_data[name], dict):
                        for key, value in portal_data[name].items():
                            print(f"     {key}: {value}")
                    else:
                        print(f"     Содержимое: {portal_data[name]}")
        else:
            print(f"Портал не найден в базе данных (лучшее совпадение: {best_match_name if 'best_match_name' in locals() else 'N/A'}, точность: {best_match_score if 'best_match_score' in locals() else 'N/A'}%).")

    except Exception as e:
        print(f"Произошла ошибка в функции on_hotkey: {e}")
        import traceback
        traceback.print_exc()

# --- ОСНОВНОЙ ЦИКЛ ПРОГРАММЫ ---
try:
    keyboard.add_hotkey('ctrl+d', on_hotkey)
    print("\n=== Albion Online Overlay Helper запущен ===")
    print(f"Нажмите Ctrl+D для сканирования портала (курсор должен быть наведен на портал).")
    print("Нажмите Esc для выхода.")
    keyboard.wait('esc')
    print("Выход из помощника оверлея.")

except Exception as e:
    print(f"\nПроизошла непредвиденная ошибка в основном цикле программы: {e}")
    import traceback
    traceback.print_exc()
    input("Нажмите Enter для выхода...")