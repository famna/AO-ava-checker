import keyboard
import pyautogui
from PIL import Image, ImageOps, ImageEnhance 
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
TOP_N_MATCHES = 3 

# Порог для бинаризации (0-255). Пиксели темнее этого значения станут черными, светлее - белыми.
# Вам, возможно, придется настроить это значение.
# Если текст белый/светлый на темном фоне, попробуйте более высокое значение (например, 200-240).
# Если текст темный на светлом фоне, попробуйте более низкое значение (например, 80-120).
BINARIZATION_THRESHOLD = 160 

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


# --- ФУНКЦИЯ ДЛЯ БАЗОВОЙ ОЧИСТКИ И НОРМАЛИЗАЦИИ ТЕКСТА (без специфических замен OCR) ---
def clean_base_text(text_input):
    text = text_input.replace('\n', ' ').replace('\r', '').strip() 
    text = ' '.join(text.split()) 
    text = re.sub(r'[^A-Za-z0-9\s-]', '', text) 

    # Удаляем части, которые, как мы знаем, не относятся к названию портала
    text = re.sub(r'FPS: \d+ Ping: \d+\.?\d*', '', text, flags=re.IGNORECASE).strip() 
    text = re.sub(r'Im\s*Icloses In \d*', '', text, flags=re.IGNORECASE).strip() 
    text = re.sub(r'n/a', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'Ecloses In \d+\s*[hm]\s*\d+s*', '', text, flags=re.IGNORECASE).strip() 
    text = re.sub(r'closes In \d+\s*[hm]\s*\d+s*', '', text, flags=re.IGNORECASE).strip() 
    text = re.sub(r'closes in \d+\s*[m]\s*\d+s*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'Closesin', '', text, flags=re.IGNORECASE).strip() 
    text = re.sub(r'\b\d+\s*/\s*\d+\b', '', text).strip() 
    text = re.sub(r'\s*cd@\s*', '', text).strip() 
    text = re.sub(r'S', '', text).strip() 
    text = re.sub(r'07 m', '', text).strip() 
    text = re.sub(r'Icloses In', '', text).strip()
    text = re.sub(r'FP Ping', '', text).strip() 
    text = re.sub(r'Qures-Arddsum=', '', text).strip() 
    text = re.sub(r'\b\d+\s*h\s*\d+\s*m\b', '', text).strip() 
    text = re.sub(r'\b\d+\s*m\b', '', text).strip() 
    
    # Удаление всех одиночных чисел или одиночных букв, если они не являются частью слова.
    # Это должно быть после других замен, чтобы не конфликтовать.
    text = re.sub(r'\b\d+\b', '', text).strip() 
    text = re.sub(r'\b[a-zA-Z]\b', '', text).strip() 
    
    # Специфические замены, которые не являются "ошибками OCR", а скорее типичными ошибками Albion OCR
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
    text = text.replace("codofatdlan", "Road of Avalon to")
    text = text.replace("Rond ofAvolan", "Road of Avalon to") 
    text = text.replace("Roudof Avolon", "Road of Avalon to") 
    text = text.replace("Oynltes-Aroosum", "Hynes-Araosum") 
    text = text.replace("Qures-Arddsum", "Hynes-Araosum") 
    text = text.replace("Conos-Avoelum", "Ceritos-Avulsum") 
    text = text.replace("~>", "") 
    text = text.replace("01321", "") 
    text = text.replace("37", "") # ОСТОРОЖНО: удаляет конкретное число, если оно появляется (можно убрать, если число может быть частью имени)

    return ' '.join(text.split()) # Финальная нормализация пробелов

# --- ФУНКЦИЯ ДЛЯ ГЕНЕРАЦИИ ВАРИАНТОВ ТЕКСТА С ОБЩИМИ ОШИБКАМИ OCR ---
def generate_ocr_mistake_variants(base_text):
    variants = {base_text} # Используем set для автоматического удаления дубликатов
    
    # Определяем "подозрительные" пары символов, которые EasyOCR часто путает
    # Эти замены будут применяться для генерации *новых* вариантов текста.
    ocr_replacements_map = {
        'I': 'l', 'l': 'I',      # Capital I and lowercase L
        'u': 'w', 'w': 'u',      # u and w (often confused due to shape)
        'a': 'o', 'o': 'a',      # a and o (round shapes)
        'O': '0', '0': 'O',      # Capital O and zero
        'Z': '2', '2': 'Z',      # Z and 2
        'S': '5', '5': 'S',      # S and 5
        'B': '8', '8': 'B',      # B and 8
        'G': '6', '6': 'G',      # G and 6
        'q': 'g', 'g': 'q',      # q and g
        'f': 't', 't': 'f',      # Added f and t as requested
        'c': 'e', 'e': 'c',      # c and e
        'v': 'u', 'u': 'v',      # v and u (similar in some fonts)
        'Y': 'V', 'V': 'Y',      # Y and V
        'r': 'n', 'n': 'r',      # r and n (especially if 'r' is short or missing leg)
        'm': 'rn', 'rn': 'm',    # m and r n (common composite error)
        'd': 'cl', 'cl': 'd',    # d and c l (common composite error)
        'l': 'li', 'li': 'l',    # l and l i
        'j': 'i', 'i': 'j',      # j and i
        'h': 'b', 'b': 'h',      # h and b
        'p': 'b', 'b': 'p'       # p and b
    }

    # Итерируем по существующим вариантам и применяем замены
    current_variants = list(variants) # Копируем, чтобы избежать изменения во время итерации
    for text_variant in current_variants:
        for old_char, new_char in ocr_replacements_map.items():
            if old_char in text_variant:
                new_variant = text_variant.replace(old_char, new_char)
                variants.add(new_variant) # Добавляем новый вариант в set

    return list(variants)

# --- ФУНКЦИЯ ДЛЯ ЗАХВАТА И РАСПОЗНАВАНИЯ ТЕКСТА ---
def get_processed_ocr_texts_with_variants():
    x, y = pyautogui.position()

    top_left_x_large = x - (SCAN_WIDTH // 2)
    top_left_y_large = y - (SCAN_HEIGHT // 2)
    large_screenshot_region = (top_left_x_large, top_left_y_large, SCAN_WIDTH, SCAN_HEIGHT)
    
    large_screenshot = pyautogui.screenshot(region=large_screenshot_region)
    large_screenshot.save("debug_large_centered_screenshot.png")
    print(f"Сохранен большой отладочный скриншот: debug_large_centered_screenshot.png (регион: {large_screenshot_region})")

    # --- Бинаризация изображения ---
    gray_screenshot = large_screenshot.convert('L')
    binarized_screenshot = gray_screenshot.point(lambda p: 255 if p > BINARIZATION_THRESHOLD else 0)
    binarized_screenshot.save("debug_binarized_screenshot.png")
    print("Сохранен бинаризованный скриншот: debug_binarized_screenshot.png")
    
    large_screenshot_np = np.array(binarized_screenshot) 
    results_full_screen = reader.readtext(large_screenshot_np) 

    all_raw_text_parts_full = [res[1] for res in results_full_screen]
    full_raw_text_combined = ' '.join(all_raw_text_parts_full) 
    print(f"EasyOCR полный сырой вывод (весь экран): '{full_raw_text_combined}'") 

    potential_base_texts = [] # Список для хранения базовых очищенных текстов (до генерации вариантов)

    best_anchor_bbox = None
    best_anchor_text = ""
    highest_anchor_prob = 0.0

    anchor_keywords = ['road', 'rodd', 'codofatdlan', 'rond', 'avalon', 'avolonto', 'atdlan', 'to', 't0', 'avolon', 'roudof']

    for (bbox, text, prob) in results_full_screen:
        normalized_text = text.replace('\n', ' ').replace('\r', '').strip().lower()
        contains_keywords = any(keyword in normalized_text for keyword in anchor_keywords)
        
        if contains_keywords and prob > 0.1: 
            if (any(kw in normalized_text for kw in ['avalon', 'avolonto', 'atdlan', 'avolon'])) and \
               (any(kw in normalized_text for kw in ['to', 't0'])):
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
        y_coords = [p[1] for p in best_anchor_bbox] 
        
        min_x, max_x = int(min(x_coords)), int(max(x_coords))
        min_y, max_y = int(min(y_coords)), int(max(y_coords))

        padding = 15 
        crop_left = max(0, min_x - padding)
        crop_top = max(0, min_y - padding)
        crop_right = min(large_screenshot.width, max_x + padding)
        crop_bottom = min(large_screenshot.height, max_y + padding)

        try:
            cropped_tooltip_initial = large_screenshot.crop((crop_left, crop_top, crop_right, crop_bottom))
            cropped_tooltip_final = cropped_tooltip_initial.convert('L').point(lambda p: 255 if p > BINARIZATION_THRESHOLD else 0)

            cropped_tooltip_final.save("debug_found_tooltip_crop.png")
            print(f"Сохранен обрезанный и бинаризованный тултип по найденным координатам: debug_found_tooltip_crop.png")
            
            cropped_tooltip_np = np.array(cropped_tooltip_final)
            re_ocr_results = reader.readtext(cropped_tooltip_np)
            
            for (bbox, text, prob) in re_ocr_results:
                cleaned_text = clean_base_text(text)
                if cleaned_text:
                    potential_base_texts.append(cleaned_text)
            print(f"OCR на обрезанном тултипе дал базовые тексты: {potential_base_texts}")

        except Exception as e:
            print(f"Ошибка при обрезке тултипа по найденным координатам: {e}")
            # Если обрезка не удалась, используем полный сырой текст как запасной вариант
            cleaned_full_raw_text = clean_base_text(full_raw_text_combined)
            if cleaned_full_raw_text:
                potential_base_texts.append(cleaned_full_raw_text)
            print("Будет использоваться полный сырой текст для обработки.")
    else:
        print("Ошибка: Не удалось найти подходящий якорь для обрезки тултипа. Будет использоваться полный сырой текст.")
        cleaned_full_raw_text = clean_base_text(full_raw_text_combined)
        if cleaned_full_raw_text:
            potential_base_texts.append(cleaned_full_raw_text)

    # В любом случае, добавляем весь распознанный текст с начального большого скриншота после базовой очистки
    cleaned_initial_full_text = clean_base_text(full_raw_text_combined)
    if cleaned_initial_full_text and cleaned_initial_full_text not in potential_base_texts:
        potential_base_texts.insert(0, cleaned_initial_full_text) # Добавляем в начало как приоритетный вариант

    final_processed_texts_with_variants = set()
    for base_text in potential_base_texts:
        variants_for_this_text = generate_ocr_mistake_variants(base_text)
        for variant in variants_for_this_text:
            final_processed_texts_with_variants.add(variant)

    return list(final_processed_texts_with_variants)


# --- ФУНКЦИЯ ОБРАБОТКИ ГОРЯЧЕЙ КЛАВИШИ ---
def on_hotkey():
    print("\n[*] Обнаружение портала...")
    try:
        # Теперь получаем список потенциально распознанных текстов с вариантами OCR ошибок
        potential_portal_texts_and_variants = get_processed_ocr_texts_with_variants() 
        
        if not potential_portal_texts_and_variants:
            print("Текст не распознан. Проверьте отладочные скриншоты и настройки обрезки/очистки.")
            return

        print(f"Все потенциальные очищенные тексты и их варианты для обработки ({len(potential_portal_texts_and_variants)}): {potential_portal_texts_and_variants}")

        all_matches = [] # Список для сбора всех совпадений из разных вариантов

        # Регулярное выражение для извлечения названия портала.
        portal_name_pattern = re.compile(
            r'(?:Road of Avalon to|Rond ofAvolan|Roudof Avolon|codofatdlan)\s*'
            r'([A-Za-z-]+(?:[ -][A-Za-z-]+)*)' 
            r'(?=\s*(?:closes in|Ecloses In|FPS|Ping|$))', 
            re.IGNORECASE
        )

        for portal_text_variant in potential_portal_texts_and_variants:
            extracted_location_name = None
            
            match = portal_name_pattern.search(portal_text_variant)
            
            if match:
                extracted_location_name = match.group(1).strip()
                # print(f"Извлечено название локации из варианта '{portal_text_variant}': '{extracted_location_name}'") # Слишком много вывода
            else:
                # Если регулярка не сработала, берем весь очищенный текст как потенциальное имя.
                extracted_location_name = portal_text_variant 
                # print(f"Не удалось извлечь название локации по шаблону. Используется весь вариант текста: '{extracted_location_name}'") # Слишком много вывода

            if not extracted_location_name:
                continue # Пропускаем пустые строки

            for name_in_db in portal_data:
                score = fuzz.token_set_ratio(name_in_db.lower(), extracted_location_name.lower())
                if score >= CONFIDENCE_THRESHOLD: 
                    all_matches.append((name_in_db, score))

        # Удаляем дубликаты и сортируем по убыванию точности
        unique_matches = list(set(all_matches))
        unique_matches.sort(key=lambda x: x[1], reverse=True)


        if unique_matches:
            print(f"Найдено {len(unique_matches)} возможных уникальных совпадений (выводится топ {TOP_N_MATCHES}):")
            for i, (name, score) in enumerate(unique_matches[:TOP_N_MATCHES]):
                print(f"  {i+1}. {name} (точность: {score}%)")
                if i == 0: 
                    if isinstance(portal_data[name], dict):
                        for key, value in portal_data[name].items():
                            print(f"     {key}: {value}")
                    else:
                        print(f"     Содержимое: {portal_data[name]}")
        else:
            print(f"Портал не найден в базе данных (лучшее совпадение: N/A, точность: N/A%).")


    except Exception as e:
        print(f"Произошла ошибка в функции on_hotkey: {e}")
        import traceback
        traceback.print_exc()

# --- ОСНОВНОЙ ЦИКЛ ПРОГРАММЫ ---
try:
    keyboard.add_hotkey('ctrl+d', on_hotkey)
    print("\n=== Albion Online Overlay Helper запущен ===")
    print(f"Нажмите Ctrl+D для сканирования портала (курсор должен быть наведен на портал).")
    print("Нажмите Insert для выхода.") # Обновил сообщение для пользователя
    keyboard.wait('insert') # <--- ИЗМЕНЕНО ЗДЕСЬ
    print("Выход из помощника оверлея.")

except Exception as e:
    print(f"\nПроизошла непредвиденная ошибка в основном цикле программы: {e}")
    import traceback
    traceback.print_exc()
    input("Нажмите Enter для выхода...")