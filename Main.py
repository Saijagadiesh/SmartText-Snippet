import cv2
import numpy as np
from PIL import ImageGrab
import os
import time
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from pynput import keyboard
import threading
import pytesseract
import pyperclip
import tkinter as tk
from tkinter import font as tkfont
import re
from collections import Counter

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print("[+] EasyOCR available - will use for better text spacing")
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[-] EasyOCR not available. For better text spacing, install with: pip install easyocr")

try:
    import nltk
    from nltk.corpus import words
    from nltk.metrics import edit_distance
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        print("[*] Downloading NLTK word corpus...")
        nltk.download('words', quiet=True)
        nltk.download('punkt', quiet=True)
    ENGLISH_WORDS = set(words.words())
    NLTK_AVAILABLE = True
    print("[+] NLTK available - will use for spell correction")
except ImportError:
    NLTK_AVAILABLE = False
    ENGLISH_WORDS = set()
    print("[-] NLTK not available. For spell correction, install with: pip install nltk")

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
        print("[+] spaCy available - will use for context-aware processing")
    except OSError:
        print("[*] spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False
    print("[-] spaCy not available. For advanced NLP, install with: pip install spacy")

try:
    pytesseract.get_tesseract_version()
    print("[+] Tesseract OCR is available")
except Exception:
    print("[-] WARNING: Tesseract OCR not found!")
    print("    Please install from: https://github.com/UB-Mannheim/tesseract/wiki")

# --- Global state ---
start_point = None
end_point = None
cropping = False
cropped_img = None
screenshot = None


def mouse_crop(event, x, y, flags, param):
    global start_point, end_point, cropping, cropped_img
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        cropping = True
    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        end_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        cropping = False
        x1, y1 = start_point
        x2, y2 = end_point
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        cropped_img = screenshot[y_min:y_max, x_min:x_max]
        cv2.destroyAllWindows()


def preprocess_image_for_ocr(img, strategy='default'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scale_factor = 3.0
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    if strategy == 'code':
        if np.mean(gray) < 127:
            gray = cv2.bitwise_not(gray)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        _, processed = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_tiny = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_tiny)
        return processed

    if strategy == 'sharp':
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel_sharpen)
        _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif strategy == 'denoise':
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        processed = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif strategy == 'contrast':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        processed = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        denoised = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
        kernel_sharpen = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
        processed = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)

    kernel_clean = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_clean)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_clean)
    return processed


def detect_code_image(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_brightness = np.mean(gray)
        edges = cv2.Canny(gray, 50, 150)
        vertical_kernel = np.ones((10, 1), np.uint8)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)
        vertical_score = np.sum(vertical_lines) / vertical_lines.size
        is_code = mean_brightness < 100 or vertical_score > 0.01
        if is_code:
            print("[+] Detected code-like structure - using code-optimized OCR")
        return is_code
    except:
        return False


def perform_ocr(image, preserve_layout=False):
    try:
        is_code = detect_code_image(image)
        if is_code or preserve_layout:
            strategies = ['code', 'sharp', 'default']
        else:
            strategies = ['default', 'sharp', 'denoise', 'contrast']
        all_results = []

        for strategy in strategies:
            processed_img = preprocess_image_for_ocr(image, strategy=strategy)
            if is_code or preserve_layout:
                psm_modes = [4, 6, 3]
            else:
                psm_modes = [6, 3, 11, 4]

            for psm in psm_modes:
                if is_code or preserve_layout:
                    config = f'--oem 3 --psm {psm} -c preserve_interword_spaces=1 -c textord_space_size_is_variable=1'
                else:
                    config = f'--oem 3 --psm {psm} -c preserve_interword_spaces=1'
                try:
                    data = pytesseract.image_to_data(processed_img, config=config, output_type=pytesseract.Output.DICT)
                    text_parts = []
                    confidences = []
                    for i, conf in enumerate(data['conf']):
                        if int(conf) > 30:
                            text_parts.append(data['text'][i])
                            confidences.append(int(conf))
                    text = ' '.join(text_parts)
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    if text.strip() and len(text.split()) > 0:
                        all_results.append({
                            'text': text,
                            'confidence': avg_confidence,
                            'word_count': len(text.split()),
                            'strategy': strategy,
                            'psm': psm
                        })
                except Exception:
                    continue

        if all_results:
            best_result = max(all_results, key=lambda x: (x['confidence'], x['word_count']))
            best_text = best_result['text']
            print(f"[+] Best Tesseract result: strategy={best_result['strategy']}, psm={best_result['psm']}, confidence={best_result['confidence']:.1f}%")
            return best_text, best_result['confidence']
        else:
            processed_img = preprocess_image_for_ocr(image)
            return pytesseract.image_to_string(processed_img, config='--oem 3 --psm 6'), 0.0

    except Exception as e:
        print(f"[-] OCR Error: {e}")
        return None, 0.0


def spell_correct_word(word):
    if not NLTK_AVAILABLE or len(word) < 2:
        return word
    if word.lower() in ENGLISH_WORDS:
        return word
    word_lower = word.lower()
    candidates = []
    for dict_word in ENGLISH_WORDS:
        if abs(len(dict_word) - len(word_lower)) <= 2:
            distance = edit_distance(word_lower, dict_word)
            if distance <= 2:
                candidates.append((dict_word, distance))
    if candidates:
        candidates.sort(key=lambda x: x[1])
        best_match = candidates[0][0]
        if word.isupper():
            return best_match.upper()
        elif word[0].isupper():
            return best_match.capitalize()
        else:
            return best_match
    return word


def enhance_text_with_spacy(text):
    if not SPACY_AVAILABLE or not text:
        return text
    try:
        doc = nlp(text)
        result = ""
        for i, token in enumerate(doc):
            if i == 0:
                result += token.text
            elif token.is_punct or token.text in ["'", '"', ')', ']', '}']:
                result += token.text
            elif doc[i-1].text in ['(', '[', '{', '"', "'"]:
                result += token.text
            else:
                result += " " + token.text
        return result.strip()
    except Exception as e:
        print(f"[-] spaCy processing error: {e}")
        return text


def post_process_text(text, preserve_code_structure=False):
    if not text:
        return text
    if preserve_code_structure:
        text = text.replace('|', 'I')
        text = text.replace('\u041e', 'O')
        text = text.replace('\u0410', 'A')
        return text
    text = text.strip()
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([.!?,;:])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([)\]}"])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])([\[({])', r'\1 \2', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    if text:
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    if NLTK_AVAILABLE:
        words_list = text.split()
        corrected_words = []
        for word in words_list:
            if word.isalpha() and len(word) > 2:
                corrected_words.append(spell_correct_word(word))
            else:
                corrected_words.append(word)
        text = ' '.join(corrected_words)
    if SPACY_AVAILABLE:
        text = enhance_text_with_spacy(text)
    return text.strip()


def perform_ocr_easyocr(image):
    try:
        if not EASYOCR_AVAILABLE:
            return None, 0.0
        strategies = ['default', 'sharp', 'contrast']
        best_text = ""
        best_confidence = 0
        for strategy in strategies:
            processed_img = preprocess_image_for_ocr(image, strategy=strategy)
            reader = easyocr.Reader(['en'], gpu=False)
            if len(processed_img.shape) == 2:
                image_rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
            elif len(processed_img.shape) == 3:
                image_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            else:
                continue
            results = reader.readtext(image_rgb, paragraph=False, detail=1,
                                      text_threshold=0.5, low_text=0.3)
            sorted_results = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))
            text_parts = []
            confidences = []
            prev_y = None
            for (bbox, text, confidence) in sorted_results:
                if confidence > 0.4:
                    curr_y = bbox[0][1]
                    if prev_y is not None and abs(curr_y - prev_y) > 20:
                        text_parts.append('\n')
                    text_parts.append(text)
                    confidences.append(confidence)
                    prev_y = curr_y
            extracted_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            if avg_confidence > best_confidence and extracted_text.strip():
                best_confidence = avg_confidence
                best_text = extracted_text
        if best_text:
            print(f"[+] EasyOCR confidence: {best_confidence*100:.1f}%")
            return post_process_text(best_text), best_confidence * 100
        return None, 0.0
    except Exception as e:
        print(f"[-] EasyOCR Error: {e}")
        return None, 0.0


def perform_ocr_combined(image):
    """Try multiple OCR methods and return (best_text, method_name, confidence)"""
    results = []
    is_code = detect_code_image(image)

    if EASYOCR_AVAILABLE and not is_code:
        easyocr_text, easyocr_conf = perform_ocr_easyocr(image)
        if easyocr_text and len(easyocr_text.split()) > 1:
            results.append(("EasyOCR", easyocr_text, len(easyocr_text.split()), easyocr_conf))

    tesseract_text, tesseract_conf = perform_ocr(image, preserve_layout=is_code)
    if tesseract_text and len(tesseract_text.split()) > 1:
        tesseract_text = post_process_text(tesseract_text, preserve_code_structure=is_code)
        results.append(("Tesseract", tesseract_text, len(tesseract_text.split()), tesseract_conf))

    if results:
        best = max(results, key=lambda x: x[2])
        method, text, word_count, conf = best
        print(f"[+] Using {method} OCR result ({word_count} words detected)")
        return text, method, conf

    if EASYOCR_AVAILABLE:
        text, conf = perform_ocr_easyocr(image)
        if text:
            return text, "EasyOCR", conf

    text, conf = perform_ocr(image)
    return text, "Tesseract", conf


# ─────────────────────────────────────────────────────────────────────────────
#  Concept C — Stats card GUI
# ─────────────────────────────────────────────────────────────────────────────

def show_text_in_gui(extracted_text, ocr_method="Tesseract", confidence=0.0):
    """Display extracted text in a modern stats-card style GUI (Concept C)."""

    word_count = len(extracted_text.split()) if extracted_text.strip() else 0
    char_count = len(extracted_text.strip())

    # ── Palette ───────────────────────────────────────────────────────────────
    BG_PAGE    = "#F5F4F0"
    BG_CARD    = "#FFFFFF"
    BG_SURFACE = "#F1EFE8"
    BG_COPY    = "#042C53"
    FG_COPY    = "#B5D4F4"

    BLUE_BAR   = "#378ADD"
    TEAL_BAR   = "#1D9E75"

    TEXT_PRI   = "#1A1A1A"
    TEXT_SEC   = "#5F5E5A"
    TEXT_TER   = "#888780"
    BORDER     = "#D3D1C7"

    CONF_HIGH  = "#1D9E75"
    CONF_MED   = "#BA7517"
    CONF_LOW   = "#A32D2D"

    TOAST_BG   = "#085041"
    TOAST_FG   = "#9FE1CB"

    W, H = 460, 490  # wider + taller for breathing room

    root = tk.Tk()
    root.title("OCR — text extracted")
    root.geometry(f"{W}x{H}")
    root.resizable(False, False)
    root.configure(bg=BG_PAGE)
    root.attributes("-topmost", True)
    root.after(200, lambda: root.attributes("-topmost", False))

    root.update_idletasks()
    sx = (root.winfo_screenwidth()  - W) // 2
    sy = (root.winfo_screenheight() - H) // 2
    root.geometry(f"{W}x{H}+{sx}+{sy}")

    # ── Fonts ─────────────────────────────────────────────────────────────────
    def fn(size, weight="normal"):
        return tkfont.Font(family="Segoe UI", size=size, weight=weight)

    F_CARD     = fn(10, "bold")
    F_SUB      = fn(8)
    F_BODY     = fn(10)
    F_STAT_VAL = fn(13, "bold")
    F_STAT_LBL = fn(8)
    F_BTN      = fn(9)
    F_COPY     = fn(10, "bold")

    # ── Canvas ────────────────────────────────────────────────────────────────
    canvas = tk.Canvas(root, width=W, height=H, bg=BG_PAGE, highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    CARD_X, CARD_Y = 20, 20
    CARD_W, CARD_H = W - 40, H - 40
    R   = 12
    PAD = 20

    def rounded_rect(cv, x1, y1, x2, y2, r, **kw):
        pts = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r,
               x2,y2-r, x2,y2, x2-r,y2, x1+r,y2,
               x1,y2, x1,y2-r, x1,y1+r, x1,y1]
        return cv.create_polygon(pts, smooth=True, **kw)

    # Card subtle shadow
    rounded_rect(canvas, CARD_X+2, CARD_Y+2, CARD_X+CARD_W+2, CARD_Y+CARD_H+2, R,
                 fill="#D3D1C7", outline="")
    # Card body
    rounded_rect(canvas, CARD_X, CARD_Y, CARD_X+CARD_W, CARD_Y+CARD_H, R,
                 fill=BG_CARD, outline=BORDER)

    # ── Top accent bar (animates blue → teal up to confidence%) ──────────────
    BAR_H  = 4
    BAR_Y  = CARD_Y
    BAR_X1 = CARD_X + R
    BAR_X2 = CARD_X + CARD_W - R
    canvas.create_rectangle(BAR_X1, BAR_Y, BAR_X2, BAR_Y + BAR_H,
                             fill=BLUE_BAR, outline="")
    bar_fill = canvas.create_rectangle(BAR_X1, BAR_Y, BAR_X1, BAR_Y + BAR_H,
                                        fill=TEAL_BAR, outline="")

    def animate_bar(step=0, total=40):
        frac = min(confidence / 100.0, 1.0) if confidence > 0 else 0.85
        target_x  = BAR_X1 + int((BAR_X2 - BAR_X1) * frac)
        current_x = BAR_X1 + int((BAR_X2 - BAR_X1) * frac * step / total)
        if step < total:
            canvas.coords(bar_fill, BAR_X1, BAR_Y, current_x, BAR_Y + BAR_H)
            root.after(16, lambda: animate_bar(step + 1, total))
        else:
            canvas.coords(bar_fill, BAR_X1, BAR_Y, target_x, BAR_Y + BAR_H)

    root.after(300, animate_bar)

    # ── Header ────────────────────────────────────────────────────────────────
    HX = CARD_X + PAD
    HY = CARD_Y + BAR_H + 14
    IW = 34

    # Icon box
    rounded_rect(canvas, HX, HY, HX+IW, HY+IW, 6,
                 fill="#E6F1FB", outline="#B5D4F4")
    canvas.create_text(HX + IW//2, HY + IW//2,
                       text="T", font=fn(15, "bold"), fill="#378ADD")

    # Title (sentence case) + subtitle
    canvas.create_text(HX+IW+10, HY+9,  anchor="w",
                       text="Text extracted", font=F_CARD, fill=TEXT_PRI)
    canvas.create_text(HX+IW+10, HY+23, anchor="w",
                       text=f"{ocr_method}  ·  region selected",
                       font=F_SUB, fill=TEXT_TER)

    # Close button
    CLX = CARD_X + CARD_W - PAD - 13
    CLY = HY + IW // 2
    close_oval = canvas.create_oval(CLX-13, CLY-13, CLX+13, CLY+13,
                                     fill="#F1EFE8", outline=BORDER)
    close_lbl  = canvas.create_text(CLX, CLY, text="✕",
                                     font=fn(9), fill=TEXT_TER)

    def on_close(e=None):
        root.destroy()

    for tag in (close_oval, close_lbl):
        canvas.tag_bind(tag, "<Button-1>", on_close)
        canvas.tag_bind(tag, "<Enter>",
                        lambda e: canvas.itemconfig(close_oval, fill="#D3D1C7"))
        canvas.tag_bind(tag, "<Leave>",
                        lambda e: canvas.itemconfig(close_oval, fill="#F1EFE8"))

    # ── Editable text area (taller: 130px) ───────────────────────────────────
    TBX = HX
    TBY = HY + IW + 14
    TBW = CARD_W - 2 * PAD
    TBH = 130

    rounded_rect(canvas, TBX, TBY, TBX+TBW, TBY+TBH, 8,
                 fill=BG_SURFACE, outline=BORDER)

    txt_frame = tk.Frame(root, bg=BG_SURFACE, bd=0, highlightthickness=0)
    txt_widget = tk.Text(
        txt_frame, wrap="word", font=F_BODY,
        bg=BG_SURFACE, fg=TEXT_PRI,
        relief="flat", bd=0, highlightthickness=0,
        insertbackground=TEXT_PRI,
        selectbackground="#B5D4F4", selectforeground="#042C53",
        spacing3=3
    )
    txt_widget.insert("1.0", extracted_text)
    txt_widget.pack(fill="both", expand=True, padx=10, pady=8)
    txt_frame.place(x=TBX+1, y=TBY+1, width=TBW-2, height=TBH-2)

    # ── Stat cells (4 cells: words / chars / confidence / engine) ────────────
    STY    = TBY + TBH + 10
    CELL_W = (CARD_W - 2*PAD - 24) // 4
    CELL_H = 52

    if confidence >= 80:
        conf_color = CONF_HIGH
    elif confidence >= 50:
        conf_color = CONF_MED
    else:
        conf_color = CONF_LOW

    conf_display   = f"{int(round(confidence))}%" if confidence > 0 else "N/A"
    engine_display = ocr_method  # full name, no truncation

    cells = [
        (str(word_count),  "words",      TEXT_PRI),
        (str(char_count),  "chars",      TEXT_PRI),
        (conf_display,     "confidence", conf_color),
        (engine_display,   "OCR engine", TEXT_PRI),
    ]

    for i, (val, lbl, color) in enumerate(cells):
        cx = HX + i * (CELL_W + 8)
        rounded_rect(canvas, cx, STY, cx+CELL_W, STY+CELL_H, 6,
                     fill=BG_SURFACE, outline="")
        canvas.create_text(cx + CELL_W//2, STY+16, text=val,
                           font=F_STAT_VAL, fill=color)
        canvas.create_text(cx + CELL_W//2, STY+36, text=lbl,
                           font=F_STAT_LBL, fill=TEXT_TER)

    # ── Toast notifications ───────────────────────────────────────────────────
    toast_items = [None, None]

    def show_toast(msg):
        for item in toast_items:
            if item:
                canvas.delete(item)
        tx, ty = W // 2, H - 18
        toast_items[0] = canvas.create_oval(tx-90, ty-12, tx+90, ty+12,
                                             fill=TOAST_BG, outline="#0F6E56")
        toast_items[1] = canvas.create_text(tx, ty, text=msg,
                                             font=fn(9), fill=TOAST_FG)
        root.after(2000, lambda: [canvas.delete(t) for t in toast_items if t])

    # ── Button factory ────────────────────────────────────────────────────────
    def make_btn(x, y, w, h, label, cmd, primary=False):
        bg_n = BG_COPY   if primary else BG_SURFACE
        fg_n = FG_COPY   if primary else TEXT_SEC
        bg_h = "#0C447C" if primary else "#D3D1C7"
        bdr  = "#185FA5" if primary else BORDER

        box = rounded_rect(canvas, x, y, x+w, y+h, 8, fill=bg_n, outline=bdr)
        lbl = canvas.create_text(x+w//2, y+h//2, text=label,
                                  font=(F_COPY if primary else F_BTN), fill=fg_n)

        def on_enter(e):  canvas.itemconfig(box, fill=bg_h)
        def on_leave(e):  canvas.itemconfig(box, fill=bg_n)
        def on_click(e):
            canvas.itemconfig(box, fill=bdr)
            root.after(100, lambda: canvas.itemconfig(box, fill=bg_n))
            cmd()

        for tag in (box, lbl):
            canvas.tag_bind(tag, "<Enter>",    on_enter)
            canvas.tag_bind(tag, "<Leave>",    on_leave)
            canvas.tag_bind(tag, "<Button-1>", on_click)

    # ── Button actions ────────────────────────────────────────────────────────
    def do_copy():
        pyperclip.copy(txt_widget.get("1.0", "end-1c"))
        show_toast("Copied to clipboard")

    def do_search():
        try:
            search_text_on_google(txt_widget.get("1.0", "end-1c"))
            show_toast("Opened Google search")
        except Exception as e:
            show_toast(f"Error: {e}")

    def do_save():
        fname = save_text_to_file(txt_widget.get("1.0", "end-1c"))
        show_toast("Saved to temp folder" if fname else "Save failed")

    # ── Button layout ─────────────────────────────────────────────────────────
    BTY   = STY + CELL_H + 12
    BTN_H = 34
    BTN_W = (CARD_W - 2*PAD - 8) // 2

    make_btn(HX,             BTY, BTN_W, BTN_H, "Search Google", do_search)
    make_btn(HX + BTN_W + 8, BTY, BTN_W, BTN_H, "Save to file",  do_save)
    make_btn(HX, BTY + BTN_H + 8, CARD_W - 2*PAD, BTN_H + 4,
             "Copy to clipboard", do_copy, primary=True)

    root.mainloop()


# ─────────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_text_to_file(text):
    try:
        filename = os.path.join(os.getenv("TEMP"), f"ocr_text_{int(time.time())}.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"[+] Text saved to: {filename}")
        return filename
    except Exception as e:
        print(f"[-] Error saving text: {e}")
        return None


def search_text_on_google(text, driver=None):
    try:
        if driver is None:
            options = Options()
            options.add_experimental_option("detach", True)
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), options=options)
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[-1])
        driver.get("https://www.google.com")
        time.sleep(2)
        search_box = driver.find_element(By.NAME, "q")
        search_box.send_keys(text[:500])
        search_box.send_keys(Keys.RETURN)
        print("[+] Text search opened in browser")
        return driver
    except Exception as e:
        print(f"[-] Error searching text: {e}")
        return driver


# ─────────────────────────────────────────────────────────────────────────────
#  Screen selection helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_rounded_rect(img, pt1, pt2, radius, color, thickness):
    x1, y1 = pt1
    x2, y2 = pt2
    if thickness < 0:
        cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, -1)
        cv2.circle(img, (x1+radius, y1+radius), radius, color, -1)
        cv2.circle(img, (x2-radius, y1+radius), radius, color, -1)
        cv2.circle(img, (x1+radius, y2-radius), radius, color, -1)
        cv2.circle(img, (x2-radius, y2-radius), radius, color, -1)
    else:
        cv2.line(img, (x1+radius, y1), (x2-radius, y1), color, thickness)
        cv2.line(img, (x1+radius, y2), (x2-radius, y2), color, thickness)
        cv2.line(img, (x1, y1+radius), (x1, y2-radius), color, thickness)
        cv2.line(img, (x2, y1+radius), (x2, y2-radius), color, thickness)
        cv2.ellipse(img, (x1+radius, y1+radius), (radius,radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y1+radius), (radius,radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y2-radius), (radius,radius),   0, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+radius, y2-radius), (radius,radius),  90, 0, 90, color, thickness)


def circle_search():
    global start_point, end_point, cropping, cropped_img, screenshot

    start_point = None
    end_point   = None
    cropping    = False
    cropped_img = None

    sys.stdout.write("📸 Taking screenshot...\n")
    sys.stdout.write("🖱️  Click and drag to select text region\n")
    sys.stdout.write("⌨️  Press 'q' to cancel\n")
    sys.stdout.flush()

    screenshot_pil = ImageGrab.grab()
    screenshot = np.array(screenshot_pil)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

    overlay = screenshot.copy()
    overlay[:] = (0, 0, 0)
    dimmed_screenshot = cv2.addWeighted(overlay, 0.55, screenshot, 0.45, 0)

    cv2.namedWindow("Screenshot", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Screenshot", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty("Screenshot", cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback("Screenshot", mouse_crop)
    cv2.imshow("Screenshot", screenshot)
    cv2.waitKey(1)

    while True:
        temp_img = dimmed_screenshot.copy()
        if start_point and end_point:
            x1, y1 = start_point
            x2, y2 = end_point
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            radius = 22
            mask = np.zeros(temp_img.shape[:2], dtype=np.uint8)
            draw_rounded_rect(mask, (x_min, y_min), (x_max, y_max), radius, 255, -1)
            temp_img[mask == 255] = screenshot[mask == 255]
            draw_rounded_rect(temp_img, (x_min, y_min), (x_max, y_max),
                              radius, (235, 235, 235), 3)
            draw_rounded_rect(temp_img, (x_min+2, y_min+2), (x_max-2, y_max-2),
                              radius-2, (200, 200, 200), 1)

        cv2.putText(temp_img, "Drag to select  \u2022  Press Q to cancel",
                    (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (180, 180, 180), 1, cv2.LINE_AA)
        cv2.imshow("Screenshot", temp_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
        if cropped_img is not None:
            break

    if cropped_img is not None:
        filename = os.path.join(os.getenv("TEMP"), "circle_to_search_temp.png")
        cv2.imwrite(filename, cropped_img)
        print(f"[+] Saved cropped image to: {filename}")

        sys.stdout.write("🔍 Performing OCR on selected region...\n")
        sys.stdout.flush()

        extracted_text, ocr_method, confidence = perform_ocr_combined(cropped_img)

        if extracted_text and extracted_text.strip():
            word_count = len(extracted_text.split())
            sys.stdout.write(f"[+] Extracted {word_count} words via {ocr_method} "
                             f"(confidence: {confidence:.1f}%)\n")
            sys.stdout.write("[+] Opening result window...\n")
            sys.stdout.flush()
            show_text_in_gui(extracted_text, ocr_method=ocr_method, confidence=confidence)
        else:
            root = tk.Tk()
            root.withdraw()
            tk.messagebox.showwarning(
                "No Text Found",
                "No text was detected in the selected region.\n"
                "Try selecting a region with clearer text."
            )
            root.destroy()
    else:
        print("[-] No region selected.")


def on_hotkey():
    sys.stdout.write("\n" + "="*50 + "\n")
    sys.stdout.write("🎯 Text Extraction Activated!\n")
    sys.stdout.write("="*50 + "\n")
    sys.stdout.flush()
    threading.Thread(target=circle_search, daemon=True).start()


def main():
    sys.stdout.write("🔍 Text Extraction is now running!\n")
    sys.stdout.write("=" * 50 + "\n")
    sys.stdout.write("HOW TO USE:\n")
    sys.stdout.write("  Press Ctrl + Shift + R anywhere to activate\n")
    sys.stdout.write("=" * 50 + "\n")
    sys.stdout.write("FEATURES:\n")
    sys.stdout.write("  - OCR text extraction\n")
    sys.stdout.write("  - Copy text to clipboard\n")
    sys.stdout.write("  - Google text search\n")
    sys.stdout.write("  - Save text to file\n")
    sys.stdout.write("=" * 50 + "\n")
    sys.stdout.write("Press Ctrl+C in terminal to exit\n")
    sys.stdout.write("Waiting for hotkey...\n")
    sys.stdout.write("=" * 50 + "\n")
    sys.stdout.flush()

    hotkey_listener = keyboard.GlobalHotKeys({
        '<ctrl>+<shift>+r': on_hotkey
    })
    hotkey_listener.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[+] Text extraction stopped.")
        hotkey_listener.stop()


if __name__ == "__main__":
    main()