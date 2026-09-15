# SmartText-Snippet
AI-powered desktop OCR tool for capturing text from screen regions, processing extracted text, and generating concise summaries.
## Features

- Screen-region text extraction
- Hybrid OCR using Tesseract and EasyOCR
- OpenCV-based image preprocessing
- NLP-based text post-processing
- Spell correction
- Copy extracted text to clipboard
- Automated Google search
- Text file export
- Global hotkey activation

## Technologies Used

- Python
- OpenCV
- Tesseract OCR
- EasyOCR
- spaCy
- NLTK
- Tkinter
- Selenium
- Pyperclip
- Pynput

## How It Works

1. Press `Ctrl + Shift + R`
2. Select a region of the screen
3. The selected image is preprocessed using OpenCV
4. OCR engines extract the text
5. The better OCR result is selected
6. NLP techniques clean the extracted text
7. The result is displayed in an editable GUI
8. The user can copy, save, or search the extracted text

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
