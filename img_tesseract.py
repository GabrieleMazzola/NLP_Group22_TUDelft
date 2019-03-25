from PIL import Image
import pytesseract
import glob, os

PATH_TO_TESSERACT = r'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = PATH_TO_TESSERACT


def extract_text(image_path, lang='eng'):
    extracted_text = pytesseract.image_to_string(Image.open(image_path), lang=lang)
    return extracted_text if extracted_text else None


def find_images_in(dir_path):
    files = [file for file in os.listdir(dir_path) if file.endswith("png") or file.endswith("jpg")]
    images = []
    for file in files:
        images.append(os.path.join(dir_path, file))
    return images