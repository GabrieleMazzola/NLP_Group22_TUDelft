from PIL import Image
import pytesseract
import glob, os

PATH_TO_TESSERACT = r'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = PATH_TO_TESSERACT


def extract_text(image_path, lang='eng'):
    extracted_text = pytesseract.image_to_string(Image.open(image_path), lang=lang)
    return extracted_text if extracted_text else None


def find_images_in(dir_path, img_type='jpg'):
    os.chdir(dir_path)
    images = []
    for file in glob.glob(f"*.{img_type}"):
        images.append(file)
    return images