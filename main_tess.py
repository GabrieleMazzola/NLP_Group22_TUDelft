import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from img_tesseract import extract_text, find_images_in

DICT_PATH = './extracted_texts.pickle'
IMAGE_TEXT_CHAR_THRES = 10


def extract_and_save_all_texts(image_path, filepath):
    image_paths = find_images_in(image_path)
    extracted_texts = {}
    for img in image_paths:
        text = extract_text(img)
        extracted_texts[img] = text if text else ""
    with open(filepath, 'wb') as f:
        pickle.dump(extracted_texts, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("done")


def load_extracted_texts_from(dict_path):
    with open(dict_path, 'rb') as f:
        b = pickle.load(f)
    return b


def text_in_image(image_path):
    text = extract_text(image_path)
    if not text:
        return 0
    text = "".join([c for c in text if c.isalpha()])
    return 1 if len(text) > IMAGE_TEXT_CHAR_THRES else 0




if __name__ == "__main__":
    extract_and_save_all_texts("./train_set/media", DICT_PATH)
    # texts = load_extracted_texts_from(DICT_PATH)
    # new = []
    # for path in texts.keys():
    #     text = texts[path]
    #     # remove all tokens that are not alphabetic
    #     text = "".join([c for c in text if c.isalpha()])
    #     new.append((text, path))
    #
    #
    #
    # total = len(new)
    # no_image = len([text for text, path in new if len(text) == 0])
    # in_thres = len([text for text, path in new if 0 < len(text) < IMAGE_TEXT_CHAR_THRES])
    # print("Total: " + str(total))
    # print("No image: " + str(no_image))
    # print("In threshold: " + str(in_thres))
    # print("Remaining: " + str(total - no_image - in_thres))
    # for text, path in [(text, path) for text, path in new if 0 < len(text) < IMAGE_TEXT_CHAR_THRES]:
    #     print(text + "\t" + path)
    #
    # x = [len(text) for text, path in new if len(text) > IMAGE_TEXT_CHAR_THRES]
    # sns.set_style('whitegrid')
    # sns.distplot(np.array(x))
    # plt.show()