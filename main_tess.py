import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import numpy as np
from img_tesseract import extract_text, find_images_in


# no_text = 0
# thres = 5
# imgs_below_thres = []
# text_lens = []
# for img in images:
#     text = extract_text(img)
#     if text:
#         text_lens.append(len(text))
#         if len(text) < thres:
#             imgs_below_thres.append((img, text))
#     else:
#         no_text += 1
#     #print("\n------\n" + str(text) + "\t"+ img + "\t" + str(len(str(text))) + "\n------\n")
#
# print(len(imgs_below_thres))
# print(imgs_below_thres)
#
# print(np.mean(text_lens), np.std(text_lens))
# print(no_text)
# print(len(images))


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


DICT_PATH = './extracted_texts.pickle'
# extract_and_save_all_texts("./train_set/media", DICT_PATH)
texts = load_extracted_texts_from(DICT_PATH)

new = []
for path in texts.keys():
    text = texts[path]
    # remove all tokens that are not alphabetic
    text = "".join([c for c in text if c.isalpha()])
    new.append((text, path))



thres = 10
total = len(new)
no_image = len([text for text, path in new if len(text) == 0])
in_thres = len([text for text, path in new if 0 < len(text) < thres])
print("Total: " + str(total))
print("No image: " + str(no_image))
print("In threshold: " + str(in_thres))
print("Remaining: " + str(total - no_image - in_thres))
for text, path in [(text, path) for text, path in new if 0 < len(text) < thres]:
    print(text + "\t" + path)

x = [len(text) for text, path in new if len(text) > thres]
sns.set_style('whitegrid')
sns.distplot(np.array(x))
plt.show()