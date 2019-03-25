from img_tesseract import extract_text, find_images_in
import numpy as np

images = find_images_in("./train_set/media")

no_text = 0
thres = 5
imgs_below_thres = []
text_lens = []
for img in images:
    text = extract_text(img)
    if text:
        text_lens.append(len(text))
        if len(text) < thres:
            imgs_below_thres.append((img, text))
    else:
        no_text += 1
    #print("\n------\n" + str(text) + "\t"+ img + "\t" + str(len(str(text))) + "\n------\n")

print(len(imgs_below_thres))
print(imgs_below_thres)

print(np.mean(text_lens), np.std(text_lens))
print(no_text)
print(len(images))
