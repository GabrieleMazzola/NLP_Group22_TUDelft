import pickle
import matplotlib.pyplot as plt
import numpy as np
DATA_PATH = r"../features/targetArticle_NER-lowercase.pickle"

with open(DATA_PATH, 'rb') as myfile:
    data = pickle.load(myfile)

tags = [elem[0] for elem in data]
bait = [elem[1] for elem in data]
no_bait = [elem[2] for elem in data]

bait_normalizer = sum(bait)
no_bait_normalizer = sum(no_bait)

bait = [100 * elem / bait_normalizer for elem in bait]
no_bait = [100 * elem / no_bait_normalizer for elem in no_bait]

print(tags)
print(bait)
print(no_bait)

ind = np.arange(len(data))
width = 0.3

plt.figure(1)
plt.title("targetArcticle - NER lowercase - no punct removal")
plt.bar(ind + width, bait, width=width, color='r', align='center', label='clickbait')
plt.bar(ind + 2 * width, no_bait, width=width, color='b', align='center', label='no-clickbait')
plt.xticks((ind + ind + 2 * width) / 2, tuple(tags), fontsize=7, rotation=45)
plt.ylabel("POS %")
plt.legend()
plt.show()

