import pickle

import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif

from gabry_dataset_parser import get_labeled_instances
from n_grams.ana_ngrams import get_all_ngrams_for_post

ngrams_clickbait = {}
ngrams_no_clickbait = {}

filenames = [(1,'unigrams'), (2, 'bigrams'), (3, 'trigrams'), (4, '4grams')]

for key, name in filenames:
    with open("./{}_clickbait_big.pickle".format(name), "rb") as f:
        ngrams_clickbait[key] = pickle.load(f)
    with open("./{}_no_clickbait_big.pickle".format(name), "rb") as f:
        ngrams_no_clickbait[key] = pickle.load(f)

all_ngrams = {}

for k in ngrams_clickbait.keys():
    for ngram in ngrams_clickbait[k]:
        all_ngrams[ngram] = ngrams_clickbait[k][ngram]

for k in ngrams_no_clickbait.keys():
    for ngram in ngrams_no_clickbait[k]:
        all_ngrams[ngram] = all_ngrams.get(ngram, 0) + ngrams_no_clickbait[k][ngram]

THRESHOLD = 5

filtered_ngrams = {}
for ngram in all_ngrams:
    if all_ngrams[ngram] >= THRESHOLD:
        filtered_ngrams[ngram] = all_ngrams[ngram]


labeled_instances = get_labeled_instances("../train_set/instances_converted_big.pickle",
                                          "../train_set/truth_converted_big.pickle")[['truthClass', 'postText', 'id']]

postTexts = list(labeled_instances.postText)
ids = list(labeled_instances.id)
dict_list = []
for idx, post_text in enumerate(postTexts):
    print(idx)
    post_text = post_text[0]
    post_dict = {x: 0 for x in filtered_ngrams.keys()}
    post_dict['id'] = ids[idx]
    ngrams = get_all_ngrams_for_post(post_text)
    for ngram in ngrams:
        if ngram in post_dict:
            post_dict[ngram] += 1

    dict_list.append(post_dict)

print("Loop done, building the dataframe...")
feat_df = pd.DataFrame(dict_list)
print(feat_df.shape)
print("DONE. Saving...")
feat_df.to_csv("./ngrams_features_counts_before_infoGain.csv", index=False)

print("Loading csv...")
feat_df = pd.read_csv("./ngrams_features_counts_before_infoGain.csv")
feat_df['id'] = labeled_instances['id']

le = preprocessing.LabelEncoder()
label_encoded = le.fit_transform(labeled_instances['truthClass'])
label_encoded = [1 if lab == 0 else 0 for lab in list(label_encoded)]
print(f"Labels encoded. Class '{labeled_instances['truthClass'][0]}' --> label '{label_encoded[0]}'")
label_encoded = pd.DataFrame(label_encoded, columns=['label'])

info_gains = mutual_info_classif(feat_df, label_encoded.values.ravel(), discrete_features=True)

print(info_gains)


features_gains = list(zip(list(feat_df.columns), list(info_gains)))
features_gains = sorted(features_gains, key=lambda x: x[1], reverse=True)
print(f"{len(features_gains)} total features.")
non_zero_features = [feat for feat in features_gains if feat[1] > 0]
print(f"{len(non_zero_features)} features are different from zero.")

TOP_PERC = 0.01
selected_features = non_zero_features[:int(TOP_PERC*len(non_zero_features))]
selected_features = [feat[0] for feat in selected_features]
print(f"{len(selected_features)} selected features.")

feat_df = feat_df[selected_features]

feat_df.to_csv("./ngrams_features_counts_after_infoGain{}.csv".format(str(TOP_PERC*100)), index=False)
print(feat_df.shape)
print([feat for feat in features_gains])
