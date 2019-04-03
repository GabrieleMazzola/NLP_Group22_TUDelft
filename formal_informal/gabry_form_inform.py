import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import words
from nltk.tokenize import RegexpTokenizer
import numpy as np

from gabry_dataset_parser import get_labeled_instances
from util import number_replacement


def extract_formal_informal_features(sentence, wordset, normalize):
    original_sentence = sentence
    sentence = sentence.replace("RT", "")
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    sentence = sentence.lower()
    sentence = number_replacement(sentence, "")
    tokens = tokenizer.tokenize(sentence)
    lemmas = [lemmatizer.lemmatize(lemmatizer.lemmatize(token), pos='v') for token in tokens]

    if not lemmas:
        print("Sentence is empty!")
        return [0]

    formal = 0
    for lemma in lemmas:
        if lemma in wordset:
            formal += 1

    if normalize:
        res = round(100 * formal / len(lemmas), 2)
        return [res]
    else:
        return [formal]


if __name__ == '__main__':

    english_words = set(words.words())

    DATASET = 'big'  # 'small' or 'big'
    target = "postText"  # "postText" or "targetTitle"
    prefix = "PT" if target == "postText" else "TA"
    NORMALIZE = True

    FEATURES_DATA_PATH = r"../features/{}/formal_informal_features_{}_{}_{}.csv".format(DATASET, DATASET, target,
                                                                                        'normalized' if NORMALIZE else "no-normalized")

    print(f"Generating features... it might take a while :P\n Path: '{FEATURES_DATA_PATH}' | {target} | {prefix}")

    labeled_instances = get_labeled_instances("../train_set/instances_converted_{}.pickle".format(DATASET),
                                              "../train_set/truth_converted_{}.pickle".format(DATASET))

    ids = list(labeled_instances.id)

    if target == 'postText':
        texts = [txt[0] for txt in list(labeled_instances.postText)]
    else:
        texts = [txt for txt in list(labeled_instances.targetTitle)]
    features = []

    for idx, txt in enumerate(texts, 1):
        print(f"Computing features for sample {idx} out of {len(texts)}...")
        features.append(extract_formal_informal_features(txt, english_words, NORMALIZE))

    data_to_df = [tuple([ids[i]] + features[i]) for i in range(len(ids))]
    labels = ['id', prefix + "_formal"]

    df = pd.DataFrame.from_records(data_to_df, columns=labels)

    df.to_csv(FEATURES_DATA_PATH, index=False)

    print("Generation of features completed, phuff!")

    temp = df.merge(labeled_instances, on='id')[[prefix + '_formal', 'truthClass']]
    cl = temp[temp['truthClass'] == 'clickbait'][prefix + '_formal']
    no_cl = temp[temp['truthClass'] == 'no-clickbait'][prefix + '_formal']
    print(np.mean(cl), np.mean(no_cl))
