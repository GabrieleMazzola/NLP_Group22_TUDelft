import pickle
from nltk.util import ngrams
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


# normalize the counts
# ---> count of each ngram / total occurrences of ngram

# select top 0.0005 of each (uni, bi, tri, 4), for C and NC separately

# loop over the dataset again, separately for C and NC rows
# for C data -> count how many posts contain top C list ngrams
# for NC data -> count how many posts contain top NC list ngrams

clickbait_final_list = []
for n, ngrams in ngrams_clickbait.items():
    normalizer = sum(ngrams.values())
    for ngram in ngrams.keys():
        ngrams[ngram] /= normalizer

    ngrams_clickbait[n] = ngrams.most_common(int(len(ngrams.keys())*0.005))
    clickbait_final_list += [elem[0] for elem in ngrams_clickbait[n]]

clickbait_final_list = set(clickbait_final_list)

no_clickbait_final_list = []
for n, ngrams in ngrams_no_clickbait.items():
    normalizer = sum(ngrams.values())
    for ngram in ngrams.keys():
        ngrams[ngram] /= normalizer

    ngrams_no_clickbait[n] = ngrams.most_common(int(len(ngrams.keys())*0.005))
    no_clickbait_final_list += [elem[0] for elem in ngrams_no_clickbait[n]]

no_clickbait_final_list = set(no_clickbait_final_list)


labeled_instances = get_labeled_instances("../train_set/instances_converted_big.pickle",
                                          "../train_set/truth_converted_big.pickle")


clickbait_df = labeled_instances[labeled_instances.truthClass == 'clickbait']
no_clickbait_df = labeled_instances[labeled_instances.truthClass == 'no-clickbait']

clickbait_ids = list(clickbait_df.id)
clickbait_texts = [txt[0] for txt in list(clickbait_df.postText)]

no_clickbait_ids = list(clickbait_df.id)
no_clickbait_texts = [txt[0] for txt in list(no_clickbait_df.postText)]

count_clickbait = 0
for idx, txt in enumerate(clickbait_texts, 1):
    ngrams = get_all_ngrams_for_post(txt)
    in_the_list = len(ngrams) - len(ngrams-clickbait_final_list)
    if in_the_list:
        count_clickbait += 1

print(count_clickbait / len(clickbait_texts))

count_no_clickbait = 0
for idx, txt in enumerate(no_clickbait_texts, 1):
    ngrams = get_all_ngrams_for_post(txt)
    in_the_list = len(ngrams) - len(ngrams-no_clickbait_final_list)
    if in_the_list:
        count_no_clickbait += 1

print(count_no_clickbait / len(no_clickbait_texts))


print("")



