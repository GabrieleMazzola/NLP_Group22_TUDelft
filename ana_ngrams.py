import nltk
import collections
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from itertools import chain
import string
from gabry_dataset_parser import get_labeled_instances
import util
import pandas as pd


def lemmatize(lemmatizer, tokenized_text):
    return [lemmatizer.lemmatize(word) for word in tokenized_text]


def stem_text(stemmer, tokenized_text):
    return [stemmer.stem(word) for word in tokenized_text]


def get_ngrams(tokenized_text, n):
    # returns a generator
    return nltk.ngrams(tokenized_text, n)


def count_all_ngrams(ngrams):
    return collections.Counter(ngrams)


def replace(token):
    if token == "123quote123":
        token = "<quote>"
    if token == "123hashtag123":
        token = "<hashtag>"
    token = util.number_replacement(token, "<d>")
    return token


def preprocess_post_text(raw_text):
    punctuation = string.punctuation.replace("@", "")
    punctuation = punctuation + "’" + '“' + '”' + "-" + '‘' + '–' + '—'

    regex_retweet = re.compile(r'RT @[\w]+[:]?')
    regex_via = re.compile(r'via @[\w]+')
    regex_mentions = re.compile(r'@[\w]+')

    post_text = util.quote_replacement(raw_text, "123quote123")
    post_text = util.hashtag_replacement(post_text, "123hashtag123")

    post_text = re.sub(regex_retweet, "", post_text)
    post_text = re.sub(regex_mentions, "", post_text)
    post_text = re.sub(regex_via, "", post_text)
    post_text = post_text.translate(str.maketrans('', '', punctuation)).lower()

    tokenized_post = nltk.word_tokenize(post_text)
    return tokenized_post


def get_all_ngrams(df, n):
    stop_words = set(stopwords.words('english'))

    ngrams = {}
    row_iterator = df.iterrows()
    for row in row_iterator:
        post_text = row[1]['postText'][0]
        tokenized_post = preprocess_post_text(post_text)

        if n == 1:
            tokenized_post = [replace(w) for w in tokenized_post if w not in stop_words]
        else:
            tokenized_post = [replace(w) for w in tokenized_post]

        if not ngrams:
            ngrams = get_ngrams(tokenized_post, n)
        else:
            ngrams = chain(ngrams, get_ngrams(tokenized_post, n))

    return ngrams


if __name__ == '__main__':
    # load frames and get the right columns
    labeled_instances = get_labeled_instances("./train_set/instances_converted_small.pickle", "./train_set/truth_converted_small.pickle")
    clickbait_df = labeled_instances[labeled_instances.truthClass == 'clickbait']
    no_clickbait_df = labeled_instances[labeled_instances.truthClass == 'no-clickbait']
    clickbait_df_cols = clickbait_df[['id', 'postText', 'targetTitle']]
    no_clickbait_df_cols = no_clickbait_df[['id', 'postText', 'targetTitle']]

    all_ngrams = {1: {}, 2: {}, 3: {}, 4: {}}
    only_clickbait = {}
    only_non_clickbait = {}
    all_ngrams_clickbait = {}
    all_ngrams_no_clickbait = {}
    intersection_ngrams = {}
    # intersection_counts = {}

    for n in all_ngrams.keys():
        all_ngrams_clickbait[n] = count_all_ngrams(get_all_ngrams(clickbait_df_cols, n))
        all_ngrams_no_clickbait[n] = count_all_ngrams(get_all_ngrams(no_clickbait_df_cols, n))
        all_ngrams[n] = all_ngrams_clickbait[n] + all_ngrams_no_clickbait[n]

        only_clickbait[n] = collections.Counter({k: v for (k, v) in all_ngrams_clickbait[n].items()
                                                 if k not in all_ngrams_no_clickbait[n]})

        only_non_clickbait[n] = collections.Counter({k: v for (k, v) in all_ngrams_no_clickbait[n].items()
                                                     if k not in all_ngrams_clickbait[n]})

        intersection_ngrams[n] = collections.Counter({k: v for (k, v) in all_ngrams[n].items()
                                                      if k in all_ngrams_clickbait[n] and k in all_ngrams_no_clickbait[n]})

        # intersection_counts[n] = [(k, all_ngrams_clickbait[n][k] - all_ngrams_no_clickbait[n][k]) for k in intersection_ngrams[n].keys()]
        # intersection_counts[n] = sorted(intersection_counts[n], key=lambda item: item[1], reverse=True)

    all_ngrams_flattened = collections.Counter()
    for k in all_ngrams.keys():
        all_ngrams_flattened = all_ngrams_flattened + all_ngrams[k]
    all_ngrams_flattened = all_ngrams_flattened.most_common()

    # total_ngram_counts = {ngram: {'clickbait': {}, 'non-clickbait': {}} for ngram, c in all_ngrams_joined}
    stop_words = set(stopwords.words('english'))

    first_n_most_common = {ngram: count for (ngram, count) in all_ngrams_flattened[:int(len(all_ngrams_flattened) * 0.005)]}
    posts_with_n_most_common = {ngram: {'clickbait': [], 'non-clickbait': []} for ngram in first_n_most_common.keys()}

    no_clickbait_messages_containing_ngrams = set()
    clickbait_messages_containing_ngrams = set()

    # OK COLLECT IDS, TEXTS, COMMON-NGRAMS CONTAINED
    COLLECT = {
        "COLLECT_IDS": [],
        "COLLECT_TEXTS": [],
        "COLLECT_NGRAMS": [],
        "COLLECT_IS_CLICKBAIT": []
    }


    # CLICKBAIT NGRAMS
    clickbait_row_iterator = clickbait_df_cols.iterrows()

    for i, row in enumerate(clickbait_row_iterator):
        post_id = row[1]['id']
        post_text = row[1]['postText'][0]

        COLLECT["COLLECT_IDS"].append(post_id)
        COLLECT["COLLECT_TEXTS"].append(post_text)
        COLLECT["COLLECT_IS_CLICKBAIT"].append('clickbait')
        COLLECT["COLLECT_NGRAMS"].append([])

        for k in all_ngrams.keys():
            # EXTRACT NGRAMS
            tokenized_post = preprocess_post_text(post_text)
            if k == 1:
                tokenized_post = [replace(w) for w in tokenized_post if w not in stop_words]
            else:
                tokenized_post = [replace(w) for w in tokenized_post]

            row_ngrams = get_ngrams(tokenized_post, k)

            # CHECK IF IN TOP N NGRAMS
            for ngram in row_ngrams:
                if ngram in first_n_most_common:
                    COLLECT["COLLECT_NGRAMS"][i].append(ngram)
                    clickbait_messages_containing_ngrams.add(post_id)
                    posts_with_n_most_common[ngram]['clickbait'].append(post_id)

    # clickbait_df[clickbait_df['id'].isin(posts_with_n_most_common[('supreme','court')]['clickbait'])]
    # NON-CLICKBAIT NGRAMS

    no_clickbait_row_iterator = no_clickbait_df_cols.iterrows()

    # loop over non-clickbait rows
    # get n-grams
    # check against the top ngrams
    # if top ngrams contained in post n-grams, add message id to list

    previous_posts = len(COLLECT["COLLECT_IDS"])
    for i, row in enumerate(no_clickbait_row_iterator):
        post_id = row[1]['id']
        post_text = row[1]['postText'][0]
        COLLECT["COLLECT_IDS"].append(post_id)
        COLLECT["COLLECT_TEXTS"].append(post_text)
        COLLECT["COLLECT_IS_CLICKBAIT"].append('no-clickbait')
        COLLECT["COLLECT_NGRAMS"].append([])

        # EXTRACT NGRAMS
        for k in all_ngrams.keys():
            tokenized_post = preprocess_post_text(post_text)
            if k == 1:
                tokenized_post = [replace(w) for w in tokenized_post if w not in stop_words]
            else:
                tokenized_post = [replace(w) for w in tokenized_post]

            row_ngrams = get_ngrams(tokenized_post, k)

            # CHECK IF IN TOP N NGRAMS
            for ngram in row_ngrams:
                if ngram in first_n_most_common:
                    COLLECT["COLLECT_NGRAMS"][i+previous_posts].append(ngram)
                    no_clickbait_messages_containing_ngrams.add(post_id)
                    posts_with_n_most_common[ngram]['non-clickbait'].append(post_id)

    # ngram_comparing_counts = [(ngram, len(total_ngram_counts[ngram]['clickbait']) - len(total_ngram_counts[ngram]['non-clickbait'])) for (ngram, count) in all_ngrams_joined]

    print(len(clickbait_messages_containing_ngrams) / len(clickbait_df_cols), len(no_clickbait_messages_containing_ngrams) / len(no_clickbait_df_cols))

    df_features = pd.DataFrame.from_dict(COLLECT)
    df_features.to_csv('bullshit_ngram.csv', index=False)
    print(f"{labeled_instances.shape[0]} instances in total. {clickbait_df.shape[0]} clickbait, {no_clickbait_df.shape[0]} no-clickbait")
