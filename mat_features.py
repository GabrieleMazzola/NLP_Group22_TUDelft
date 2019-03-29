import string
import nltk
import pandas as pd
import itertools
from nltk.corpus import stopwords
from nltk.corpus import wordnet_ic
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

import numpy as np
import matplotlib.pyplot as plt

stop_words = set(stopwords.words('english'))
stop_words.add('\'s')
stop_words.add('RT')
stop_words.add('\'re')
st = StanfordNERTagger(
    '/home/esilezz/Scrivania/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz',
    '/home/esilezz/Scrivania/stanford-ner-2018-10-16/stanford-ner.jar',
    encoding='utf-8')


def count_determiners(sentence):
    text = word_tokenize(sentence)
    text = [w.lower() for w in text]
    tags = nltk.pos_tag(text)
    count = 0
    for tup in tags:
        if tup[1] == 'DT':
            count = count + 1
    return count


def ner_stanford(sentence):
    tokenized_text = word_tokenize(sentence)
    classified_text = st.tag(tokenized_text)
    filtered = []
    for tup in classified_text:
        if tup[1] != 'PERSON' and tup[1] != 'ORGANIZATION' and tup[1] != 'LOCATION' and tup[0] not in general_phrases:
            filtered.append(tup[0])
    return filtered


semcor_ic = wordnet_ic.ic('ic-semcor.dat')

general_phrases = ['RT', "'s", "'re", "u"]


def list_to_string(filtered_sentence):
    sentence = ''
    for word in filtered_sentence:
        sentence = sentence + ' ' + word
    return sentence


def lemmatize_string(filtered_sentence):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in filtered_sentence]


def clean_string(sentence):
    without_punct = sentence.translate(str.maketrans('', '', string.punctuation))
    word_tokens = word_tokenize(without_punct)
    filtered_sentence = [w.lower() for w in word_tokens if not w.lower() in stop_words]
    # filtered_sentence = []
    # for w in word_tokens:
    #     if w not in stop_words:
    #         filtered_sentence.append(w)
    return filtered_sentence


def final_check(list_of_words):
    for word in list_of_words:
        if not (word.isalpha()) or len(word) < 2:
            list_of_words.remove(word)
    return list_of_words


def avg_similarity(sentence):
    filtered_sentence = ner_stanford(sentence)
    filtered_sentence = clean_string(list_to_string(filtered_sentence))
    filtered_sentence = lemmatize_string(filtered_sentence)
    final_sentence = final_check(filtered_sentence)
    print(final_sentence)
    combi = set(itertools.combinations(final_sentence, 2))
    sum = 0
    for tup in combi:
        word1 = wn.synsets(tup[0])
        if len(word1) > 0:
            word1 = word1[0]
        word2 = wn.synsets(tup[1])
        if len(word2) > 0:
            word2 = word2[0]
        if (word1 and word2) and word1._pos == word2._pos and word1._pos != 's' and word2._pos != 's' and \
                word1._pos != 'a' and word2._pos != 'a' and word1._pos != 'r' and word2._pos != 'r':
            sum = sum + word1.lin_similarity(word2, semcor_ic)
    if len(combi) != 0:
        avg = sum / len(combi)
    else:
        avg = 0
    return avg


def avg_word_length(sentence):
    filtered_sentence = clean_string(sentence)
    sum = 0
    for word in filtered_sentence:
        sum = sum + len(word)
    avg = sum / len(filtered_sentence)
    return avg


def char_based_features(row, features_dict):
    lenPostTitle = len(row['postText'][0])
    lenArtTitle = len(row['targetTitle'])
    lenArtDesc = len(row['targetDescription'])
    lenArtKeywords = len(row['targetKeywords'])
    # print(len(row['postText'][0].split()))
    features_dict['numCharPostTitle'] = lenPostTitle
    # MISSING THE NUMBER OF CHARACTERS FROM POST'S IMAGE
    features_dict['numCharArticleTitle'] = lenArtTitle
    features_dict['numCharArticleDescr'] = lenArtDesc
    features_dict['numCharArticleKeywords'] = lenArtKeywords

    lenArtCap = 0
    for caption in row['targetCaptions']:
        lenArtCap = lenArtCap + len(caption)
    features_dict['numCharArticleCaption'] = lenArtCap

    lenArtPar = 0
    for paragraph in row['targetParagraphs']:
        lenArtPar = lenArtPar + len(paragraph)
    features_dict['numCharArticleParagraph'] = lenArtPar


truth = pd.read_json("train_set/truth.json")
instances = pd.read_json("train_set/instances.json")

instances['class'] = truth['truthClass']

# id postText postTimestamp postMedia targetTitle targetDescription targetKeywords targetParagraphs targetCaptions
cols = ['numCharPostTitle', 'numCharArticleTitle', 'numCharArticleDescr', 'numCharArticleKeywords',
        'numCharArticleCaption', 'numCharArticleParagraph', 'avgSimilarityPostTitle', 'capitalPostTitle',
        'avgWordLenPostTitle', 'avgWordLenArticleTitle', 'numDeterminers']

features = pd.DataFrame(columns=cols)

for index, row in instances.iterrows():
    features_dict = {}
    print("processing " + str(index) + " out of " + str(instances.shape[0]) + ": " + row['postText'][0])
    char_based_features(row, features_dict)
    features_dict['avgSimilarityPostTitle'] = avg_similarity(row['postText'][0])
    features_dict['capitalPostTitle'] = sum(1 for c in row['postText'][0] if c.isupper())
    features_dict['avgWordLenPostTitle'] = avg_word_length(row['postText'][0])
    features_dict['avgWordLenArticleTitle'] = avg_word_length(row['targetTitle'])
    features_dict['numDeterminers'] = count_determiners(row['postText'][0])
    features = features.append(features_dict, ignore_index=True)

features.to_csv("./features_matteo.csv", index=False)
print("mannaccia il cristo appeso")
print(features)

click_det = 0
no_click_det = 0

click = 0
no_click = 0

for index, row in instances.iterrows():
    if row['class'] == 'clickbait':
        click = click + 1
        if count_determiners(row['postText'][0]) > 0:
            click_det = click_det + 1
        else:
            no_click_det = no_click_det + 1
    else:
        no_click = no_click + 1

print("Number of clickbait articles: " + str(click))
print("Number of non-clickbait articles: " + str(no_click))

print("Number of clickbait articles with determiners: " + str(click_det))
print("Number of non-clickbait articles with determiners: " + str(no_click_det))
print("\n")
print("Percentage of clickbait articles with determiners: " + str((click_det) / click))
print("Percentage of non-clickbait articles with determiners: " + str((no_click_det) / no_click))
