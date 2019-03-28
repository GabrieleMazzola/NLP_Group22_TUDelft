import string
import pandas as pd
import itertools
from nltk.corpus import stopwords
from nltk.corpus import wordnet_ic
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer


def stem_string(filtered_sentence):
    ps = PorterStemmer()
    return [ps.stem(item) for item in filtered_sentence]


def clean_string(sentence):
    stop_words = set(stopwords.words('english'))

    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence


def avg_similarity(sentence):
    semcor_ic = wordnet_ic.ic('ic-semcor.dat')
    filtered_sentence = clean_string(sentence)
    # filtered_sentence = stem_string(filtered_sentence)
    combi = set(itertools.combinations(filtered_sentence, 2))
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
    avg = sum / len(combi)
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
        'avgWordLenPostTitle', 'avgWordLenArticleTitle']

features = pd.DataFrame(columns=cols)

for index, row in instances.iterrows():
    features_dict = {}
    char_based_features(row, features_dict)
    features_dict['avgSimilarityPostTitle'] = avg_similarity(row['postText'][0])
    features_dict['capitalPostTitle'] = sum(1 for c in row['postText'][0] if c.isupper())
    features_dict['avgWordLenPostTitle'] = avg_word_length(row['postText'][0])
    features_dict['avgWordLenArticleTitle'] = avg_word_length(row['targetTitle'])

    features = features.append(features_dict, ignore_index=True)

print("mannaccia il cristo appeso")
print(features)
