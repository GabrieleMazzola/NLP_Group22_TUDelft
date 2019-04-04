import string
import json_lines
import nltk
import pandas as pd
import itertools
from nltk.corpus import stopwords
from nltk.corpus import wordnet_ic
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import os

java_path = "C:\\Program Files (x86)\\Common Files\\Oracle\\Java\\javapath\\java.exe"
os.environ['JAVAHOME'] = java_path

tmp = pd.DataFrame()
with open('C:\\Users\\matte\\Desktop\\instances.jsonl', 'rb') as f: # opening file in binary(rb) mode
   i = 0
   for item in json_lines.reader(f):
       print('processing ' + str(i))
       # print(type(item)) #or use print(item['X']) for printing specific data
       tmp = tmp.append(item, ignore_index=True)
       i = i + 1

stop_words = set(stopwords.words('english'))
stop_words.add('\'s')
stop_words.add('RT')
stop_words.add('\'re')
st = StanfordNERTagger(
    'C:\\Users\\matte\\Desktop\\TU Delft\\Quarter III\\Information Retrieval\\NLP Project\\stanford-ner-2018-10-16\\classifiers\\english.all.3class.distsim.crf.ser.gz',
    'C:\\Users\\matte\\Desktop\\TU Delft\\Quarter III\\Information Retrieval\\NLP Project\\stanford-ner-2018-10-16\\stanford-ner.jar',
    encoding='utf-8')

semcor_ic = wordnet_ic.ic('ic-semcor.dat')
general_phrases = ['RT', "'s", "'re", "u", 'http', 'https']
with open("matteo/common_phrases.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
common_phrases = [x.strip() for x in content]


def check(string, sub_str):
    if string.find(sub_str) == -1:
        return 0
    else:
        return 1


def common(sentence):
    res = 0
    for phrase in common_phrases:
        res = check(sentence, phrase)
        if res == 1:
            print("DEEEEE CRII")
            return res
    return res


def count_capital(sentence):
    tokenized_text = word_tokenize(sentence)
    classified_text = st.tag(tokenized_text)
    filtered = []
    for tup in classified_text:
        if tup[1] != 'O':
            filtered.append(tup[0].lower())
        else:
            filtered.append(tup[0])
    capital = 0
    for word in filtered:
        capital = capital + sum(1 for c in word if c.isupper())
    return [capital/len(tokenized_text), len(tokenized_text)]


def count_determiners(sentence):
    text = word_tokenize(sentence)
    text = [w.lower() for w in text]
    tags = nltk.pos_tag(text)
    count = 0
    for tup in tags:
        if tup[1] == 'DT':
            count = count + 1
    return count


def pre_process(sentence):
    filtered = []
    for word in sentence.split():
        if word[0] != '@' and word[0] != '#' and word not in general_phrases:
            filtered.append(word)
    return list_to_string(filtered)


def ner_stanford(sentence):
    tokenized_text = word_tokenize(sentence)
    classified_text = st.tag(tokenized_text)
    filtered = []
    for tup in classified_text:
        if tup[1] != 'PERSON' and tup[1] != 'ORGANIZATION' and tup[1] != 'LOCATION' and tup[0] not in general_phrases:
            filtered.append(tup[0])
    return filtered


def list_to_string(filtered_sentence):
    sentence = ''
    for word in filtered_sentence:
        sentence = sentence + ' ' + word
    return sentence


def lemmatize_string(filtered_sentence):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos="v") for word in filtered_sentence]


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
        if not (word.isalpha()) or len(word) <= 2:
            list_of_words.remove(word)
    return list_of_words


def avg_similarity(sentence):
    without_hash = pre_process(sentence)
    filtered_sentence = ner_stanford(without_hash)
    filtered_sentence = clean_string(list_to_string(filtered_sentence))
    filtered_sentence = lemmatize_string(filtered_sentence)
    final_sentence = final_check(filtered_sentence)
    final_sentence = list(dict.fromkeys(final_sentence))
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
    features_dict['numCharPostTitle'] = lenPostTitle
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


# truth = pd.read_json("/home/esilezz/Scrivania/nlp_project_22/train_set/truth_small.json")
# instances = pd.read_json("/home/esilezz/Scrivania/nlp_project_22/train_set/instances_small.json")

instances = tmp

# instances['class'] = truth['truthClass']

# id postText postTimestamp postMedia targetTitle targetDescription targetKeywords targetParagraphs targetCaptions
# cols = ['id', 'numCharPostTitle', 'numCharArticleTitle', 'numCharArticleDescr', 'numCharArticleKeywords',
#         'numCharArticleCaption', 'numCharArticleParagraph', 'avgSimilarityPostTitle',
#         'avgWordLenPostTitle', 'avgWordLenArticleTitle', 'capitalPostTitle', 'numWords', 'commonPhrase']
cols = ['id', 'avgSimilarityPostTitle']

features = pd.DataFrame(columns=cols)

for index, row in instances.iterrows():
    features_dict = {}
    print("processing " + str(index) + " out of " + str(instances.shape[0]) + ": " + row['postText'][0])
    features_dict['id'] = str(row['id'])
    # char_based_features(row, features_dict)
    features_dict['avgSimilarityPostTitle'] = avg_similarity(row['postText'][0])
    # features_dict['avgWordLenPostTitle'] = avg_word_length(row['postText'][0])
    # features_dict['avgWordLenArticleTitle'] = avg_word_length(row['targetTitle'])
    # capital_plus_nwords = count_capital(row['postText'][0])
    # features_dict['capitalPostTitle'] = capital_plus_nwords[0]
    # features_dict['numWords'] = capital_plus_nwords[1]
    # features_dict['commonPhrase'] = common(row['postText'][0])
    features = features.append(features_dict, ignore_index=True)

features.to_csv("./matteo_full_similarity.csv", index=False)
print(features)

# click_det = 0
# no_click_det = 0
#
# click = 0
# no_click = 0
#
# for index, row in instances.iterrows():
#     if row['class'] == 'clickbait':
#         click = click + 1
#         if count_determiners(row['postText'][0]) > 0:
#             click_det = click_det + 1
#         else:
#             no_click_det = no_click_det + 1
#     else:
#         no_click = no_click + 1
#
# print("Number of clickbait articles: " + str(click))
# print("Number of non-clickbait articles: " + str(no_click))
#
# print("Number of clickbait articles with determiners: " + str(click_det))
# print("Number of non-clickbait articles with determiners: " + str(no_click_det))
# print("\n")
# print("Percentage of clickbait articles with determiners: " + str((click_det) / click))
# print("Percentage of non-clickbait articles with determiners: " + str((no_click_det) / no_click))
