from gabry_dataset_parser import get_labeled_instances
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import csv

labeled_instances = get_labeled_instances("./train_set/instances_converted.pickle",
                                          "./train_set/truth_converted.pickle")


print(labeled_instances.columns)


def get_normalized_stopwords_count(sentence):
    stopWords = set(stopwords.words('english'))

    sentence_lower = sentence.lower()
    word_tokens = word_tokenize(sentence_lower)
    stopwords_in_sentence = [w for w in word_tokens if w in stopWords]

    return len(stopwords_in_sentence)/len(word_tokens)

def get_sentiment_features(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)

    return score

def get_normalized_shortenings_count(sentence):
    text_file = open("shortenings.txt", "r")
    shortenings = text_file.read().split('\n')

    sentence_lower = sentence.lower()
    words = sentence_lower.split()
    shortenings_in_sentence = [w for w in words if w in shortenings]

    return len(shortenings_in_sentence)/len(words)

def get_slang_words_list():
    slang_data = []
    with open('slang_dict.doc', 'r') as exRtFile:
        exchReader = csv.reader(exRtFile, delimiter='`', quoting=csv.QUOTE_NONE)
        for row in exchReader:
            slang_data.append(row)

    slang_words = []
    for i, text in enumerate(slang_data):
        slang_words.append(text[0])
    return slang_words

def get_normalized_slang_count(sentence):
    slang_words = get_slang_words_list()

    text_post = sentence.replace("RT", "")
    words = text_post.split()
    slang_in_sentence = [w for w in words if w in slang_words]

    return len(slang_in_sentence)/ len(words)

extracted_features = {}
extracted_features['id'] = []
extracted_features['normalized_stopwords_count'] = []
extracted_features['positive_sentiment_score'] = []
extracted_features['negative_sentiment_score'] = []
extracted_features['polarity_score'] = []
extracted_features['normalized_shortenings_count'] = []
extracted_features['normalized_slang_count'] = []

for i, id in enumerate(labeled_instances['id']):
    extracted_features['id'].append(id)
    extracted_features['normalized_stopwords_count'].append(get_normalized_stopwords_count(labeled_instances['postText'][i][0]))

    sentiment_scores = get_sentiment_features(labeled_instances['postText'][i][0])
    extracted_features['positive_sentiment_score'].append(sentiment_scores['pos'])
    extracted_features['negative_sentiment_score'].append(sentiment_scores['neg'])
    extracted_features['polarity_score'].append(1-sentiment_scores['neu'])

    extracted_features['normalized_shortenings_count'].append(get_normalized_shortenings_count(labeled_instances['postText'][i][0]))
    extracted_features['normalized_slang_count'].append(get_normalized_slang_count(labeled_instances['postText'][i][0]))

print(extracted_features)

df_features = pd.DataFrame.from_dict(extracted_features)
df_features.to_csv('bianca_features.csv', index=False)

