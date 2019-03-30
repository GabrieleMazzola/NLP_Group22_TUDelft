from nltk import WordNetLemmatizer
from nltk.corpus import words
from nltk.tokenize import RegexpTokenizer

from util import number_replacement


def extract_formal_informal_features(sentence, wordset):
    sentence = sentence.replace("RT", "")
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    sentence = sentence.lower()
    sentence = number_replacement(sentence, "")
    tokens = tokenizer.tokenize(sentence)
    lemmas = [lemmatizer.lemmatize(lemmatizer.lemmatize(token), pos='v') for token in tokens]

    formal = 0
    informal = 0
    for lemma in lemmas:
        if lemma in wordset:
            formal += 1
        else:
            informal += 1
    return formal / len(lemmas), informal / len(lemmas)


english_words = set(words.words())

test = "RT @CNNMoney: \"Emotions aren't really sanctioned in Corporate America,\" so is it OK to cry at work?"
feats = extract_formal_informal_features(test, english_words)
print(feats)