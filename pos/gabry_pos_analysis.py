import nltk

from nltk.tag import StanfordNERTagger
from pos.gabry_pos_feature_generation import generate_pos_features

post_text = "The guy who’s followed his girlfriend around the world photographed their wedding perfectly"
target_article = "The Dude Who Followed Christine Around The World Just Photographed Their Wedding Perfectly"

st = StanfordNERTagger('../ner/english.all.3class.distsim.crf.ser.gz',
                       '../ner/stanford-ner.jar',
                       encoding='utf-8')

tagset = nltk.load("help/tagsets/upenn_tagset.pickle")
feats = generate_pos_features(target_article, st, tagset)
print(feats)
print(len(feats))
