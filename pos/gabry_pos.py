import pickle

import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk.tokenize import RegexpTokenizer

from gabry_dataset_parser import get_labeled_instances


tokenizer = RegexpTokenizer(r'\w+')  # removing punctuation
st = nltk.StanfordNERTagger('../ner/english.all.3class.distsim.crf.ser.gz',
					   '../ner/stanford-ner.jar',
                            encoding='utf-8')

labeled_instances = get_labeled_instances("../train_set/instances_converted.pickle",
                                          "../train_set/truth_converted.pickle")
clickbait_df = labeled_instances[labeled_instances.truthClass == 'clickbait']
no_clickbait_df = labeled_instances[labeled_instances.truthClass == 'no-clickbait']

LOWERCASE = True


def pos_tag_lowercase_all_but_NE(sentence):
    tokenized_text = nltk.word_tokenize(sentence)
    classified_text = st.tag(tokenized_text)

    tokens_to_lowercase = [token[0] for token in classified_text if token[1] == 'O']
    lowercased_tokens = [token.lower() if token in tokens_to_lowercase else token for token in tokenized_text]

    tagging = nltk.pos_tag(lowercased_tokens)
    print(tokens_to_lowercase)
    print(tagging)
    print("")
    return [tag[1] for tag in tagging]


def count_tags(sentences):
    counts = {}
    for idx, sentence in enumerate(sentences):
        if LOWERCASE:
            print(f"Lowercasing: {idx}/{len(sentences)}")
            pos_tagging = pos_tag_lowercase_all_but_NE(sentence)
        else:
            print(f"No lowercasing: {idx}/{len(sentences)}")
            tokens = nltk.word_tokenize(sentence)
            taggings = nltk.pos_tag(tokens)
            pos_tagging = [pos_tag[1] for pos_tag in taggings]
        for pos_tag in pos_tagging:
            counts[pos_tag] = counts.get(pos_tag, 0) + 1
    return counts


bait_post_texts = [elem[0] for elem in list(clickbait_df['postText'])]
bait_target_title = list(clickbait_df['targetTitle'])

no_bait_post_texts = [elem[0] for elem in list(no_clickbait_df['postText'])]
no_bait_target_title = list(no_clickbait_df['targetTitle'])

bait_counts = count_tags(bait_post_texts)
no_bait_counts = count_tags(no_bait_post_texts)

tags = list(set(list(bait_counts.keys()) + list(no_bait_counts.keys())))  # ordered
bait_tags_count = []
no_bait_tags_count = []
for tag in tags:
    bait_tags_count.append(bait_counts.get(tag, 0))
    no_bait_tags_count.append(no_bait_counts.get(tag, 0))

if not len(tags) == len(no_bait_tags_count) == len(bait_tags_count):
    raise Exception("ERRONEOUS LENGTHS")

zipped_counts = zip(tags, bait_tags_count, no_bait_tags_count)
zipped_counts = sorted(zipped_counts, key=lambda elem: elem[2], reverse=True)


with open(f"../features/postText_NER_no-punct-rem.pickle", 'wb') as f:
    pickle.dump(zipped_counts, f, protocol=pickle.HIGHEST_PROTOCOL)
