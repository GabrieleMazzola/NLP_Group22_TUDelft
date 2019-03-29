from collections import Counter

import nltk


def generate_pos_features(sentence, ner_tagger, tagset):
    tokenized_text = nltk.word_tokenize(sentence)
    classified_text = ner_tagger.tag(tokenized_text)

    tokens_to_lowercase = [token[0] for token in classified_text if token[1] == 'O']
    lowercased_tokens = [token.lower() if token in tokens_to_lowercase else token for token in tokenized_text]

    tagging = nltk.pos_tag(lowercased_tokens)

    count = Counter([j for i, j in tagging])

    sentence_tags = list(count.keys())
    sentence_counts = [count[tag] for tag in sentence_tags]

    possible_tags = list(tagset.keys())
    sentence_features = []
    for tag in possible_tags:
        try:
            idx = sentence_tags.index(tag)
            sentence_features.append(sentence_counts[idx])  # take count
        except Exception:
            sentence_features.append(0)  # tag not found in sentence --> 0

    print(list(zip(possible_tags, sentence_features)))

    return sentence_features
