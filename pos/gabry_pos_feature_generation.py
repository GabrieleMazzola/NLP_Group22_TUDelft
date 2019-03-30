from collections import Counter
import pandas as pd
import nltk

from gabry_dataset_parser import get_labeled_instances


def generate_pos_features(sentence, ner_tagger, possible_tags, normalize):
    tokenized_text = nltk.word_tokenize(sentence)
    classified_text = ner_tagger.tag(tokenized_text)

    tokens_to_lowercase = [token[0] for token in classified_text if token[1] == 'O']
    lowercased_tokens = [token.lower() if token in tokens_to_lowercase else token for token in tokenized_text]

    tagging = nltk.pos_tag(lowercased_tokens)

    count = Counter([j for i, j in tagging])

    #print(count)
    sentence_tags = list(count.keys())
    sentence_counts = [count[tag] for tag in sentence_tags]

    if normalize:
        sentence_counts = [100*elem/sum(sentence_counts) for elem in sentence_counts]

    sentence_features = []
    for tag in possible_tags:
        try:
            idx = sentence_tags.index(tag)
            sentence_features.append(sentence_counts[idx])  # take count
        except Exception:
            sentence_features.append(0)  # tag not found in sentence --> 0

    return sentence_features


if __name__ == '__main__':
    print("Generating POS features... it might take a while :P")

    FEATURES_DATA_PATH = r"../features/pos_features_small_dataset_postText_NER-lowercasing_normalized.csv"

    labeled_instances = get_labeled_instances("../train_set/instances_converted.pickle",
                                              "../train_set/truth_converted.pickle")

    tagger = nltk.StanfordNERTagger('../ner/english.all.3class.distsim.crf.ser.gz',
                                    '../ner/stanford-ner.jar',
                                    encoding='utf-8')

    tagset = nltk.load("help/tagsets/upenn_tagset.pickle")
    possible_tags = list(tagset.keys())

    ids = list(labeled_instances.id)
    texts = [txt[0] for txt in list(labeled_instances.postText)]
    features = []
    for idx, txt in enumerate(texts, 1):
        print(f"Computing features for sample {idx} out of {len(texts)}...")
        features.append(generate_pos_features(txt, tagger, possible_tags, normalize=True))

    data_to_df = [tuple([ids[i]] + features[i]) for i in range(len(ids))]
    labels = ['id'] + ["pos_feat_" + tag for tag in possible_tags]

    df = pd.DataFrame.from_records(data_to_df, columns=labels)

    df.to_csv(FEATURES_DATA_PATH, index=False)

    print("Generation of POS features completed, phuff!")
