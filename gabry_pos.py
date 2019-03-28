import nltk

from gabry_dataset_parser import get_labeled_instances

labeled_instances = get_labeled_instances("./train_set/instances_converted.pickle", "./train_set/truth_converted.pickle")
clickbait_df = labeled_instances[labeled_instances.truthClass == 'clickbait']
no_clickbait_df = labeled_instances[labeled_instances.truthClass == 'no-clickbait']

bait_post_texts = list(clickbait_df['postText'])
bait_target_title = list(clickbait_df['targetTitle'])

no_bait_post_texts = list(no_clickbait_df['postText'])
no_bait_target_title = list(no_clickbait_df['targetTitle'])

tmp = bait_post_texts[0]
print(tmp)
print(nltk.pos_tag(nltk.word_tokenize(tmp)))
