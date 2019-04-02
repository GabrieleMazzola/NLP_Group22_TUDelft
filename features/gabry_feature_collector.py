import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


from gabry_dataset_parser import get_labeled_instances

POS_TEXT_FEATURE_PATH = r"./big/pos_features_big_postText_normalized.csv"
#POS_TARGET_FEATURE_PATH = r"./pos_features_small_targetTitle_normalized.csv"
#SENTIMENT_WORDS_FEATURES_PATH = r"./bianca_features.csv"
#MATTEO_FEATURES_PATH = r"./matteo_features.csv"
#FORMAL_INFORMAL_POST_FEATURES_PATH = r"./formal_informal/formal_informal_features_small_postText_normalized.csv"
#FORMAL_INFORMAL_TARGET_FEATURES_PATH = r"./formal_informal/formal_informal_features_small_targetTitle_normalized.csv"


labeled_instances = get_labeled_instances("../train_set/instances_converted_big.pickle",
                                          "../train_set/truth_converted_big.pickle")
print(f"Labeled instances loaded. Shape: {labeled_instances.shape}")

#### FEATURES LOADING ####
pos_post_feature_df = pd.read_csv(POS_TEXT_FEATURE_PATH)
print(f"POS features post loaded. Shape: {pos_post_feature_df.shape}")

# pos_target_feature_df = pd.read_csv(POS_TARGET_FEATURE_PATH)
# print(f"POS features target loaded. Shape: {pos_target_feature_df.shape}")
#
# sent_word_features_df = pd.read_csv(SENTIMENT_WORDS_FEATURES_PATH)
# print(f"Sentiment and Words features loaded. Shape: {sent_word_features_df.shape}")
#
# matteo_features_df = pd.read_csv(MATTEO_FEATURES_PATH)
# print(f"Matteo features loaded. Shape: {matteo_features_df.shape}")
#
# form_inform_post = pd.read_csv(FORMAL_INFORMAL_POST_FEATURES_PATH)
# print(f"Formal post features loaded. Shape: {form_inform_post.shape}")
#
# form_inform_target = pd.read_csv(FORMAL_INFORMAL_TARGET_FEATURES_PATH)
# print(f"Formal target features loaded. Shape: {form_inform_target.shape}")

#### FEATURES LOADING DONE ####


labeled_instances['id'] = labeled_instances['id'].astype(str)
pos_post_feature_df['id'] = pos_post_feature_df['id'].astype(str)
# pos_target_feature_df['id'] = pos_target_feature_df['id'].astype(str)
# sent_word_features_df['id'] = sent_word_features_df['id'].astype(str)
# matteo_features_df['id'] = matteo_features_df['id'].astype(str)
# form_inform_post['id'] = form_inform_post['id'].astype(str)
# form_inform_target['id'] = form_inform_target['id'].astype(str)


print("Merging features into one big dataframe...")
data_df = labeled_instances
data_df = pd.merge(data_df, pos_post_feature_df, on=['id'])
# data_df = pd.merge(data_df, pos_target_feature_df, on=['id'])
# data_df = pd.merge(data_df, sent_word_features_df, on=['id'])
# data_df = pd.merge(data_df, matteo_features_df, on=['id'])
# data_df = pd.concat([data_df, matteo_features_df.drop('id', 1)], 1)
# data_df = pd.merge(data_df, form_inform_post, on=['id'])
# data_df = pd.merge(data_df, form_inform_target, on=['id'])

print(f"Merged. Shape: {data_df.shape}")

le = preprocessing.LabelEncoder()
label_encoded = le.fit_transform(data_df['truthClass'])
print("Labels encoded")

feat_df = data_df[
    list(pos_post_feature_df.columns)
    ]
feat_df = feat_df.drop(['id'], 1)
print(f"Kept only features columns. Shape: {feat_df.shape}")

print(list(feat_df.columns))

models = []

models.append(LogisticRegression(solver='liblinear'))
#models.append(SVC(gamma='auto'))
models.append(RandomForestClassifier(n_estimators=300, max_depth=300))
models.append(GaussianNB())
models.append(MultinomialNB())

for model in models:
    scores = cross_val_score(model, feat_df, label_encoded, cv=5)
    print(np.mean(scores), np.std(scores))


df_to_visualize = np.vstack([feat_df.columns, feat_df.values])




print()