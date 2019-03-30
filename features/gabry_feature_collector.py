import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


from gabry_dataset_parser import get_labeled_instances

POS_FEATURE_PATH = r"./pos_features_small_dataset_postText_NER-lowercasing_normalized.csv"
labeled_instances = get_labeled_instances("../train_set/instances_converted.pickle",
                                          "../train_set/truth_converted.pickle")
print(f"Labeled instances loaded. Shape: {labeled_instances.shape}")

pos_feature_df = pd.read_csv(POS_FEATURE_PATH)
print(f"POS features loaded. Shape: {pos_feature_df.shape}")

labeled_instances['id'] = labeled_instances['id'].astype(str)
pos_feature_df['id'] = pos_feature_df['id'].astype(str)

print("Merging features into one big dataframe...")
data_df = pd.merge(pos_feature_df, labeled_instances, on=['id'])
print(f"Merged. Shape: {data_df.shape}")

le = preprocessing.LabelEncoder()
label_encoded = le.fit_transform(data_df['truthClass'])
print("Labels encoded")

feat_df = data_df[list(pos_feature_df.columns)]
print(f"Kept only features columns. Shape: {feat_df.shape}")

model = GaussianNB()
scores = cross_val_score(model, feat_df, label_encoded, cv=5)
print(np.mean(scores), np.std(scores))

df_to_visualize = np.vstack([feat_df.columns, feat_df.values])




print()