import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


from gabry_dataset_parser import get_labeled_instances

POS_FEATURE_PATH = r"./pos_features_small_dataset_NER-lowercasing_normalized.json"
labeled_instances = get_labeled_instances("../train_set/instances_converted.pickle",
                                          "../train_set/truth_converted.pickle")


pos_feature_df = pd.read_json(POS_FEATURE_PATH)
# TODO: what about those POS in which is all zero? theoretically, they do not contribute in any way to the classification. Avoid computation?
labeled_instances['id'] = labeled_instances['id'].astype(str)
pos_feature_df['id'] = pos_feature_df['id'].astype(str)

feat_df = pd.merge(pos_feature_df, labeled_instances, on=['id'])[list(pos_feature_df.columns) + ['id', 'truthClass']]


le = preprocessing.LabelEncoder()
label_encoded = le.fit_transform(feat_df['truthClass'])
print(label_encoded)

feat_df = feat_df.drop(['id', 'truthClass'], 1)

#Create a Gaussian Classifier
model = GaussianNB()

X_train, X_test, y_train, y_test = train_test_split(feat_df, label_encoded, test_size=0.2, random_state=0)

# Train the model using the training sets
model.fit(X_train, y_train)
scores = model.score(X_test, y_test)

df_to_visualize = np.vstack([feat_df.columns, feat_df.values])




print()