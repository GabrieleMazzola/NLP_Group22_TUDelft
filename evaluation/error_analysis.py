import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, mean_squared_error, \
    confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from gabry_dataset_parser import get_labeled_instances

DATASET = 'big'  # 'big' or 'small'

PATH_TO_FEATURE_FOLDER = "../features/"

feature_paths = []
feature_paths.append(('POS postText no-normalized pruned', PATH_TO_FEATURE_FOLDER + "{}/pos_features_{}_postText_no-normalized_infoGain70.0.csv".format(DATASET, DATASET)))

feature_paths.append(('POS targetTitle normalized pruned', PATH_TO_FEATURE_FOLDER + "{}/pos_features_{}_targetTitle_normalized_infoGain70.0.csv".format(DATASET, DATASET)))


feature_paths.append(("Formal postText normalized",  PATH_TO_FEATURE_FOLDER + "{}/formal_informal_features_{}_postText_normalized.csv".format(DATASET, DATASET))) # useless
feature_paths.append(("Formal targetTitle normalized",PATH_TO_FEATURE_FOLDER + "{}/formal_informal_features_{}_targetTitle_normalized.csv".format(DATASET, DATASET))) # useless

feature_paths.append(("Matteo features",PATH_TO_FEATURE_FOLDER + "{}/matteo_features_full.csv".format(DATASET)))


feature_paths.append(("Similarity lin", PATH_TO_FEATURE_FOLDER + "{}/matteo_full_similarity.csv".format(DATASET)))

feature_paths.append(("Bianca features", PATH_TO_FEATURE_FOLDER + "{}/bianca_features.csv".format(DATASET)))

feature_paths.append(("Ngrams", "../features/big/ngrams_features_counts_after_infoGain1.0.csv"))

data_df = get_labeled_instances("../train_set/instances_converted_{}.pickle".format(DATASET),
                                "../train_set/truth_converted_{}.pickle".format(DATASET))
data_df = data_df[['id', 'truthClass', 'truthMean']]
print(f"Labeled instances loaded. Shape: {data_df.shape}. Only 'id' and 'truthClass' kept.")

for feat_name, feat_path in feature_paths:
    print("-------------")
    print(f"Loading features: {feat_name}")
    feat_data = pd.read_csv(feat_path)
    print(f"Obtained {feat_data.shape[1] - 1} feature columns")

    feat_data['id'] = feat_data['id'].astype(str)

    print(f"Merging them with the original dataframe")

    if feat_name == 'Matteo features':
        data_df = pd.concat([data_df, feat_data.drop('id', 1)], 1)
    else:
        data_df = pd.merge(data_df, feat_data, on=['id'])
    print(f"Obtained shape: {data_df.shape}")

    print("-------------\n")

print("\n-------------\nEncoding labels.")
le = preprocessing.LabelEncoder()
label_encoded = le.fit_transform(data_df['truthClass'])
label_encoded = [1 if lab == 0 else 0 for lab in list(label_encoded)]
print(f"Labels encoded. Class '{data_df['truthClass'][0]}' --> label '{label_encoded[0]}'")
label_encoded = pd.DataFrame(label_encoded, columns=['label'])
print("-------------\n")


model = RandomForestClassifier(criterion='entropy', max_depth=125, max_features='log2', n_estimators=400)

# Split in train (which we use for Grid Search) and test
X_train_with_ID, X_test_with_ID, y_train, y_test = train_test_split(data_df, label_encoded, test_size=0.2, random_state=42)

means = data_df['truthMean']
y_means = means.loc[pd.Series(y_test.index)]

data_df = data_df.drop(['id', 'truthClass', 'truthMean'], 1)
X_test = X_test_with_ID.drop(['id', 'truthClass', 'truthMean'], 1)
X_train = X_train_with_ID.drop(['id', 'truthClass', 'truthMean'], 1)

print(f"'id' and 'truthClass' dropped. Final shape of the feature dataframe: {data_df.shape}")
print(f"Columns : {list(data_df.columns)}")

print("\n\n---------------\nEVALUATION\n---------------\n")

eval_df = data_df.copy()




model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("acc", accuracy)
print("prec", precision)
print("rec", recall)
print("f1", f1)

y_test['pred'] = y_pred
misclassified = y_test[y_test['label'] != y_test['pred']]
misclassified_with_features = X_test_with_ID.loc[misclassified.index]

data_df = get_labeled_instances("../train_set/instances_converted_{}.pickle".format(DATASET),
                                "../train_set/truth_converted_{}.pickle".format(DATASET))

misclassified_data_with_features = pd.merge(misclassified_with_features, data_df, on='id')
print("saving")
misclassified_data_with_features.to_csv("./miscl_with_feat.csv", index=False)
print()