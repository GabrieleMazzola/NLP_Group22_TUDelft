import random

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

from gabry_dataset_parser import get_labeled_instances

DATASET = 'big'  # 'big' or 'small'

PATH_TO_FEATURE_FOLDER = "../features/"

feature_paths = []
feature_paths.append(('POS postText no-normalized pruned', PATH_TO_FEATURE_FOLDER + "{}/pos_features_{}_postText_no-normalized_infoGain70.0.csv".format(DATASET, DATASET)))
# alone -> 66.8 % log reg
feature_paths.append(('POS targetTitle normalized', PATH_TO_FEATURE_FOLDER + "{}/pos_features_{}_targetTitle_normalized.csv".format(DATASET, DATASET)))
# alone -> 62.7 % log reg
# POS together -> 67.3%

# feature_paths.append(("Formal postText normalized",  PATH_TO_FEATURE_FOLDER + "{}/formal_informal_features_{}_postText_normalized.csv".format(DATASET, DATASET))) # useless
# feature_paths.append(("Formal targetTitle normalized",PATH_TO_FEATURE_FOLDER + "{}/formal_informal_features_{}_targetTitle_normalized.csv".format(DATASET, DATASET))) # useless

feature_paths.append(("Matteo features",PATH_TO_FEATURE_FOLDER + "{}/matteo_features_full.csv".format(DATASET)))
# alone -> 69.4 % (logreg)


# feature_paths.append(("Similarity lin", PATH_TO_FEATURE_FOLDER + "{}/matteo_full_similarity.csv".format(DATASET)))
# alone -> 0 precision

feature_paths.append(("Bianca features", PATH_TO_FEATURE_FOLDER + "{}/bianca_features.csv".format(DATASET)))
# alone -> 68.7% logreg

feature_paths.append(("Ngrams", "../features/big/ngrams_features_counts_after_infoGain1.0.csv"))

data_df = get_labeled_instances("../train_set/instances_converted_{}.pickle".format(DATASET),
                                "../train_set/truth_converted_{}.pickle".format(DATASET))
data_df = data_df[['id', 'truthClass', 'truthMedian']]
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

medians = data_df['truthMedian']

data_df = data_df.drop(['id', 'truthClass', 'truthMedian'], 1)
print(f"'id' and 'truthClass' dropped. Final shape of the feature dataframe: {data_df.shape}")
print(f"Columns : {list(data_df.columns)}")

print("\n\n---------------\nEVALUATION\n---------------\n")

eval_df = data_df.copy()

CHECK_RANDOM = False
if CHECK_RANDOM:
    print("Performing test with random feature column.")
    random_feat = np.array([random.randint(-1, 1) for _ in range(eval_df.shape[0])])
    eval_df = pd.DataFrame(data=random_feat)

# Split in train (which we use for Grid Search) and test
X_train, X_test, y_train, y_test = train_test_split(eval_df, label_encoded, test_size=0.2, random_state=42)

param_randForest = {
    'n_estimators': [400],
    'max_depth': [100, 125, 150, 175, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy']
}

param_logisticReg = {
    'penalty': ['l1', 'l2'],
    #'C': np.logspace(-3, 3, 7)
}

param_svm = {'C': [1, 10, 100, 1000],
             'kernel': ['linear', 'rbf'],
             'gamma': ['auto', 'scale']
             }

models = []
models.append(("Random forest ", RandomForestClassifier(n_estimators=100, max_depth=200), param_randForest))
models.append(("Logistic ", LogisticRegression(solver='liblinear'), param_logisticReg))
# models.append(("SVM", svm.SVC(), param_svm))


eval_metric = 'precision'
print(f"'{eval_metric}' is being used as evaluation metric")

# results = []
# for model in models:
#     clf = GridSearchCV(model[1], model[2], cv=5, scoring=eval_metric)
#     clf.fit(X_train, y_train.values.ravel())
#     res = clf.cv_results_
#     results.append(res)
#     print(clf.best_params_, clf.best_score_)


for model_name, model, model_params in models:
    print(f"using model {model_name}")
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=36851234)
    precisions = []
    accuracies = []
    recalls = []
    f1s = []
    mses = []
    for train_index, test_index in skf.split(eval_df, label_encoded):

        X_train, X_test = eval_df.loc[train_index], eval_df.loc[test_index]
        y_train, y_test = label_encoded.loc[train_index], label_encoded.loc[test_index]
        y_medians = medians.loc[test_index]

        model.fit(X_train, y_train.values.ravel())
        y_pred_hard = model.predict(X_test)
        y_pred_soft = model.predict_proba(X_test)[:, 1]
        precision = precision_score(y_test, y_pred_hard)
        precisions.append(precision)
        accuracies.append(accuracy_score(y_test, y_pred_hard))
        recalls.append(recall_score(y_test, y_pred_hard))
        f1s.append(f1_score(y_test, y_pred_hard))
        mses.append(mean_squared_error(y_medians, y_pred_soft))

    print(model_name)
    print("acc", np.mean(accuracies), np.std(accuracies))
    print("prec", np.mean(precisions), np.std(precisions))
    print("rec", np.mean(recalls), np.std(recalls))
    print("f1", np.mean(f1s), np.std(f1s))
    print("mse", np.mean(mses), np.std(mses))
