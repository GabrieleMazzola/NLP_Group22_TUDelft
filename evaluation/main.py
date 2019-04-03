import random

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from gabry_dataset_parser import get_labeled_instances
import pandas as pd
from sklearn import preprocessing
import numpy as np

DATASET = 'big'  # 'big' or 'small'

PATH_TO_FEATURE_FOLDER = "../features/"

feature_paths = []
feature_paths.append(('POS postText normalized', PATH_TO_FEATURE_FOLDER + "{}/pos_features_{}_postText_normalized.csv".format(DATASET, DATASET)))
feature_paths.append(("Formal postText normalized", PATH_TO_FEATURE_FOLDER + "{}/formal_informal_features_{}_postText_normalized.csv".format(DATASET, DATASET)))
feature_paths.append(("Formal targetTitle normalized", PATH_TO_FEATURE_FOLDER + "{}/formal_informal_features_{}_targetTitle_normalized.csv".format(DATASET, DATASET)))

# TODO: wait for generated features for the big dataset and then add them.
#POS_TARGET_FEATURE_PATH = r"./pos_features_small_targetTitle_normalized.csv"
#SENTIMENT_WORDS_FEATURES_PATH = r"./bianca_features.csv"
#MATTEO_FEATURES_PATH = r"./matteo_features.csv"


data_df = get_labeled_instances("../train_set/instances_converted_{}.pickle".format(DATASET),
                                          "../train_set/truth_converted_{}.pickle".format(DATASET))[['id', 'truthClass']]
print(f"Labeled instances loaded. Shape: {data_df.shape}. Only 'id' and 'truthClass' kept.")


for feat_name, feat_path in feature_paths:
    print("-------------")
    print(f"Loading features: {feat_name}")
    feat_data = pd.read_csv(feat_path)
    print(f"Obtained {feat_data.shape[1] - 1} feature columns")
    feat_data['id'] = feat_data['id'].astype(str)
    print(f"Merging them with the original dataframe")
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


data_df = data_df.drop(['id', 'truthClass'], 1)
print(f"'id' and 'truthClass' dropped. Final shape of the feature dataframe: {data_df.shape}")
print(f"Columns : {list(data_df.columns)}")

print("\n\n---------------\nEVALUATION\n---------------\n")

eval_df = data_df.copy()
models = []
models.append(("Logistic ", LogisticRegression(solver='liblinear')))
#models.append(RandomForestClassifier(n_estimators=200, max_depth=200))


eval_metric = 'precision'
print(f"'{eval_metric}' is being used as evaluation metric")

CHECK_RANDOM = False
if CHECK_RANDOM:
    print("Performing test with random feature column.")
    random_feat = np.array([random.randint(-1, 1) for _ in range(eval_df.shape[0])])
    eval_df = pd.DataFrame(data=random_feat)

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)
for model_name, model in models:
    skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=36851234)

    for train_index, test_index in rskf.split(eval_df, label_encoded):

        X_train, X_test = eval_df.loc[train_index], eval_df.loc[test_index]

        y_train, y_test = label_encoded.loc[train_index], label_encoded.loc[test_index]

        clickbait_train = y_train[y_train['label'] == 1]
        clickbait_test = y_test[y_test['label'] == 1]
