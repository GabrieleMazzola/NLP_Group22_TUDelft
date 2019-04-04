import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif
from gabry_dataset_parser import get_labeled_instances

DATASET = 'big'

path = "../features/{}/pos_features_{}_postText_no-normalized{}.csv"
POS_FEAT_PATH = path.format(DATASET, DATASET, "")
feat_data = pd.read_csv(POS_FEAT_PATH)


data_df = get_labeled_instances("../train_set/instances_converted_{}.pickle".format(DATASET),
                                "../train_set/truth_converted_{}.pickle".format(DATASET))[['id', 'truthClass']]
print(f"Labeled instances loaded. Shape: {data_df.shape}. Only 'id' and 'truthClass' kept.")
feat_data['id'] = feat_data['id'].astype(str)
data_df = pd.merge(data_df, feat_data, on=['id'])

le = preprocessing.LabelEncoder()
label_encoded = le.fit_transform(data_df['truthClass'])
label_encoded = [1 if lab == 0 else 0 for lab in list(label_encoded)]
print(f"Labels encoded. Class '{data_df['truthClass'][0]}' --> label '{label_encoded[0]}'")
label_encoded = pd.DataFrame(label_encoded, columns=['label'])

data_df = data_df.drop(['id', 'truthClass'], 1)

print("Computing infogain...")
info_gains = mutual_info_classif(data_df, label_encoded.values.ravel())

features_gains = list(zip(list(data_df.columns), list(info_gains)))
features_gains = sorted(features_gains, key=lambda x: x[1], reverse=True)
print(f"{len(features_gains)} total features.")
non_zero_features = [feat for feat in features_gains if feat[1] > 0]
print(f"{len(non_zero_features)} features are different from zero.")

TOP_PERC = 0.7
selected_features = non_zero_features[:int(TOP_PERC*len(non_zero_features))]
selected_features = [feat[0] for feat in selected_features]
print(f"{len(selected_features)} selected features.")

feat_data = pd.read_csv(POS_FEAT_PATH)
feat_data = feat_data[['id'] + selected_features]

feat_data.to_csv(path.format(DATASET, DATASET, "_infoGain" + str(TOP_PERC*100)), index=False)
print(feat_data.shape)
print([feat for feat in features_gains])
