import pandas as pd

test = pd.read_csv("./features_matteo.csv")
instances = pd.read_json("train_set/instances.json")

test['id'] = instances['id']
cols = test.columns.tolist()
cols = cols[-1:] + cols[:-2]
test = test[cols]

test.to_json("./matteo_feat.json")


print(max(test['avgSimilarityPostTitle']))