import pandas as pd

test = pd.read_csv("./features_matteo.csv")

print(max(test['avgSimilarityPostTitle']))