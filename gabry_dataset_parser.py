import pickle
import pandas as pd


def read_pickle(filepath):
    with open(filepath, 'rb') as f:
        b = pickle.load(f)
    return b


def get_labeled_instances(instances_path, truth_path):
    instances = read_pickle(instances_path)
    truth = read_pickle(truth_path)
    instances = pd.DataFrame(instances)
    truth = pd.DataFrame(truth)
    labeled_instances = pd.merge(instances, truth, on=['id'])
    return labeled_instances


if __name__ == '__main__':
    labeled_instances = get_labeled_instances("./train_set/instances_converted_small.pickle", "./train_set/truth_converted_small.pickle")

    clickbait_df = labeled_instances[labeled_instances.truthClass == 'clickbait']
    no_clickbait_df = labeled_instances[labeled_instances.truthClass == 'no-clickbait']

    print(f"{labeled_instances.shape[0]} instances in total. {clickbait_df.shape[0]} clickbait, {no_clickbait_df.shape[0]} no-clickbait")

