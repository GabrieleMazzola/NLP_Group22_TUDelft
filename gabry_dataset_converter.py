import json
import pickle

if __name__ == '__main__':

    DATASET = 'big'  # 'small' or 'big'

    for file in ['truth_{}'.format(DATASET), 'instances_{}'.format(DATASET)]:

        with open(f'./train_set/{file}.jsonl', 'r', encoding="utf8") as myfile:
            data = myfile.read()

        print(f"Opened file {file}")
        if DATASET == 'big':
            result = [json.loads(jline) for jline in data.split('\n')[:-1]]
        else:
            result = [json.loads(jline) for jline in data.split('\n')]

        print(f"Read {len(result)} lines.")

        with open(f"./train_set/{file.split('_')[0]}_converted_{DATASET}.pickle", 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"{file} converted")
