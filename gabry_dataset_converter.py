import json
import pickle

if __name__ == '__main__':

    for file in ['truth', 'instances']:

        with open(f'./train_set/{file}.jsonl', 'r', encoding="utf8") as myfile:
            data = myfile.read()

        result = [json.loads(jline) for jline in data.split('\n')]

        with open(f"./train_set/{file}_converted.pickle", 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"{file} converted")
