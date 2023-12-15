import numpy as np
import random

map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}


def clean_data(in_path, out_path, file_name, split=True):

    with open(f'{in_path}/{file_name}', 'r') as file:
        raw_data = file.readlines()

    raw_data = [x.strip() for x in raw_data]

    data = []
    for x in raw_data:
        if x[0] == '>':
            data.append([])
        else:
            for l in x:
                data[-1].append(map[l])
    random.shuffle(data)
    data = np.array(data)
    split_values = len(data)//10
    if split:
        np.save(f'{out_path}/{file_name.split(".")[0]}_train.npy', data[:split_values*7])
        np.save(f'{out_path}/{file_name.split(".")[0]}_val.npy', data[split_values*7:split_values*9])
        np.save(f'{out_path}/{file_name.split(".")[0]}_test.npy', data[split_values*9:])
    else:
        np.save(f'{out_path}/{file_name.split(".")[0]}.npy', data)


def clean_data_submission(in_path, out_path, file_name):

    with open(f'{in_path}/{file_name}', 'r') as file:
        raw_data = file.readlines()

    raw_data = [x.strip() for x in raw_data]

    data = []
    for x in raw_data:
        data.append([map[l] for l in x])
    data = np.array(data)
    np.save(f'{out_path}/{file_name.split(".")[0]}.npy', data)

clean_data('raw_data', 'clean_data', 'accessible.fasta')
clean_data('raw_data', 'clean_data', 'notaccessible.fasta')
clean_data('raw_data', 'clean_data', 'test.fasta', split=False)

# clean_data_submission('submissions', 'test', 'predictions_seq3.csv')