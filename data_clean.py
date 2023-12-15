import numpy as np
import random
from maps import get_kmer_map

def clean_data(in_path, out_path, file_name, k=1, split=True):
    kmer_map = get_kmer_map(k)
    with open(f'{in_path}/{file_name}', 'r') as file:
        raw_data = file.readlines()

    raw_data = [x.strip() for x in raw_data]

    sequences = []
    for x in raw_data:
        if x[0] == '>':
            sequences.append('')
        else:
            sequences[-1] += x
    data = []
    for s in sequences:
        data.append([])
        for i in range(0, len(s)-k+1):
            data[-1].append(kmer_map[s[i:i+k]])
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

clean_data('raw_data', 'clean_data_k3_2', 'accessible.fasta', k=3)
clean_data('raw_data', 'clean_data_k3_2', 'notaccessible.fasta', k=3)
clean_data('raw_data', 'clean_data_k3_2', 'test.fasta', k=3, split=False)

# clean_data_submission('submissions', 'test', 'predictions_seq3.csv')