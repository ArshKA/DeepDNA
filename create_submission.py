from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import numpy as np
import torch
from tqdm import tqdm

from model_k6 import model2
from maps import get_kmer_map

device = 'cuda:1'
k = 3
submission_num = 8

model = model2(device, kmers=k)
model.to(device)
model.load_state_dict(torch.load('models/k3_model_epoch15_0.6408.pt'))
model.eval()

def create_seq_file(seq_test, top_indices):
    kmer_map = get_kmer_map(k=k, inverse=True)
    sequence_predictions = []
    with open(f'submission_sequences/predictions{submission_num}.csv', 'w') as file:
        for idx in tqdm(top_indices):
            seq = ''

            seq = "".join([kmer_map[i.item()][0] for i in seq_test[idx][:-1]])
            seq += kmer_map[seq_test[idx, -1].item()]
            file.write(seq + "\n")
            sequence_predictions.append(seq)
    return sequence_predictions

def create_submission(submission_predictions):
    with open('raw_data/test.fasta', 'r') as file:
        unlabeled = file.readlines()

    unlabeled = [x.strip() for x in unlabeled]

    test = {}
    for line in unlabeled:
        if line[0] == '>':
            cur_seq = line[1:]
            test[cur_seq] = ""
        else:
            test[cur_seq] += line

    seq_to_id = {y: x for x, y in test.items()}
    with open(f'submissions/predictions{submission_num}.csv', 'w') as file:
        for p in submission_predictions:
            file.write(seq_to_id[p] + '\n')


seq_test = np.load(f'clean_data_k{k}/test.npy')
seq_test = torch.from_numpy(seq_test).to(device)

# Perform inference in batches of size 100
batch_size = 512
n_batches = len(seq_test) // batch_size
y_pred = []
with torch.no_grad():
    for i in range(n_batches):
        batch = seq_test[i * batch_size: (i + 1) * batch_size]
        y_pred.append(model(batch).cpu())
y_pred = torch.squeeze(torch.cat(y_pred, dim=0))

print((y_pred.argmax()))
print(y_pred.shape)
top = torch.topk(y_pred, 10000, sorted=True)

print((y_pred[top.indices].round() == 1).sum())

submission_predictions = create_seq_file(seq_test.cpu(), top.indices)

create_submission(submission_predictions)

