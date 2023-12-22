import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from tqdm import tqdm


from helpers import process_test_data, top_k_accuracy
from model_k6 import model2

device = 'cuda:1'
k = 3

seq_test, labels_test = process_test_data(k=k)

seq_test = torch.from_numpy(seq_test).int().to(device)
labels_test = torch.from_numpy(labels_test).float().to(device)

model = model2(device, kmers=k)
model.load_state_dict(torch.load('models/k3_model_epoch15_0.6408.pt'))
model.to(device)
model.eval()

batch_size = 512
y_pred = []
with torch.no_grad():
    for i in range(len(seq_test) // batch_size + 1):
        y_pred.append(model(seq_test[i * batch_size: (i + 1) * batch_size]).cpu())
y_pred = torch.cat(y_pred, dim=0)

labels_test = labels_test.cpu()
test_acc = (y_pred.round() == labels_test).float().mean()

top_acc, correct_n = top_k_accuracy(y_pred, labels_test, return_counts=True)
print(f"Accuracy: {test_acc:.2%}, TopK Accuracy: {correct_n}/{(labels_test == 1).sum()} or {top_acc:.2%} out of {len(labels_test)} total samples")