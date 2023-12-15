import numpy as np
import torch
def load_data(label, train=False, val=False, test=False):
    loaded_datasets = []
    if train:
        loaded_datasets.append(np.load(f'clean_data/{label}_train.npy'))
    if val:
        loaded_datasets.append(np.load(f'clean_data/{label}_val.npy'))
    if test:
        loaded_datasets.append(np.load(f'clean_data/{label}_test.npy'))

    return tuple(loaded_datasets)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def process_train_data():
    accessible_train, accessible_val = load_data('accessible', train=True, val=True)
    not_accessible_train, not_accessible_val = load_data('notaccessible', train=True, val=True)

    # accessible_train = np.tile(accessible_train, (10, 1))
    # accessible_val = np.tile(accessible_val, (10, 1))

    # not_accessible_train = not_accessible_train[:len(not_accessible_train)//10]
    # not_accessible_val = not_accessible_val[:len(not_accessible_val)//10]

    seq_train = np.concatenate([accessible_train, not_accessible_train])
    seq_val = np.concatenate([accessible_val, not_accessible_val])

    lab_train = np.concatenate([np.ones((accessible_train.shape[0], 1)), np.zeros((not_accessible_train.shape[0], 1))]).astype(
        np.int8)
    lab_val = np.concatenate([np.ones((accessible_val.shape[0], 1)), np.zeros((not_accessible_val.shape[0], 1))]).astype(np.int8)


    seq_train, lab_train = unison_shuffled_copies(seq_train, lab_train)
    seq_val, lab_val = unison_shuffled_copies(seq_val, lab_val)

    return seq_train, lab_train, seq_val, lab_val

def process_test_data():
    accessible_test = load_data('accessible', test=True)[0]
    not_accessible_test = load_data('notaccessible', test=True)[0]

    seq_test = np.concatenate([accessible_test, not_accessible_test])

    lab_test = np.concatenate([np.ones((accessible_test.shape[0], 1)), np.zeros((not_accessible_test.shape[0], 1))]).astype(np.int8)

    return seq_test, lab_test

def calculate_weights(tensor, w0=.5, w1=.5):
    count_0s = (tensor == 0).sum()
    count_1s = (tensor == 1).sum()

    weight_0s = w0 / count_0s
    weight_1s = w1 / count_1s

    weighted_tensor = tensor * weight_1s + (1 - tensor) * weight_0s
    return weighted_tensor

def top_k_accuracy(y_pred, labels, return_counts=False):
    correct = torch.topk(torch.squeeze(y_pred), (labels == 1).sum().item())
    correct = labels[correct.indices]
    top_acc = (correct.sum() / len(correct)).item()
    if return_counts:
        return top_acc, int(correct.sum())
    return top_acc
