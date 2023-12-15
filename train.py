import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score

from helpers import process_train_data, calculate_weights, top_k_accuracy
from model6 import model2

device = 'cuda:1'
trial_name = 'random_hyper'
batch_size = 512
learning_rate = .001
n_epochs = 100

# batch_size = 64
# learning_rate = .0003
# n_epochs = 100

seq_train, labels_train, seq_val, labels_val = process_train_data(k=3)
# Convert the numpy arrays to PyTorch tensors
seq_train = torch.from_numpy(seq_train).int().to(device)
seq_test = torch.from_numpy(seq_val).int().to(device)
lab_train = torch.from_numpy(labels_train).float().to(device)
lab_test = torch.from_numpy(labels_val).float().to(device)

# Create PyTorch datasets from the tensors
train_dataset = TensorDataset(seq_train, lab_train)
test_dataset = TensorDataset(seq_test, lab_test)

# Create PyTorch raw_data loaders from the datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# Model
model = model2(device)
model.to(device)
criterion = nn.BCELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

best_acc = - np.inf  # init to negative infinity
best_weights = None

for epoch in range(n_epochs):
    model.train()

    train_loss = []
    train_acc = []
    bar = tqdm(train_loader)
    bar.set_description(f"Epoch {epoch}")
    for seqs, labels in bar:
        y_pred = model(seqs)
        weights = calculate_weights(labels)
        intermediate_losses = criterion(y_pred, labels)
        weighted_loss = torch.mean(weights*intermediate_losses)
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        # print progress
        acc = (y_pred.round() == labels).float().mean()
        train_loss.append(weighted_loss.item())
        train_acc.append(acc.item())
        bar.set_postfix(
            loss=float(weighted_loss),
            acc=float(acc)
        )
    # evaluate accuracy at end of each epoch
    model.eval()

    with torch.no_grad():
        val_loss = 0.0
        val_acc = 0.0
        f1 = 0.0
        top_acc = 0.0
        for seqs, labels in val_loader:
            y_pred = model(seqs)
            weights = calculate_weights(labels)
            val_loss += torch.mean(weights*criterion(y_pred, labels))
            val_acc += (y_pred.round() == labels).float().mean()
            f1 += f1_score(labels.cpu(), y_pred.round().cpu())
            top_acc += top_k_accuracy(y_pred, labels)
        val_acc /= len(val_loader)
        val_loss /= len(val_loader)
        f1 /= len(val_loader)
        top_acc /= len(val_loader)
        if top_acc > best_acc:
            print(f"Saving epoch {epoch} with {top_acc:.2%} top k accuracy")
            torch.save(model.state_dict(), f'models/{trial_name}_epoch{epoch}_{top_acc:.4f}.pt')
            best_acc = top_acc

    print(f"Finished Epoch {epoch}, Train Loss: {sum(train_loss)/len(train_loss):.4f}, Train Accuracy: {sum(train_acc)/len(train_acc):.2%}, Test Loss: {val_loss:.4f}, Test Accuracy: {val_acc:.2%}, F1 Score: {f1:.4f}, Top K Accuracy: {top_acc:.2%}")






