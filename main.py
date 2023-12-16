import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from model import SequenceModel

device = 'cuda:1'

# accessible = np.load('clean_data_k1/accessible.npy')
#
# not_accessible = np.load('clean_data_k1/notaccessible.npy')

accessible_train = np.load('clean_data_k1/accessible_train.npy')
accessible_test = np.load('clean_data_k1/accessible_test.npy')

not_accessible_train = np.load('clean_data_k1/notaccessible_train.npy')
not_accessible_test = np.load('clean_data_k1/notaccessible_test.npy')

# labels = np.concatenate([np.ones(accessible.shape[0]), np.zeros(not_accessible.shape[0])]).astype(np.int8)
#
#
# # labels = np.expand_dims(labels, axis=-1)
# labels = np.eye(2)[labels]

# sequences = np.concatenate([accessible, not_accessible])



# seq_train, seq_test, lab_train, lab_test = train_test_split(sequences, labels, test_size=0.2)


seq_train = np.concatenate([accessible_train, not_accessible_train])
seq_test = np.concatenate([accessible_test, not_accessible_test])

lab_train = np.concatenate([np.ones(accessible_train.shape[0]), np.zeros(not_accessible_train.shape[0])]).astype(np.int8)
lab_test = np.concatenate([np.ones(accessible_test.shape[0]), np.zeros(not_accessible_test.shape[0])]).astype(np.int8)

lab_train = np.eye(2)[lab_train]
lab_test = np.eye(2)[lab_test]

print(lab_test.shape, lab_test.argmax(axis=1).mean())
print(lab_train.shape, lab_train.argmax(axis=1).mean())

# Create an instance of the model
model = SequenceModel(device)

# model.load_state_dict(torch.load('models/third.pt'))

model.to(device)
# Define the binary crossentropy loss function
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.2, .8]).to(device))

# Define the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=.001)


# Import PyTorch and other libraries

# Define the batch size
batch_size = 256

# Convert the numpy arrays to PyTorch tensors
seq_train = torch.from_numpy(seq_train).int().to(device)
seq_test = torch.from_numpy(seq_test).int().to(device)
lab_train = torch.from_numpy(lab_train).float().to(device)
lab_test = torch.from_numpy(lab_test).float().to(device)

# Create PyTorch datasets from the tensors
train_dataset = TensorDataset(seq_train, lab_train)
test_dataset = TensorDataset(seq_test, lab_test)

# Create PyTorch raw_data loaders from the datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Define the number of epochs
epochs = 40
name = 'eighth'


# Loop over the epochs
for epoch in range(epochs):

    # Initialize the running loss and accuracy
    running_loss = 0.0
    running_acc = 0.0

    # Loop over the batches
    for i, (inputs, labels) in enumerate(train_loader):

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update the running loss and accuracy
        running_loss += loss.item()
        running_acc += torch.sum(torch.argmax(outputs, axis=1) == torch.argmax(labels, axis=1)).item() / batch_size

        # Print the statistics every 200 batches
        if (i + 1) % 200 == 0:
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 200:.4f}, Accuracy: {running_acc / 200:.4f}')
            running_loss = 0.0
            running_acc = 0.0

    # Evaluate the model on the validation set
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_acc = 0.0
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += torch.sum(torch.argmax(outputs, axis=1) == torch.argmax(labels, axis=1)).item() / batch_size
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_acc / len(val_loader):.4f}')

    if epoch%10==0:
        torch.save(model.state_dict(), f'models/{name}_epoch{epoch}_{val_loss / len(val_loader):.4f}.pt')
    model.train()


torch.save(model.state_dict(), f'models/{name}.pt')
