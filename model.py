# Import PyTorch and other libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the number of items in the embedding
num_items = 4

# Define the embedding dimension
embedding_dim = 8


# Create a custom model class that inherits from nn.Module
class SequenceModel(nn.Module):
    def __init__(self, device):
        super(SequenceModel, self).__init__()

        self.device = device

        # Define an embedding layer with 4 items and 8 dimensions
        self.seq_embedding = nn.Embedding(num_items, embedding_dim)
        self.pos_embedding = nn.Embedding(200, 1)

        # Define convolutional layers with leaky relu activation
        self.conv1 = nn.Conv1d(8, 16, 3)
        self.conv2 = nn.Conv1d(16, 32, 3)
        self.conv3 = nn.Conv1d(32, 16, 3)
        self.conv4 = nn.Conv1d(16, 8, 3)
        self.conv5 = nn.Conv1d(8, 4, 3)

        # Define the negative slope for leaky relu
        self.alpha = 0.1

        # Define the output size after flattening the embedding output
        self.output_size = 760

        # Define dense layers with leaky relu activation
        self.fc1 = nn.Linear(self.output_size, 128)
        self.fc2 = nn.Linear(128, 64)

        # Define a final dense layer with sigmoid activation for binary output classification
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # Pass the input through the embedding layer
        x = self.seq_embedding(x)
        # x = x + self.pos_embedding(torch.arange(0, 200).to(self.device))

        # Permute the dimensions of x to match the expected input of conv1d
        x = x.permute(0, 2, 1)

        # Pass the input through the convolutional layers and apply leaky relu
        x = F.leaky_relu(self.conv1(x), self.alpha)
        x = F.leaky_relu(self.conv2(x), self.alpha)
        x = F.leaky_relu(self.conv3(x), self.alpha)
        x = F.leaky_relu(self.conv4(x), self.alpha)
        x = self.conv5(x)

        # Flatten the output of the convolutional layers
        x = x.view(-1, self.output_size)

        # Pass the input through the dense layers and apply leaky relu
        x = F.leaky_relu(self.fc1(x), self.alpha)
        x = F.leaky_relu(self.fc2(x), self.alpha)

        # Pass the input through the final dense layer and apply sigmoid
        x = self.fc3(x)

        x = F.sigmoid(x)

        return x
