# Import PyTorch and other libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the embedding dimension
embedding_dim = 16


# Create a custom model class that inherits from nn.Module
class SequenceModel(nn.Module):
    def __init__(self, device):
        super(SequenceModel, self).__init__()

        self.device = device

        # Define an embedding layer with 4 items and 8 dimensions
        self.base_embedding = nn.Embedding(4, embedding_dim)
        self.pos_embedding = nn.Embedding(200, embedding_dim)

        # Define convolutional layers with leaky relu activation
        self.attention1 = nn.MultiheadAttention(embedding_dim, 8, batch_first=True)
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)


        self.conv1 = nn.Conv1d(embedding_dim, 32, 3)
        self.conv2 = nn.Conv1d(32, 16, 3)
        self.conv3 = nn.Conv1d(16, 8, 3)
        self.conv4 = nn.Conv1d(8, 4, 3)

        # Define the negative slope for leaky relu
        self.alpha = 0.1

        # Define the output size after flattening the embedding output
        self.output_size = 768

        # Define dense layers with leaky relu activation
        self.fc1 = nn.Linear(self.output_size, 128)
        self.fc2 = nn.Linear(128, 64)

        # Define a final dense layer with sigmoid activation for binary output classification
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # Pass the input through the embedding layer
        embeddings = self.base_embedding(x)

        embeddings = embeddings + self.pos_embedding(torch.arange(0, 200).to(self.device))

        queries = self.query(embeddings)
        keys = self.key(embeddings)
        values = self.value(embeddings)

        x, weights = self.attention1(queries, keys, values)

        x = x.permute(0, 2, 1)

        # Pass the input through the convolutional layers and apply leaky relu
        x = F.leaky_relu(self.conv1(x), self.alpha)
        x = F.leaky_relu(self.conv2(x), self.alpha)
        x = F.leaky_relu(self.conv3(x), self.alpha)
        x = F.leaky_relu(self.conv4(x), self.alpha)

        # Flatten the output of the convolutional layers
        x = x.view(-1, self.output_size)

        # Pass the input through the dense layers and apply leaky relu
        x = F.leaky_relu(self.fc1(x), self.alpha)
        x = F.leaky_relu(self.fc2(x), self.alpha)

        # Pass the input through the final dense layer and apply sigmoid
        x = self.fc3(x)

        return x


