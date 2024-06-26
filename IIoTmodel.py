import torch
from torch import nn
from torch.optim import Adam


class DNN(nn.Module):
    def __init__(self, input_features, num_classes, hidden_layers, hidden_nodes):
        super(DNN, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(nn.Linear(input_features, hidden_nodes))
        self.layers.append(nn.ReLU())

        # hidden layers
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_nodes, hidden_nodes))
            self.layers.append(nn.ReLU())

        # output layer
        self.layers.append(nn.Linear(hidden_nodes, num_classes))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        # Applying the softmax function to the output layer
        # x = nn.functional.softmax(x, dim=1)
        return x


# Instantiate the model
# model = DNN(input_features, num_classes, hidden_layers, hidden_nodes)

# Define the optimizer
# optimizer = Adam(model.parameters())
