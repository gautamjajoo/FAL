import torch
from torch import nn
from torchviz import make_dot

# Define your DNN model
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

        return x

# Create an instance of your DNN model
input_features = 78
num_classes = 15
hidden_layers = 2
hidden_nodes = 90
model = DNN(input_features, num_classes, hidden_layers, hidden_nodes)

# Visualize the model architecture
x = torch.randn(1, input_features)
visual_graph = make_dot(model(x), params=dict(model.named_parameters()))
visual_graph.view()
