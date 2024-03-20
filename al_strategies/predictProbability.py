import torch
import torch.nn as nn

class PredictProbability:
    def __init__(self, model):
        self.model = model

    def predict_probabilities(self, args, dataloader):
        device = next(self.model.parameters()).device
        probabilities = []

        with torch.no_grad():
            for inputs, _, _ in dataloader:
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                probabilities.append(nn.functional.softmax(outputs, dim=1).squeeze().tolist())

        return probabilities