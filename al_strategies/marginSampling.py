import torch
import torch.nn as nn

class MarginSampler:
    def __init__(self, model):
        self.model = model

    def predict_probabilities(self, args, dataloader):
        device = next(self.model.parameters()).device
        probabilities = []

        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                probabilities.append(nn.functional.softmax(outputs, dim=1).squeeze().tolist())

        return probabilities

    def sample(self, args, probabilities, num_samples):
        margins = self.calculate_margins(probabilities)
        selected_indices = self.select_indices(margins, num_samples)
        return selected_indices

    def calculate_margins(self, probabilities):
        margins = []
        for prob in probabilities:
            max_prob = torch.max(torch.tensor(prob))
            second_max_prob = torch.max(torch.tensor(prob), dim=0).values
            margin = max_prob - second_max_prob
            margins.append(margin.item())
        return margins

    def select_indices(self, margins, num_samples):
        sorted_indices = sorted(range(len(margins)), key=lambda k: margins[k], reverse=True)
        selected_indices = sorted_indices[:num_samples]
        return selected_indices
