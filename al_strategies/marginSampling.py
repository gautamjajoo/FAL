import torch
import torch.nn as nn
from al_strategies.predictProbability import PredictProbability

class MarginSampler:
    def __init__(self, model):
        self.model = model
        self.predict_probability = PredictProbability(self.model)

    def sample(self, args, train_loader, num_samples):
        print("I AM IN MARGIN SAMPLING")
        probabilities = self.predict_probability.predict_probabilities(args, train_loader)
        margins = self.calculate_margins(probabilities)
        selected_indices = self.select_indices(margins, num_samples)
        return selected_indices

    def calculate_margins(self, probabilities):
        margins = []
        for prob in probabilities:
            max_prob = torch.max(torch.tensor(prob))
            second_max_prob = torch.max(torch.tensor(prob), dim=0).values
            margin = max_prob - second_max_prob
            margins.append(margin.mean().item())
        return margins

    def select_indices(self, margins, num_samples):
        sorted_indices = sorted(range(len(margins)), key=lambda k: margins[k], reverse=True)
        selected_indices = sorted_indices[:num_samples]
        return selected_indices
