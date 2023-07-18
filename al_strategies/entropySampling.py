import torch
import torch.nn as nn
from al_strategies.predictProbability import PredictProbability

class EntropySampler:
    def __init__(self, model):
        self.model = model
        self.predict_probability = PredictProbability(self.model)

    def sample(self, args, train_loader, num_samples):
        probabilities = self.predict_probability.predict_probabilities(args, train_loader)
        entropies = self.calculate_entropies(probabilities)
        selected_indices = self.select_indices(entropies, num_samples)
        return selected_indices

    def calculate_entropies(self, probabilities):
        entropies = []
        for prob in probabilities:
            entropy = -torch.sum(torch.tensor(prob) * torch.log2(torch.tensor(prob) + 1e-10))
            entropies.append(entropy.item())
        return entropies

    def select_indices(self, entropies, num_samples):
        sorted_indices = sorted(range(len(entropies)), key=lambda k: entropies[k], reverse=True)
        selected_indices = sorted_indices[:num_samples]
        return selected_indices
    
