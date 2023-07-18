import torch
import torch.nn as nn

class EntropySampler:
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
    
