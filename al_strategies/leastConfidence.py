import torch
import torch.nn as nn
from al_strategies.predictProbability import PredictProbability

class LeastConfidenceSampler:
    def __init__(self, model):
        self.model = model
        self.predict_probability = PredictProbability(self.model)

    def sample(self, args, train_loader, num_samples):
        print("I AM IN LEAST CONFIDENCE")
        probabilities = self.predict_probability.predict_probabilities(args, train_loader)
        confidences = self.calculate_confidences(probabilities)
        selected_indices = self.select_indices(confidences, num_samples)
        
        # Get the original training dataset indices using DatasetSplit
        original_indices = []
        for idx in selected_indices:
            _, _, original_idx = train_loader.dataset[idx]
            original_indices.append(original_idx)

        return original_indices

    def calculate_confidences(self, probabilities):
        confidences = []
        for prob in probabilities:
            max_prob = torch.max(torch.tensor(prob))
            confidence = 1 - max_prob
            confidences.append(confidence.item())
        return confidences

    def select_indices(self, confidences, num_samples):
        sorted_indices = sorted(range(len(confidences)), key=lambda k: confidences[k], reverse=True)
        selected_indices = sorted_indices[:num_samples]
        return selected_indices
