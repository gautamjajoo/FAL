from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

class EntropySampler:
    def __init__(self):
        pass

    def calculate_entropy(self, predictions):
        probabilities = predictions.softmax(dim=1)
        log_probabilities = torch.log2(probabilities)
        entropy = -torch.sum(probabilities * log_probabilities, dim=1)
        return entropy

    def sample(self, args, model, unlabeled_dataset, num_samples):
        model.eval()
        entropy_scores = []

        # Iterate over the unlabeled dataset and calculate entropy for each sample
        for data, _ in unlabeled_dataset:
            data = data.to(args.device)
            predictions = model(data)
            print("HELLO")
            print(predictions.shape)
            entropy = self.calculate_entropy(predictions)
            entropy_scores.append(entropy.item())

        # Sort the samples based on entropy scores and select the top-k samples
        indices = np.argsort(entropy_scores)
        selected_indices = indices[-num_samples:]

        # Return the selected samples from the unlabeled dataset
        selected_samples = [unlabeled_dataset[i] for i in selected_indices]

        return selected_samples
    
