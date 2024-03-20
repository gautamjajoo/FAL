import random

class RandomSampler:
    def __init__(self, model):
        self.model = model

    def sample(self, args, train_loader, num_samples):
        print("I am in Random Sampling")
        num_total_samples = len(train_loader.dataset)
        
        selected_indices = random.sample(range(num_total_samples), num_samples)

        original_indices = []
        for idx in selected_indices:
            _, _, original_idx = train_loader.dataset[idx]
            original_indices.append(original_idx)

        return original_indices
