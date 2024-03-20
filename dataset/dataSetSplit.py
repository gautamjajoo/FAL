import torch
from torch.utils.data import Dataset


# This class is a custom dataset that receives a base dataset and a list of indices.
# It enables getting only a subset of the data in the base dataset, as specified by the indices list.
# This is helpful in a federated learning scenario,
# where each client might only have access to a subset of the total data.
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        data, target = self.dataset[self.idxs[item]]
        original_idx = self.idxs[item]
        return data.clone().detach(), target.clone().detach(), original_idx

        # data = self.dataset[self.idxs[item]]
        # return torch.tensor(data), torch.tensor(data)
