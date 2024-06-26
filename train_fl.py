import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from dataset.dataSetSplit import DatasetSplit
import torch.optim as optim
from IIoTmodel import DNN
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
from al_strategies.entropySampling import EntropySampler
from al_strategies.marginSampling import MarginSampler
from al_strategies.leastConfidence import LeastConfidenceSampler
from torch.utils.data import TensorDataset
import pandas as pd

class DNNModel(object):
    def __init__(self, args, train_dataset, test_dataset, idxs, model, logger):

        self.args = args
        self.logger = logger
        self.train_loader = DataLoader(DatasetSplit(train_dataset, idxs), 
                                       batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        self.batch_size = args.batch_size

        self.train_dataset = train_dataset
        self.idxs = idxs
        self.device = args.device

        self.criterion = nn.CrossEntropyLoss()
        self.client_epochs = args.client_epochs
        self.net = model
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr)

        self.history = {'train_loss': [], 'test_loss': []}

    def train(self, model):
        mean_losses_superv = []
        total = 0
        correct = 0
        for epoch in range(self.args.client_epochs):
            h = np.array([])

            for x, y, z in self.train_loader:
                self.optimizer.zero_grad()
                x = x.float()

                output = self.net(x)

                y = y.long()

                loss = self.criterion(output, y)

                h = np.append(h, loss.item())
                # raise

                # ===================backward====================
                loss.backward()
                self.optimizer.step()
                output = output.argmax(axis=1)

                total += y.size(0)

                y = y.float()
                output = output.float()

                correct += (output == y).sum().item()

            # raise
            # ===================log========================
            mean_loss_superv = np.mean(h)
            train_acc = correct / total

            mean_losses_superv.append(mean_loss_superv)

        path = "state_dict_model_IIoT_edge.pt"
        torch.save(self.net.state_dict(), path)

        return sum(mean_losses_superv) / len(mean_losses_superv), train_acc, self.net.state_dict()

    def test_inference(self, model, test_dataset):
        model.load_state_dict(torch.load("state_dict_model_IIoT_edge.pt"))
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        output_list = torch.zeros(0, dtype=torch.long)
        target_list = torch.zeros(0, dtype=torch.long)
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.args.device), target.to(self.args.device)

                output = model(data.float())

                batch_loss = self.criterion(output, target.long())
                # print("done... test...")
                # raise
                test_loss += batch_loss.item()
                total += target.size(0)

                target = target.float()

                output = output.argmax(axis=1)
                output = output.float()

                output_list = torch.cat([output_list, output.view(-1).long()])
                target_list = torch.cat([target_list, target.view(-1).long()])

                correct += (output == target).sum().item()

            test_loss /= total
            acc = correct / total

            f1score = f1_score(target_list, output_list, average="macro", zero_division=0)
            precision = precision_score(target_list, output_list, average="macro", zero_division=0)
            recall = recall_score(target_list, output_list, average="macro", zero_division=0)

            # Format the metrics to have six decimal places
            f1score = format(f1score, ".6f")
            precision = format(precision, ".6f")
            recall = format(recall, ".6f")
            class_report = classification_report(target_list, output_list, digits=4)

            return acc, f1score, precision, recall, class_report, test_loss
        
    def testglobal_inference(self, model, test_dataset):
        self.net = model
        self.net.eval()
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)
        test_loss = 0
        correct = 0
        total = 0
        output_list = torch.zeros(0, dtype=torch.long)
        target_list = torch.zeros(0, dtype=torch.long)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.args.device), target.to(self.args.device)

                output = model(data.float())

                batch_loss = self.criterion(output, target.long())
                # print("done... test...")
                # raise
                test_loss += batch_loss.item()
                total += target.size(0)

                target = target.float()

                output = output.argmax(axis=1)
                output = output.float()

                output_list = torch.cat([output_list, output.view(-1).long()])
                target_list = torch.cat([target_list, target.view(-1).long()])

                correct += (output == target).sum().item()

            test_loss /= total
            acc = correct / total

            f1score = f1_score(target_list, output_list, average="macro", zero_division=0)
            precision = precision_score(target_list, output_list, average="macro", zero_division=0)
            recall = recall_score(target_list, output_list, average="macro", zero_division=0)

            # Format the metrics to have six decimal places
            f1score = format(f1score, ".6f")
            precision = format(precision, ".6f")
            recall = format(recall, ".6f")
            class_report = classification_report(target_list, output_list, digits=4)

            return acc, f1score, precision, recall, class_report, test_loss
