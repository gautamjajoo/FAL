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
from al_strategies.randomSampling import RandomSampler
from torch.utils.data import TensorDataset
import pandas as pd


class DNNModel(object):
    def __init__(self, args, model, X_train, y_train, X_test, y_test):

        self.args = args
        self.num_samples = args.num_samples
        self.batch_size = args.batch_size
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()
        self.client_epochs = args.client_epochs
        self.net = model
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr)
        self.history = {'train_loss': [], 'test_loss': []}
        print(X_train[:10])
        print(y_train[:10])

        X_train_tensor = torch.Tensor(X_train.values.astype(np.float32))
        y_train_tensor = torch.LongTensor(y_train.values.astype(np.int64))
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        X_test_tensor = torch.Tensor(X_test.values.astype(np.float32))
        y_test_tensor = torch.LongTensor(y_test.values.astype(np.int64))
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # Create DataLoader using the TensorDataset
        self.training_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_dataset, shuffle=True, batch_size=self.batch_size)

    def train(self):
        mean_losses_superv = []
        # self.net.train()
        total = 0
        correct = 0
        for epoch in range(self.args.client_epochs):
            h = np.array([])

            for x, y in self.training_loader:
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

        path = "state_dict_model_IIoT_edge_classical.pt"
        torch.save(self.net.state_dict(), path)
        
        return sum(mean_losses_superv) / len(mean_losses_superv), train_acc, self.net.state_dict()
            # print('Done.....')

    def test_inference(self, model):
        nb_classes = 15
        confusion_matrix = np.zeros((nb_classes, nb_classes))
        model.load_state_dict(torch.load("state_dict_model_IIoT_edge_classical.pt"))
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

            f1score = f1_score(target_list, output_list, average="macro")  # labels=np.unique(output_list))))
            precision = precision_score(target_list, output_list, average="macro")
            recall = recall_score(target_list, output_list, average="macro")

            # Format the metrics to have six decimal places
            f1score = format(f1score, ".6f")
            precision = format(precision, ".6f")
            recall = format(recall, ".6f")

            class_report = classification_report(target_list, output_list, digits=4)

            # print(' F1 Score : ' + str(f1_score(target_list, output_list, average = "macro")))
            # #labels=np.unique(output_list))))
            # print(' Precision : '+str(precision_score(target_list, output_list,
            # average="macro", labels=np.unique(output_list))))
            # print(' Recall : '+str(recall_score(target_list, output_list, average="macro",
            # labels=np.unique(output_list))))
            # print("report", classification_report(target_list,output_list, digits=4))

            return acc, f1score, precision, recall, class_report, test_loss
        