import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dataset.dataSetSplit import DatasetSplit
import torch.optim as optim
from IIoTmodel import DNN
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix


class GlobalModel(object):
    def __init__(self, args, labeled_dataset, test_dataset):
        self.args = args
        self.device = args.device
        self.net = DNN(args.input_features, args.num_classes, args.hidden_layers, args.hidden_nodes).to(args.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        self.history = {'train_loss': [], 'test_loss': []}

    def train(self):
        for epoch in range(self.args.num_epochs):
            self.net.train()
            total_loss = 0
            correct = 0
            total = 0

            for x, y in self.train_loader:
                x = x.float().to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

            train_loss = total_loss / len(self.train_loader)
            train_accuracy = 100.0 * correct / total
            self.history['train_loss'].append(train_loss)

            print(f"Epoch [{epoch + 1}/{self.args.num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        print("Training complete!")

        return train_loss, train_accuracy

    def test_inference(self, model):
        nb_classes = 15
        confusion_matrix = np.zeros((nb_classes, nb_classes))
        model.eval()
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
        class_report = classification_report(target_list, output_list, digits=4)

        return acc, f1score, precision, recall, class_report, test_loss
