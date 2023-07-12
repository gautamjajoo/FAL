import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dataset.dataSetSplit import DatasetSplit
import torch.optim as optim
from IIoTmodel import DNN
import numpy as np
from data.entropysampling import EntropySampling
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix


class DNNModel(object):
    def __init__(self, args, train_dataset, test_dataset, idxs, logger):

        self.args = args
        self.logger = logger
        self.train_loader = DataLoader(DatasetSplit(train_dataset, idxs), batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # size=sys.getsizeof(self.train_loader)
        # print("data_user", size)
        self.device = args.device
        # Default criterion set to NLL loss function
        # self.criterion = nn.NLLLoss()
        self.criterion = nn.CrossEntropyLoss()
        # for one hot encoding
        # self.criterion= nn.BCELoss()
        self.client_epochs = args.client_epochs
        self.net = DNN(args.input_features, args.num_classes, args.hidden_layers, args.hidden_nodes).to(args.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr)

        self.history = {'train_loss': [], 'test_loss': []}

    def train(self, model):
        mean_losses_superv = []
        # self.net.train()
        total = 0
        correct = 0
        for epoch in range(self.args.client_epochs):
            h = np.array([])

            for x, y in self.train_loader:
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
            # Save
            torch.save(self.net.state_dict(), path)
            return sum(mean_losses_superv) / len(mean_losses_superv), train_acc, self.net.state_dict()
            # print('Done.....')

    def test_inference(self, model, test_dataset):
        nb_classes = 15
        confusion_matrix = np.zeros((nb_classes, nb_classes))
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        output_list = torch.zeros(0, dtype=torch.long)
        target_list = torch.zeros(0, dtype=torch.long)
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.args.device), target.to(self.args.device)

                output = self.net(data.float())

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

            f1score = f1_score(target_list, output_list, average="macro",
                               zero_division=0)  # labels=np.unique(output_list))))
            precision = precision_score(target_list, output_list, average="macro", zero_division=0)
            recall = recall_score(target_list, output_list, average="macro", zero_division=0)
            class_report = classification_report(target_list, output_list, digits=4)

            # print(' F1 Score : ' + str(f1_score(target_list, output_list, average = "macro")))
            # #labels=np.unique(output_list))))
            # print(' Precision : '+str(precision_score(target_list, output_list,
            # average="macro", labels=np.unique(output_list))))
            # print(' Recall : '+str(recall_score(target_list, output_list, average="macro",
            # labels=np.unique(output_list))))
            # print("report", classification_report(target_list,output_list, digits=4))

            return acc, f1score, precision, recall, class_report, test_loss
