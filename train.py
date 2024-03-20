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
    def __init__(self, args, model, train_dataset, labeled_dataset, test_dataset, idxs, labeled_idxs, logger):

        self.args = args
        self.logger = logger
        self.train_loader = DataLoader(DatasetSplit(train_dataset, idxs), batch_size=args.batch_size, shuffle=True)
        self.labeled_loader = DataLoader(DatasetSplit(labeled_dataset, labeled_idxs), batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        self.num_samples = args.num_samples
        self.batch_size = args.batch_size

        print("Hello")
        print(len(self.train_loader))
        print(len(self.train_loader) * args.batch_size)

        self.labeled_dataset = labeled_dataset
        self.labeled_idxs = labeled_idxs

        self.train_dataset = train_dataset
        self.idxs = idxs

        self.labeled_loader2 = DataLoader(DatasetSplit(labeled_dataset, labeled_idxs), shuffle=True)
        self.subset_loader = DataLoader(DatasetSplit(train_dataset, idxs), shuffle=True)

        print("train_dataset", len(self.subset_loader))
        print("labeled_dataset", len(self.labeled_loader2))

        # size=sys.getsizeof(self.train_loader)
        # print("data_user", size)
        self.device = args.device
        # Default criterion set to NLL loss function
        # self.criterion = nn.NLLLoss()
        self.criterion = nn.CrossEntropyLoss()
        # for one hot encoding
        # self.criterion= nn.BCELoss()
        self.client_epochs = args.client_epochs
        self.net = model
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr)

        self.history = {'train_loss': [], 'test_loss': []}

    def train(self, model, training_loader):
        mean_losses_superv = []
        # self.net.train()
        total = 0
        correct = 0
        for epoch in range(self.args.client_epochs):
            h = np.array([])

            for x, y, z in training_loader:
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
            # print('Done.....')

    def train_with_sampling(self, model):
        print("length of labeled dataset before AL", len(self.labeled_loader) * self.batch_size)
        print("length of training dataset before AL", len(self.train_loader) * self.batch_size)

        num_samples = int(self.num_samples * len(self.subset_loader))

        print("length of num_samples", num_samples)

        if(self.args.al_method == "entropysampling"):
            self.entropy_sampler = EntropySampler(self.net)
            unlabeled_indices = self.entropy_sampler.sample(self.args, self.subset_loader, num_samples)

        elif(self.args.al_method == "marginsampling"):
            self.margin_sampler = MarginSampler(self.net)
            unlabeled_indices = self.margin_sampler.sample(self.args, self.subset_loader, num_samples)

        elif(self.args.al_method == "randomsampling"):
            self.random_sampler = RandomSampler(self.net)
            unlabeled_indices = self.random_sampler.sample(self.args, self.subset_loader, num_samples)        

        else:
            self.least_confidence_sampler = LeastConfidenceSampler(self.net)
            unlabeled_indices = self.least_confidence_sampler.sample(self.args, self.subset_loader, num_samples)

        # Get dataset consisting of labeled_idxs of the labeled dataset
        labeled_split_dataset = DatasetSplit(self.labeled_dataset, self.labeled_idxs)
        
        # Get dataset for unlabeled_idxs of the training dataset
        unlabeled_split_dataset = DatasetSplit(self.train_dataset, unlabeled_indices)

        combined_dataset = ConcatDataset([labeled_split_dataset, unlabeled_split_dataset])
        combined_loader = DataLoader(combined_dataset, batch_size=self.args.batch_size, shuffle=True)
        
        # Train the model on the combined dataset
        loss, train_acc, w = self.train(self.net, combined_loader)

        return loss, train_acc, w, unlabeled_indices

    def test_inference(self, model, test_dataset):
        nb_classes = 15
        confusion_matrix = np.zeros((nb_classes, nb_classes))
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
        
    def testglobal_inference(self, model, val_dataset):
        self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)
        self.net = model
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        output_list = torch.zeros(0, dtype=torch.long)
        target_list = torch.zeros(0, dtype=torch.long)
        with torch.no_grad():
            for data, target in self.val_loader:
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

            # print(' F1 Score : ' + str(f1_score(target_list, output_list, average = "macro")))
            # #labels=np.unique(output_list))))
            # print(' Precision : '+str(precision_score(target_list, output_list,
            # average="macro", labels=np.unique(output_list))))
            # print(' Recall : '+str(recall_score(target_list, output_list, average="macro",
            # labels=np.unique(output_list))))
            # print("report", classification_report(target_list,output_list, digits=4))

            return acc, f1score, precision, recall, class_report, test_loss

