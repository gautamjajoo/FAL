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
from torch.utils.data import TensorDataset
import pandas as pd



class DNNModel(object):
    def __init__(self, args, train_dataset, labeled_dataset, test_dataset, idxs, labeled_idxs, logger):

        self.args = args
        self.logger = logger
        self.train_loader = DataLoader(DatasetSplit(train_dataset, idxs), batch_size=args.batch_size, shuffle=True)
        self.labeled_loader = DataLoader(DatasetSplit(labeled_dataset, labeled_idxs), batch_size=args.batch_size,shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        self.num_samples = args.num_samples

        self.labeled_dataset = labeled_dataset
        self.labeled_idxs = labeled_idxs

        self.train_dataset = train_dataset
        self.idxs = idxs

        # print("idxs", idxs)
        # print("labeled_idxs", labeled_idxs)

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

        self.entropy_sampler = EntropySampler(self.net)

    def train(self, model, training_loader):
        mean_losses_superv = []
        # self.net.train()
        total = 0
        correct = 0
        for epoch in range(self.args.client_epochs):
            h = np.array([])

            for x, y in training_loader:
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

    def train_with_entropy_sampling(self, model):
        loss, train_acc, w = self.train(model, self.labeled_loader)

        unlabeled_probs = self.entropy_sampler.predict_probabilities(self.args, self.train_loader)
        num_samples = int(self.num_samples * self.train_loader.__len__())
        unlabeled_indices = self.entropy_sampler.sample(self.args, unlabeled_probs, num_samples)

        print("unlabeled_indices", unlabeled_indices)

        labeled_split_dataset = DatasetSplit(self.labeled_dataset, self.labeled_idxs)
        unlabeled_split_dataset = DatasetSplit(self.train_dataset, unlabeled_indices)

        combined_dataset = ConcatDataset([labeled_split_dataset, unlabeled_split_dataset])

        combined_loader = DataLoader(combined_dataset, batch_size=self.args.batch_size, shuffle=True)
        
        # labeled_data_loader = self.labeled_loader
        # unlabeled_data = Subset(self.train_loader.dataset, unlabeled_indices)
        # print("unlabeled_data", unlabeled_data)
        # unlabeled_data_loader = DataLoader(DatasetSplit(unlabeled_data, range(len(unlabeled_data))), batch_size=self.args.batch_size, shuffle=True)
        
        # combined_data_loader = ConcatDataset([labeled_data_loader, unlabeled_data_loader])
        # combined_loader = DataLoader(DatasetSplit(combined_data_loader, range(len(combined_data_loader))), batch_size=self.args.batch_size, shuffle=True)

        # labeled_dataset = combined_data_loader    
        # unlabeled_dataset = Subset(self.train_loader, [idx for idx in self. train_loader.dataset.idxs if idx not in unlabeled_indices])

        # X_train = combined_data.drop(['Attack_type'], axis=1)
        # y_train = combined_data['Attack_type']
        
        # X_train_tensor = torch.Tensor(X_train.values.astype(np.float32))
        # y_train_tensor = torch.LongTensor(y_train.values.astype(np.int64))

        # train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        # self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        

        # Train the `model` on the combined dataset
        loss, train_acc, w = self.train(model, combined_loader)

        # filtered_indices = [idx for idx in range(len(self.train_dataset)) if idx not in unlabeled_indices]
        # unlabeled_data = []
        # for idx in filtered_indices:
        #     if self.train_dataset[idx][0].dim() > 0 and self.train_dataset[idx][1].dim() > 0:
        #         unlabeled_data.append((self.train_dataset[idx][0], self.train_dataset[idx][1]))

        # unlabeled_data = TensorDataset(*[torch.stack((X_train, y_train)) for X_train, y_train in unlabeled_data])

        # combined_data = [combined_dataset[i] for i in range(len(combined_dataset))]
        # combined_data = list(zip(*combined_data))

        # combined_tensor_dataset = TensorDataset(*[torch.stack(samples) for samples in combined_data])

        return loss, train_acc, w, unlabeled_indices

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
