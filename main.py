import torch
from config import args_parser
from IIoTmodel import DNN
from dataset.data import preprocess_dataset, split_dataset
from torch.utils.data import TensorDataset
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
from train import DNNModel
import copy
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from al_strategies.entropySampling import EntropySampler
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, Dataset
from dataset.dataSetSplit import DatasetSplit
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

file_path = '/Users/gautamjajoo/Desktop/FAL/dataset/Edge-IIoTset/DNN-EdgeIIoT-dataset.csv'
preprocessed_file_path = '/Users/gautamjajoo/Desktop/FAL/preprocessed_DNN.csv'

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(args.seed)

    # Splitting the dataset into num_users parts
    def split_iid(dataset, num_users):
        """
        Splits a given dataset into `num_users` number of disjoint subsets, where each subset has an equal number of samples.

        Args:
            dataset (list): The dataset to be split.
            num_users (int): The number of disjoint subsets to split the dataset into.

        Returns:
            dict: A dictionary where the keys are integers representing the user IDs and the values are sets of indices
            representing the samples allocated to each user.
        """
        num_items = int(len(dataset) // num_users)  # the number of allocated samples for each client
        print("num_items", num_items)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        return dict_users

    def update_datasets(train_dataset, labeled_dataset, unlabeled_indices):
        """
        Updates the labeled and unlabeled datasets by removing and adding labeled data.

        Args:
            train_dataset (TensorDataset): The original training dataset.
            labeled_dataset (TensorDataset): The labeled dataset.
            unlabeled_indices (list): A list of indices of the labeled data to remove from the training dataset
                and add to the labeled dataset.

        Returns:
            TensorDataset, TensorDataset: Updated training dataset and labeled dataset.
        """
        X_train = train_dataset.tensors[0]
        y_train = train_dataset.tensors[1]
        X_labeled = labeled_dataset.tensors[0]
        y_labeled = labeled_dataset.tensors[1]

        print("No. of additions")
        print(len(unlabeled_indices))

        seen = set()
        duplicates = set()

        for num in unlabeled_indices:
            if num in seen:
                duplicates.add(num)
            else:
                seen.add(num)
        
        print("Number of unique indices")
        print(len(seen))

        # Get the set of unique indices to be updated in the labeled dataset
        unique_unlabeled_indices = list(set(unlabeled_indices))

        # Remove labeled data from the training dataset and add it to the labeled dataset
        X_labeled_additions = X_train[unique_unlabeled_indices]
        y_labeled_additions = y_train[unique_unlabeled_indices]

        # Update the labeled dataset
        X_labeled_updated = torch.cat((X_labeled, X_labeled_additions), dim=0)
        y_labeled_updated = torch.cat((y_labeled, y_labeled_additions), dim=0)

        print("No. of labeled indices")
        print(len(X_labeled_updated))

        # Remove the labeled data from the training dataset using boolean indexing
        mask = torch.ones(len(X_train), dtype=torch.bool)
        mask[unlabeled_indices] = 0
        X_train_updated = X_train[mask]
        y_train_updated = y_train[mask]

        print("No. of trained indices")
        print(len(X_train_updated))

        # Create updated datasets
        updated_train_dataset = TensorDataset(X_train_updated, y_train_updated)
        updated_labeled_dataset = TensorDataset(X_labeled_updated, y_labeled_updated)

        return updated_train_dataset, updated_labeled_dataset

    # Define the function to plot the best and worst client in each round
    def plot_best_worst_clients(best_clients_list, worst_clients_list):
        plt.figure(figsize=(10, 6))
        rounds = len(best_clients_list)
        rounds_list = list(range(1, rounds + 1))
        plt.plot(rounds_list, best_clients_list, 'go-', label='Best Client')
        plt.plot(rounds_list, worst_clients_list, 'ro-', label='Worst Client')
        
        # Add labels for the best client points
        for round_num, accuracy in zip(rounds_list, best_clients_list):
            plt.annotate(f'{accuracy:.4f}', (round_num, accuracy), textcoords="offset points", xytext=(0,10), ha='center', rotation = 60, fontsize = 6)
        
        # Add labels for the worst client points
        for round_num, accuracy in zip(rounds_list, worst_clients_list):
            plt.annotate(f'{accuracy:.4f}', (round_num, accuracy), textcoords="offset points", xytext=(0,-20), ha='center', rotation = 60, fontsize = 6)
        
        plt.xlabel('Round')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy of Best and Worst Clients in Each Round')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_metric_per_round(rounds_list, metric_values, ylabel, title, label):
        plt.figure(figsize=(10, 6))
        plt.plot(rounds_list, metric_values, marker='o', label=label)

        for round_num, metric_value in zip(rounds_list, metric_values):
            formatted_value = f'{metric_value:.6f}'
            plt.annotate(formatted_value, (round_num, float(formatted_value)), textcoords="offset points", xytext=(0, 10), ha='center', rotation = 60, fontsize = 6)

        plt.xlabel('Round')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()

    def plot_global_accuracy_per_round(rounds_list, accuracy_values):
        plt.figure(figsize=(10, 6))
        plt.plot(rounds_list, accuracy_values, marker='o', label='Global Accuracy')

        for round_num, acc_value in zip(rounds_list, accuracy_values):
            formatted_value = f'{acc_value:.6f}'
            plt.annotate(formatted_value, (round_num, acc_value), textcoords="offset points", xytext=(0, 10), ha='center', rotation = 60, fontsize = 6)

        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title('Global Accuracy after each Round')
        plt.legend()
        plt.grid()
        plt.show()

    # Function to get the dataset
    def get_dataset(args):

        df = preprocess_dataset(file_path)
        # df = pd.read_csv(preprocessed_file_path, low_memory=False)
        
        num_classes = df['Attack_type'].nunique()
        input_features = df.drop(['Attack_type'], axis=1).shape[1]

        print("Number of classes:", num_classes)
        print("Number of input features:", input_features)

        X_train, X_val, X_test, X_labeled, y_train, y_val, y_test, y_labeled = \
            split_dataset(df, seed=args.seed, size=args.size, labeled_data_ratio=args.labeled_data_ratio)

        # Print the shapes of the resulting datasets
        print("Training set shape:", X_train.shape)
        print("Labeled set shape:", X_labeled.shape)
        print("Validation set shape:", X_val.shape)
        print("Test set shape:", X_test.shape)

        X_train_tensor = torch.Tensor(X_train.values.astype(np.float32))
        y_train_tensor = torch.LongTensor(y_train.values.astype(np.int64))

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        X_labeled_tensor = torch.Tensor(X_labeled.values.astype(np.float32))
        y_labeled_tensor = torch.LongTensor(y_labeled.values.astype(np.int64))

        labeled_dataset = TensorDataset(X_labeled_tensor, y_labeled_tensor)

        X_test_tensor = torch.Tensor(X_test.values.astype(np.float32))
        y_test_tensor = torch.LongTensor(y_test.values.astype(np.int64))

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        X_val_tensor = torch.Tensor(X_val.values.astype(np.float32))
        y_val_tensor = torch.LongTensor(y_val.values.astype(np.int64))

        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        if(args.iid == 1):
            user_groups = split_iid(train_dataset, args.num_users)
            labeled_groups = split_iid(labeled_dataset, args.num_users)
        
        # print("user_group", user_groups)
        print("Done...")

        return train_dataset, test_dataset, labeled_dataset, val_dataset, user_groups, labeled_groups

    # Function to average the weights
    def average_weights(w):
        """
        Returns the average of the weights.
        """
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg
    
    def fedprox(local_models, global_model, rho):
        # Compute local updates
        local_updates = []
        for local_model in local_models:
            local_updates.append(local_model)

        # Compute global update
        global_update = copy.deepcopy(global_model)
        for key in global_update.keys():
            for local_update in local_updates:
                global_update[key] += local_update[key]
            global_update[key] = torch.div(global_update[key], len(local_updates))

        # Add regularization term
        for key in global_update.keys():
            global_update[key] += rho * (global_update[key] - global_model[key])

        # Update global model
        return global_update

    global_accuracy_list_entropy = []
    global_accuracy_list_margin = []
    global_accuracy_list_random= []
    global_accuracy_list_least_confidence = []   

    for i in range(4):
        if i == 0:
            args.al_method = "entropysampling"
        elif i == 1:
            args.al_method = "marginsampling"
        elif i == 2:
            args.al_method = "randomsampling"
        else:
            args.al_method = "leastconfidence"

        # Add an option for choosing the dataset
        if args.dataset == "edgeiiot":
            logger = SummaryWriter('../logs')
            # load data
            train_dataset, test_dataset, labeled_dataset, val_dataset, user_groups, labeled_groups = get_dataset(args)

        else:
            exit('Error: unrecognized dataset')

        if args.model == 'IIoTmodel':
            DNN_model = DNN(args.input_features, args.num_classes, args.hidden_layers, args.hidden_nodes)
            print(DNN_model)

            unlabeled_indices = []
            num_labeled_samples_list = []

            global_accuracy_per_round = []
            global_F1_score_per_round = []
            global_Precision_per_round = []
            global_Recall_per_round = []

            best_clients_list = []
            worst_clients_list = []

            for rounds in tqdm(range(args.rounds)):
                # in the server
                local_weights, local_losses = [], []
                client_test_accuracy = []

                print(f'\n | Training Round : {rounds + 1} |\n')

                # global_model.train(auto_encoder_model)
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                print("idxs_users", idxs_users)
                client_test_accuracy_per_round = [[] for _ in range(max(idxs_users) + 1)]

                if rounds > 0:
                    print("After round ")
                    print(rounds)

                    train_dataset, labeled_dataset = update_datasets(train_dataset, labeled_dataset, unlabeled_indices)
                    unlabeled_indices = []

                    # Splitting the train and labeled dataset into user groups
                    user_groups = split_iid(train_dataset, args.num_users)
                    labeled_groups = split_iid(labeled_dataset, args.num_users)
                
                # This statement is to get the labeled_dataset after each round,
                # this would be common to all the strategies

                num_labeled_samples_list.append(len(labeled_dataset))

                print("train_dataset", len(train_dataset))
                print("labeled_dataset", len(labeled_dataset))

                for idx in idxs_users:
                    local_model = copy.deepcopy(DNN_model)
                    DNN_client = DNNModel(args=args, model = local_model, train_dataset=train_dataset, labeled_dataset =labeled_dataset, 
                                        test_dataset=test_dataset, idxs=user_groups[idx], 
                                        labeled_idxs = labeled_groups[idx],
                                        logger=logger)

                    loss, train_acc, w, client_labeled_indices = DNN_client.train_with_sampling(model=local_model)

                    # Collecting the labeled indices from all the clients
                    unlabeled_indices += client_labeled_indices
                    local_weights.append(copy.deepcopy(w))

                    test_acc, F1_score, Precision, Recall, class_report, test_loss = DNN_client.test_inference(local_model,
                                                                                                            test_dataset)
                    print(f'client_id {idx}')
                    print("|---- Test Accuracy_client: {:.6f}%".format(test_acc))
                    print("|---- F1_score:", F1_score)
                    print("|---- Precision:", Precision)
                    print("|---- Recall:", Recall)
                    print(class_report)
                    print(f'Testing Loss : {np.mean(np.array(test_loss))}')

                    client_test_accuracy.append(test_acc)
                    client_test_accuracy_per_round[idx].append(test_acc)
                
                # This updates the global model
                if (args.fl_algo == "fedavg"):
                    DNN_model.load_state_dict(average_weights(local_weights))
                elif(args.fl_algo == "fedprox"):
                    global_dnn = DNN_model.state_dict()
                    DNN_model.load_state_dict(fedprox(local_weights, global_dnn, args.rho))

                global_acc, F1_score, Precision, Recall, class_report, test_loss = DNN_client.testglobal_inference(DNN_model, val_dataset)
                print("|---- Global Model Accuracy: {:.6f}%".format(global_acc))
                print("|---- F1_score:", F1_score)
                print("|---- Precision:", Precision)
                print("|---- Recall:", Recall)
                print(class_report)
                print(f'Testing Loss : {np.mean(np.array(test_loss))}')

                global_accuracy_per_round.append(global_acc)
                global_F1_score_per_round.append(float(F1_score))
                global_Precision_per_round.append(float(Precision))
                global_Recall_per_round.append(float(Recall))

                if args.al_method == "entropysampling":
                    global_accuracy_list_entropy.append(global_acc)
                    print(global_accuracy_list_entropy)

                elif args.al_method == "marginsampling":
                    global_accuracy_list_margin.append(global_acc)
                    print(global_accuracy_list_margin)

                elif args.al_method == "randomsampling":
                    global_accuracy_list_random.append(global_acc)
                    print(global_accuracy_list_random)
                    
                else:
                    global_accuracy_list_least_confidence.append(global_acc)
                    print(global_accuracy_list_least_confidence)

                print(num_labeled_samples_list)

                best_client_idx = np.argmax(np.array(client_test_accuracy))
                worst_client_idx = np.argmin(np.array(client_test_accuracy))
                best_clients_list.append(client_test_accuracy[best_client_idx])
                worst_clients_list.append(client_test_accuracy[worst_client_idx])
            
            plot_best_worst_clients(best_clients_list, worst_clients_list)
            
            # Plot Test Accuracy for each client across rounds
            rounds_list = range(1, args.rounds + 1)
            plot_global_accuracy_per_round(rounds_list, global_accuracy_per_round)
            # plot_metric_per_round(rounds_list, global_accuracy_per_round, 'Accuracy', 'Global Accuracy after each Round', 'Global Accuracy')
            plot_metric_per_round(rounds_list, global_F1_score_per_round, 'F1 Score', 'Global F1 Score after each Round', 'Global F1 Score')
            plot_metric_per_round(rounds_list, global_Precision_per_round, 'Precision', 'Global Precision after each Round', 'Global Precision')
            plot_metric_per_round(rounds_list, global_Recall_per_round, 'Recall', 'Global Recall after each Round', 'Global Recall')

            print(best_clients_list)
            print(worst_clients_list)
            if args.al_method == "entropysampling":
                print(global_accuracy_list_entropy)

            elif args.al_method == "marginsampling":
                print(global_accuracy_list_margin)

            elif args.al_method == "randomsampling":
                print(global_accuracy_list_random)
                
            else:
                print(global_accuracy_list_least_confidence)

            print(global_F1_score_per_round)
            print(global_Precision_per_round)
            print(global_Recall_per_round)
            print(num_labeled_samples_list)

    print(global_accuracy_list_entropy)
    print(global_accuracy_list_margin)
    print(global_accuracy_list_random)
    print(global_accuracy_list_least_confidence)
    print(num_labeled_samples_list)

    plt.figure(figsize=(10, 6))
    plt.plot(num_labeled_samples_list, global_accuracy_list_entropy, label='Entropy Sampling')
    plt.plot(num_labeled_samples_list, global_accuracy_list_margin, label='Margin Sampling')
    plt.plot(num_labeled_samples_list, global_accuracy_list_random, label='Random Sampling')
    plt.plot(num_labeled_samples_list, global_accuracy_list_least_confidence, label='Least Confidence Sampling')
    plt.xlabel('Number of Labeled Samples')
    plt.ylabel('Global Accuracy')
    plt.title('Global Accuracy vs Number of Labeled Samples')
    plt.legend()
    plt.show()

