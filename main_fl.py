import torch
from config import args_parser
from IIoTmodel import DNN
from dataset.data import preprocess_dataset
from torch.utils.data import TensorDataset
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
from train_fl import DNNModel
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

    # Define the function to plot the best and worst client in each round
    def plot_best_worst_clients(best_clients_list, worst_clients_list):
        plt.figure(figsize=(10, 6))
        rounds = len(best_clients_list)
        rounds_list = list(range(1, rounds + 1))
        plt.plot(rounds_list, best_clients_list, 'go-', label='Best Client')
        plt.plot(rounds_list, worst_clients_list, 'ro-', label='Worst Client')
        
        # Add labels for the best client points
        for round_num, accuracy in zip(rounds_list, best_clients_list):
            plt.annotate(f'{accuracy:.4f}', (round_num, accuracy), textcoords="offset points", xytext=(0,10), ha='center')
        
        # Add labels for the worst client points
        for round_num, accuracy in zip(rounds_list, worst_clients_list):
            plt.annotate(f'{accuracy:.4f}', (round_num, accuracy), textcoords="offset points", xytext=(0,-20), ha='center')
        
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
            plt.annotate(formatted_value, (round_num, float(formatted_value)), textcoords="offset points", xytext=(0, 10), ha='center')

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
            plt.annotate(formatted_value, (round_num, acc_value), textcoords="offset points", xytext=(0, 10), ha='center')

        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title('Global Accuracy after each Round')
        plt.legend()
        plt.grid()
        plt.show()

    def split_dataset(df, seed, size):
        y = df['Attack_type']
        X = df.drop(['Attack_type'], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=0.2)

        print("Train set size: ", len(X_train))
        print("Test set size: ", len(X_test))

        # Feature scaling using min-max scaling
        scaler = MinMaxScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        d = {1: 10000, 3: 10000, 12: 10000}

        smote = SMOTE(sampling_strategy = d, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        unique_classes = [1, 3, 12]
        # Calculate the counts and percentages of each class in y_train
        class_sample_counts = [np.sum(y_train == cls) for cls in unique_classes]
        class_sample_percentages = [(count / len(y_train)) * 100 for count in class_sample_counts]

        for cls, count, percentage in zip(unique_classes, class_sample_counts, class_sample_percentages):
            print(f"Class {cls}: {count} samples ({percentage:.2f}%)")
  
        return X_train, X_test, y_train, y_test
    
    # Function to get the dataset
    def get_dataset(args):

        df = preprocess_dataset(file_path)
        # df = pd.read_csv(preprocessed_file_path, low_memory=False)
        
        num_classes = df['Attack_type'].nunique()
        input_features = df.drop(['Attack_type'], axis=1).shape[1]

        print("Number of classes:", num_classes)
        print("Number of input features:", input_features)

        X_train, X_test, y_train, y_test = split_dataset(df, seed=args.seed, size=args.size)

        # Print the shapes of the resulting datasets
        print("Training set shape:", X_train.shape)
        print("Test set shape:", X_test.shape)

        X_train_tensor = torch.Tensor(X_train.values.astype(np.float32))
        y_train_tensor = torch.LongTensor(y_train.values.astype(np.int64))

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        X_test_tensor = torch.Tensor(X_test.values.astype(np.float32))
        y_test_tensor = torch.LongTensor(y_test.values.astype(np.int64))

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        if(args.iid == 1):
            user_groups = split_iid(train_dataset, args.num_users)
        
        # print("user_group", user_groups)
        print("Done...")

        return train_dataset, test_dataset, user_groups

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
    # Add an option for choosing the dataset
    if args.dataset == "edgeiiot":
        logger = SummaryWriter('../logs')
        train_dataset, test_dataset, user_groups = get_dataset(args)

    else:
        exit('Error: unrecognized dataset')

    if args.model == 'IIoTmodel':
        DNN_model = DNN(args.input_features, args.num_classes, args.hidden_layers, args.hidden_nodes)
        print(DNN_model)

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

            user_groups = split_iid(train_dataset, args.num_users)

            # global_model.train(auto_encoder_model)
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            print("idxs_users", idxs_users)
            client_test_accuracy_per_round = [[] for _ in range(max(idxs_users) + 1)]

            for idx in idxs_users:
                local_model = copy.deepcopy(DNN_model)
                DNN_client = DNNModel(args=args, train_dataset=train_dataset, 
                                    test_dataset=test_dataset, idxs=user_groups[idx], model=local_model,
                                    logger=logger)

                loss, train_acc, w = DNN_client.train(model=local_model)
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

            global_acc, F1_score, Precision, Recall, class_report, test_loss = DNN_client.testglobal_inference(DNN_model, test_dataset)
            print(f' \nAvg Training Stats after {rounds+1} global rounds:')
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

            best_client_idx = np.argmax(np.array(client_test_accuracy))
            worst_client_idx = np.argmin(np.array(client_test_accuracy))
            best_clients_list.append(client_test_accuracy[best_client_idx])
            worst_clients_list.append(client_test_accuracy[worst_client_idx])
        
        plot_best_worst_clients(best_clients_list, worst_clients_list)
        
        # Plot Test Accuracy for each client across rounds
        rounds_list = range(1, args.rounds + 1)
        plot_global_accuracy_per_round(rounds_list, global_accuracy_per_round)
        plot_metric_per_round(rounds_list, global_F1_score_per_round, 'F1 Score', 'Global F1 Score after each Round', 'Global F1 Score')
        plot_metric_per_round(rounds_list, global_Precision_per_round, 'Precision', 'Global Precision after each Round', 'Global Precision')
        plot_metric_per_round(rounds_list, global_Recall_per_round, 'Recall', 'Global Recall after each Round', 'Global Recall')

        print(best_clients_list)
        print(worst_clients_list)
