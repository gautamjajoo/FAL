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


file_path = 'C:/Users/perera/Desktop/FAL/FAL/dataset/Edge-IIoTset/DNN-EdgeIIoT-dataset.csv'

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(args.seed)

    # Splitting the dataset into num_users parts
    def split_iid(dataset, num_users):
        num_items = int(len(dataset) / num_users)  # the number of allocated samples for each client
        print("num_items", num_items)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random .choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        return dict_users

    # Removing the labeled idxs from the training dataset
    def remove_labeled_data(dataset, idxs):
        X_train_list = dataset.tensors[0].tolist()
        y_train_list = dataset.tensors[1].tolist()

        filtered_X_train = []
        filtered_y_train = []

        for index in range(len(X_train_list)):
            if index not in idxs:
                filtered_X_train.append(X_train_list[index])
                filtered_y_train.append(y_train_list[index])

        filtered_X_train_tensor = torch.tensor(filtered_X_train)
        filtered_y_train_tensor = torch.tensor(filtered_y_train)

        filtered_train_dataset = TensorDataset(filtered_X_train_tensor, filtered_y_train_tensor)

        return filtered_train_dataset
    
    # Adding the labeled idxs to the labeled dataset
    def add_labeled_data(labeled_dataset, train_dataset, idxs):
        X_labeled_list = labeled_dataset.tensors[0].tolist()
        y_labeled_list = labeled_dataset.tensors[1].tolist()

        X_train_list = train_dataset.tensors[0].tolist()
        y_train_list = train_dataset.tensors[1].tolist()

        for index in idxs:
            X_labeled_list.append(X_train_list[index])
            y_labeled_list.append(y_train_list[index])

        filtered_X_labeled_tensor = torch.tensor(X_labeled_list)
        filtered_y_labeled_tensor = torch.tensor(y_labeled_list)

        filtered_labeled_dataset = TensorDataset(filtered_X_labeled_tensor, filtered_y_labeled_tensor)

        return filtered_labeled_dataset
    

    # Define the function to plot the best and worst client in each round
    def plot_best_worst_clients(best_clients_list, worst_clients_list):
        plt.figure(figsize=(10, 6))
        rounds = len(best_clients_list)
        rounds_list = list(range(1, rounds + 1))
        plt.plot(rounds_list, best_clients_list, 'go-', label='Best Client')
        plt.plot(rounds_list, worst_clients_list, 'ro-', label='Worst Client')
        plt.xlabel('Round')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy of Best and Worst Clients in Each Round')
        plt.legend()
        plt.grid()
        plt.show()

    # Function to get the dataset
    def get_dataset(args):

        df = preprocess_dataset(file_path)

        num_classes = df['Attack_type'].nunique()
        input_features = df.drop(['Attack_type'], axis=1).shape[1]

        print("Number of classes:", num_classes)
        print("Number of input features:", input_features)

        X_train, X_val, X_test, X_labeled, y_train, y_val, y_test, y_labeled = \
            split_dataset(df, seed=args.seed, size=args.size, labeled_data_ratio=args.labeled_data_ratio)

        scaler = MinMaxScaler()
        scaled_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        scaled_X_labeled = pd.DataFrame(scaler.fit_transform(X_labeled), columns=X_train.columns)
        scaled_X_val = pd.DataFrame(scaler.fit_transform(X_val), columns=X_val.columns)
        scaled_X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)

        # X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, seed=1, size=0.2)

        # Print the shapes of the resulting datasets
        print("Training set shape:", X_train.shape)
        print("Labeled set shape:", X_labeled.shape)
        print("Validation set shape:", X_val.shape)
        print("Test set shape:", X_test.shape)

        X_train_tensor = torch.Tensor(scaled_X_train.values.astype(np.float32))
        y_train_tensor = torch.LongTensor(y_train.values.astype(np.int64))

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        X_labeled_tensor = torch.Tensor(scaled_X_labeled.values.astype(np.float32))
        y_labeled_tensor = torch.LongTensor(y_labeled.values.astype(np.int64))

        labeled_dataset = TensorDataset(X_labeled_tensor, y_labeled_tensor)

        X_test_tensor = torch.Tensor(scaled_X_test.values.astype(np.float32))
        y_test_tensor = torch.LongTensor(y_test.values.astype(np.int64))

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        X_val_tensor = torch.Tensor(scaled_X_val.values.astype(np.float32))
        y_val_tensor = torch.LongTensor(y_val.values.astype(np.int64))

        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

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

    num_labeled_samples_list = []
    global_accuracy_list_entropy = []
    global_accuracy_list_margin = []
    global_accuracy_list_least_confidence = []   

    for i in range(3):
        if i == 0:
            args.al_method = "entropysampling"
        elif i == 1:
            args.al_method = "marginsampling"
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
            # Training
            train_loss, train_accuracy = [], []
            val_acc_list, net_list = [], []
            cv_loss, cv_acc = [], []
            print_every = 2
            val_loss_pre, counter = 0, 0
            unlabeled_indices = []

            global_accuracy_per_round = []
            global_F1_score_per_round = []
            global_Precision_per_round = []

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

                    # Removing the active learning labeled indices from the train dataset
                    combined_train_dataset = remove_labeled_data(train_dataset, unlabeled_indices)
                    
                    # Adding active learning labeled indices again to the labeled dataset
                    combined_labeled_dataset = add_labeled_data(labeled_dataset, train_dataset, unlabeled_indices)
                    
                    train_dataset = combined_train_dataset
                    labeled_dataset = combined_labeled_dataset

                    # Splitting the train and labeled dataset into user groups
                    user_groups = split_iid(train_dataset, args.num_users)
                    labeled_groups = split_iid(labeled_dataset, args.num_users)

                if(i == 0):
                    num_labeled_samples_list.append(len(labeled_dataset))


                print("train_dataset", len(train_dataset))
                print("labeled_dataset", len(labeled_dataset))

                for idx in idxs_users:

                    DNN_client = DNNModel(args=args, train_dataset=train_dataset, labeled_dataset =labeled_dataset, 
                                        test_dataset=test_dataset, idxs=user_groups[idx], 
                                        labeled_idxs = labeled_groups[idx],
                                        logger=logger)

                    loss, train_acc, w, client_labeled_indices = DNN_client.train_with_sampling(DNN_model)

                    # Collecting the labeled indices from all the clients
                    unlabeled_indices += client_labeled_indices

                    # print("unlabeled_indices", unlabeled_indices)                
                    # print(w)
                    # print("w", w)
                    # print("loss", loss)
                    local_weights.append(copy.deepcopy(w))
                    # local_losses.append(copy.deepcopy(loss))

                    test_acc, F1_score, Precision, Recall, class_report, test_loss = DNN_client.test_inference(DNN_model,
                                                                                                            test_dataset)
                    print(f'client_id {idx}')
                    print("|---- Test Accuracy_client: {:.2f}%".format(test_acc))
                    print("|---- F1_score:", F1_score)
                    print("|---- Precision:", Precision)
                    print("|---- Recall:", Recall)
                    print(class_report)
                    print(f'Testing Loss : {np.mean(np.array(test_loss))}')

                    # client_test_accuracy.append(test_acc)
                    # client_F1_score.append(F1_score)
                    # client_precision.append(Precision)
                    # client_recall.append(Recall)
                    # client_testing_loss.append(np.mean(np.array(test_loss)))
                    client_test_accuracy.append(test_acc)
                    client_test_accuracy_per_round[idx].append(test_acc)
                # print(local_weights)
                # This updates the global model
                DNN_model.load_state_dict(average_weights(local_weights))

                global_acc, F1_score, Precision, Recall, class_report, test_loss = DNN_client.test_inference(DNN_model, val_dataset)
                print("|---- Global Model Accuracy: {:.2f}%".format(global_acc))
                print("|---- F1_score:", F1_score)
                print("|---- Precision:", Precision)
                print("|---- Recall:", Recall)
                print(class_report)
                print(f'Testing Loss : {np.mean(np.array(test_loss))}')

                global_accuracy_per_round.append(global_acc)
                global_F1_score_per_round.append(F1_score)
                global_Precision_per_round.append(Precision)

                if args.al_method == "entropysampling":
                    global_accuracy_list_entropy.append(global_acc)

                elif args.al_method == "marginsampling":
                    global_accuracy_list_margin.append(global_acc)
                    
                else:
                    global_accuracy_list_least_confidence.append(global_acc)

                print(global_accuracy_list_entropy)
                print(global_accuracy_list_margin)
                print(global_accuracy_list_least_confidence)
                print(num_labeled_samples_list)

                best_client_idx = np.argmax(np.array(client_test_accuracy))
                worst_client_idx = np.argmin(np.array(client_test_accuracy))
                best_clients_list.append(client_test_accuracy[best_client_idx])
                worst_clients_list.append(client_test_accuracy[worst_client_idx])
            
            plot_best_worst_clients(best_clients_list, worst_clients_list)
            
            # Plot Test Accuracy for each client across rounds
            plt.figure(figsize=(10, 6))
            rounds_list = range(1, args.rounds + 1)
            plt.plot(rounds_list, global_accuracy_per_round, marker='o', label='Global Accuracy')
            plt.xlabel('Round')
            plt.ylabel('Accuracy')
            plt.title('Global Accuracy after each Round')
            plt.legend()
            plt.show()

            plt.figure(figsize=(10, 6))
            rounds_list = range(1, args.rounds + 1)
            plt.plot(rounds_list, global_F1_score_per_round, marker='o', label='Global F1 score')
            plt.xlabel('Round')
            plt.ylabel('F1 Score')
            plt.title('Global F1 Score after each Round')
            plt.legend()
            plt.show()

            plt.figure(figsize=(10, 6))
            rounds_list = range(1, args.rounds + 1)
            plt.plot(rounds_list, global_Precision_per_round, marker='o', label='Global Precision')
            plt.xlabel('Round')
            plt.ylabel('Precision')
            plt.title('Global Precision after each Round')
            plt.legend()
            plt.show()

            print(global_accuracy_list_entropy)
            print(global_accuracy_list_margin)
            print(global_accuracy_list_least_confidence)
            print(num_labeled_samples_list)

    print(global_accuracy_list_entropy)
    print(global_accuracy_list_margin)
    print(global_accuracy_list_least_confidence)
    print(num_labeled_samples_list)

    plt.figure(figsize=(10, 6))
    plt.plot(num_labeled_samples_list, global_accuracy_list_entropy, label='Entropy Sampling')
    plt.plot(num_labeled_samples_list, global_accuracy_list_margin, label='Margin Sampling')
    plt.plot(num_labeled_samples_list, global_accuracy_list_least_confidence, label='Least Confidence Sampling')
    plt.xlabel('Number of Labeled Samples')
    plt.ylabel('Global Accuracy')
    plt.title('Global Accuracy vs Number of Labeled Samples')
    plt.legend()
    plt.show()

