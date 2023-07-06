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

file_path = '/Users/gautamjajoo/Desktop/FAL/dataset/Edge-IIoTset/DNN-EdgeIIoT-dataset.csv'

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(args.seed)

    def split_iid(dataset, num_users):
        num_items = int(len(dataset) / num_users)  # the number of allocated samples for each client
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        return dict_users

    def get_dataset(args):

        df = preprocess_dataset(file_path)

        num_classes = df['Attack_type'].nunique()
        input_features = df.drop(['Attack_type'], axis=1).shape[1]

        print("Number of classes:", num_classes)
        print("Number of input features:", input_features)
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, seed=args.seed, size=args.size)

        scaler = MinMaxScaler()
        scaled_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        scaled_X_val = pd.DataFrame(scaler.fit_transform(X_val), columns=X_val.columns)
        scaled_X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)

        # X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, seed=1, size=0.2)


        # Print the shapes of the resulting datasets
        print("Training set shape:", X_train.shape)
        print("Validation set shape:", X_val.shape)
        print("Test set shape:", X_test.shape)

        X_train_tensor = torch.Tensor(scaled_X_train.values.astype(np.float32))
        y_train_tensor = torch.LongTensor(y_train.values.astype(np.int64))

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        X_test_tensor = torch.Tensor(scaled_X_test.values.astype(np.float32))
        y_test_tensor = torch.LongTensor(y_test.values.astype(np.int64))

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        user_groups = split_iid(train_dataset, args.num_users)
        # print("user_group", user_groups)
        print("Done...")

        return train_dataset, test_dataset, user_groups


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

    # Add an option for choosing the dataset
    if args.dataset == "edgeiiot":
        logger = SummaryWriter('../logs')
        # load data
        train_dataset, test_dataset, user_groups = get_dataset(args)

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

        for rounds in tqdm(range(args.rounds)):
            # in the server
            local_weights, local_losses = [], []
            print(f'\n | Training Round : {rounds + 1} |\n')

            # global_model.train(auto_encoder_model)
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            print("idxs_users", idxs_users)

            for idx in idxs_users:
                DNN_client = DNNModel(args=args, train_dataset=train_dataset, test_dataset=test_dataset,
                                      idxs=user_groups[idx], logger=logger)
                loss, train_acc, w = DNN_client.train(model=copy.deepcopy(DNN_model))
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
            # print(local_weights)
            DNN_model.load_state_dict(average_weights(local_weights))
