import os
import sys
import torch
from torchvision import datasets, transforms
from config import args_parser
from IIoTmodel import DNN
from dataset.data import preprocess_dataset, split_dataset

file_path = '/Users/gautamjajoo/Desktop/FAL/dataset/Edge-IIoTset/DNN-EdgeIIoT-dataset.csv'

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(args.seed)

    # Add an option for choosing the dataset
    if args.dataset == "edgeiiot":
        df = preprocess_dataset(file_path)
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, seed=args.seed, size=args.size)

        # Print the shapes of the resulting datasets
        print("Training set shape:", X_train.shape)
        print("Validation set shape:", X_val.shape)
        print("Test set shape:", X_test.shape)

        # Add option if iid is true divide it into num_users
    else:
        exit('Error: unrecognized dataset')

if args.model == 'IIoTmodel':
    net_glob = DNN(args=args).to(args.device)

print(net_glob)
net_glob.train()



