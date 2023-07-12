import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # basic arguments
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--size', type=int, default=0.2, help='size (0.2)')

    # federated learning arguments
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--rounds', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_classes', type=int, default=15, help="number of classes")
    parser.add_argument('--num_users', type=int, default=15, help="number of users")
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients")
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size")
    parser.add_argument('--test_bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="momentum")
    parser.add_argument('--weight_decay', type=float, default=0.00001, help="weight decay (default: 0.00001)")
    parser.add_argument('--lr_decay', type=float, default=0.1, help="learning rate decay ratio")
    parser.add_argument('--fl_algo', type=str, default='fedavg', help='federated learning algorithm')

    # dataset arguments
    parser.add_argument('--data_dir', type=str, default='./data', help='data path')
    parser.add_argument('--iid', type=int, default=1, help='iid data')
    parser.add_argument('--labeled_data_ratio', type=float, default=0.2, help='split ratio:labeled & unlabeled data')
    parser.add_argument('--dataset', type=str, default='edgeiiot', help="name of dataset")
    parser.add_argument('--partition', type=str, default="dir_balance", help="methods for Non-IID")
    parser.add_argument('--num_classes_per_user', type=int, default=15, help="classes per user")
    parser.add_argument('--input_features', type=int, default=78, help="number of input features")


    # model arguments
    parser.add_argument('--model', type=str, default='IIoTmodel', help='model name')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--client_epochs', type=int, default=3, help="number of client epochs")
    parser.add_argument('--server_epochs', type=int, default=10, help="number of server epochs")
    parser.add_argument('--hidden_layers', type=int, default=2, help="number of hidden layers")
    parser.add_argument('--hidden_nodes', type=int, default=90, help="number of hidden nodes")

    # active learning arguments
    parser.add_argument('--label_percent', type=float, default=0.1, help="percentage of labeled dataset")
    parser.add_argument('--query_ratio', type=float, default=0.05, help="ratio of data examples per one query")
    parser.add_argument('--end_ratio', type=float, default=0.0, help="ratio for stopping query")
    parser.add_argument('--al_method', type=str, default=None)

    args = parser.parse_args()

    return args
