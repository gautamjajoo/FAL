import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    category_columns = []
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
        category_columns.append(dummy_name)
    df.drop(name, axis=1, inplace=True)
    return df

def preprocess_dataset():
    data = pd.read_csv("/Users/gautamjajoo/Desktop/FAL/dataset/5G-NIDD/Combined.csv", low_memory=False)
    data.info()
    print(data['Attack Type'].value_counts())
    data.drop(columns=['Unnamed: 0'], inplace = True)
    data = data.drop_duplicates()
    data = shuffle(data)

    encode_text_dummy(data, 'Proto')
    encode_text_dummy(data, 'Cause')
    encode_text_dummy(data, 'State')
    encode_text_dummy(data, 'Label')
    encode_text_dummy(data, 'sDSb')
    encode_text_dummy(data, 'dDSb')
    encode_text_dummy(data, 'Attack Tool')

    attacks = {'Benign': 0, 'UDPFlood': 1, 'HTTPFlood': 2, 'SlowrateDoS': 3, 
        'TCPConnectScan': 4, 'SYNScan': 5, 'UDPScan': 6, 'SYNFlood': 7, 'ICMPFlood': 8}
    data['Attack Type'] = data['Attack Type'].map(attacks)
    
    print(data['Attack Type'].value_counts())
    data.info()
    return data

def split_dataset(df, seed, size, labeled_data_ratio):
    y = df['Attack Type']
    X = df.drop(['Attack Type'], axis=1)

    X_train, X_break, y_train, y_break = train_test_split(X, y, random_state=seed, test_size=size)
    X_test, X_val, y_test, y_val = train_test_split(X_break, y_break, random_state=seed+1, test_size=0.5)

    print("Train set size: ", len(X_train))
    print("Validation set size: ", len(X_val))
    print("Test set size: ", len(X_test))

    # Feature scaling using min-max scaling
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    labeled_size = int(len(X_train) * labeled_data_ratio)

    labeled_indices = np.random.choice(len(X_train), labeled_size, replace=False)
    X_labeled = X_train.iloc[labeled_indices]
    y_labeled = y_train.iloc[labeled_indices]

    X_train = X_train.drop(X_train.index[labeled_indices])
    y_train = y_train.drop(y_train.index[labeled_indices])

    print("Labeled set size: ", len(X_labeled))

    print("Train set size after labeled data: ", len(X_train))
    print("Validation set size after labeled data: ", len(X_val))
    print("Test set size after labeled data: ", len(X_test))
    print("Labeled set size after labeled data: ", len(X_labeled))  

    return X_train, X_val, X_test, X_labeled, y_train, y_val, y_test, y_labeled


# df = preprocess_dataset()
# print(df.info())
