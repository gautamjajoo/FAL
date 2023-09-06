import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def fixDataType(df_dataset):
    df_dataset = df_dataset[df_dataset['Dst Port'] != 'Dst Port']
    df_dataset['Dst Port'] = df_dataset['Dst Port'].astype(int)
    df_dataset['Protocol'] = df_dataset['Protocol'].astype(int)
    df_dataset['Flow Duration'] = df_dataset['Flow Duration'].astype(int)
    df_dataset['Tot Fwd Pkts'] = df_dataset['Tot Fwd Pkts'].astype(int)
    df_dataset['Tot Bwd Pkts'] = df_dataset['Tot Bwd Pkts'].astype(int)
    df_dataset['TotLen Fwd Pkts'] = df_dataset['TotLen Fwd Pkts'].astype(int)
    df_dataset['TotLen Bwd Pkts'] = df_dataset['TotLen Bwd Pkts'].astype(int)
    df_dataset['Fwd Pkt Len Max'] = df_dataset['Fwd Pkt Len Max'].astype(int)
    df_dataset['Fwd Pkt Len Min'] = df_dataset['Fwd Pkt Len Min'].astype(int)
    df_dataset['Fwd Pkt Len Mean'] = df_dataset['Fwd Pkt Len Mean'].astype(float)
    df_dataset['Fwd Pkt Len Std'] = df_dataset['Fwd Pkt Len Std'].astype(float)
    df_dataset['Bwd Pkt Len Max'] = df_dataset['Bwd Pkt Len Max'].astype(int)
    df_dataset['Bwd Pkt Len Min'] = df_dataset['Bwd Pkt Len Min'].astype(int)
    df_dataset['Bwd Pkt Len Mean'] = df_dataset['Bwd Pkt Len Mean'].astype(float)
    df_dataset['Bwd Pkt Len Std'] = df_dataset['Bwd Pkt Len Std'].astype(float)
    df_dataset['Flow Byts/s'] = df_dataset['Flow Byts/s'].astype(float)
    df_dataset['Flow Pkts/s'] = df_dataset['Flow Pkts/s'].astype(float)
    df_dataset['Flow IAT Mean'] = df_dataset['Flow IAT Mean'].astype(float)
    df_dataset['Flow IAT Std'] = df_dataset['Flow IAT Std'].astype(float)
    df_dataset['Flow IAT Max'] = df_dataset['Flow IAT Max'].astype(int)
    df_dataset['Flow IAT Min'] = df_dataset['Flow IAT Min'].astype(int)
    df_dataset['Fwd IAT Tot'] = df_dataset['Fwd IAT Tot'].astype(int)
    df_dataset['Fwd IAT Mean'] = df_dataset['Fwd IAT Mean'].astype(float)
    df_dataset['Fwd IAT Std'] = df_dataset['Fwd IAT Std'].astype(float)
    df_dataset['Fwd IAT Max'] = df_dataset['Fwd IAT Max'].astype(int)
    df_dataset['Fwd IAT Min'] = df_dataset['Fwd IAT Min'].astype(int)
    df_dataset['Bwd IAT Tot'] = df_dataset['Bwd IAT Tot'].astype(int)
    df_dataset['Bwd IAT Mean'] = df_dataset['Bwd IAT Mean'].astype(float)
    df_dataset['Bwd IAT Std'] = df_dataset['Bwd IAT Std'].astype(float)
    df_dataset['Bwd IAT Max'] = df_dataset['Bwd IAT Max'].astype(int)
    df_dataset['Bwd IAT Min'] = df_dataset['Bwd IAT Min'].astype(int)
    df_dataset['Fwd PSH Flags'] = df_dataset['Fwd PSH Flags'].astype(int)
    df_dataset['Bwd PSH Flags'] = df_dataset['Bwd PSH Flags'].astype(int)
    df_dataset['Fwd URG Flags'] = df_dataset['Fwd URG Flags'].astype(int)
    df_dataset['Bwd URG Flags'] = df_dataset['Bwd URG Flags'].astype(int)
    df_dataset['Fwd Header Len'] = df_dataset['Fwd Header Len'].astype(int)
    df_dataset['Bwd Header Len'] = df_dataset['Bwd Header Len'].astype(int)
    df_dataset['Fwd Pkts/s'] = df_dataset['Fwd Pkts/s'].astype(float)
    df_dataset['Bwd Pkts/s'] = df_dataset['Bwd Pkts/s'].astype(float)
    df_dataset['Pkt Len Min'] = df_dataset['Pkt Len Min'].astype(int)
    df_dataset['Pkt Len Max'] = df_dataset['Pkt Len Max'].astype(int)
    df_dataset['Pkt Len Mean'] = df_dataset['Pkt Len Mean'].astype(float)
    df_dataset['Pkt Len Std'] = df_dataset['Pkt Len Std'].astype(float)
    df_dataset['Pkt Len Var'] = df_dataset['Pkt Len Var'].astype(float)
    df_dataset['FIN Flag Cnt'] = df_dataset['FIN Flag Cnt'].astype(int)
    df_dataset['SYN Flag Cnt'] = df_dataset['SYN Flag Cnt'].astype(int)
    df_dataset['RST Flag Cnt'] = df_dataset['RST Flag Cnt'].astype(int)
    df_dataset['PSH Flag Cnt'] = df_dataset['PSH Flag Cnt'].astype(int)
    df_dataset['ACK Flag Cnt'] = df_dataset['ACK Flag Cnt'].astype(int)
    df_dataset['URG Flag Cnt'] = df_dataset['URG Flag Cnt'].astype(int)
    df_dataset['CWE Flag Count'] = df_dataset['CWE Flag Count'].astype(int)
    df_dataset['ECE Flag Cnt'] = df_dataset['ECE Flag Cnt'].astype(int)
    df_dataset['Down/Up Ratio'] = df_dataset['Down/Up Ratio'].astype(int)
    df_dataset['Pkt Size Avg'] = df_dataset['Pkt Size Avg'].astype(float)
    df_dataset['Fwd Seg Size Avg'] = df_dataset['Fwd Seg Size Avg'].astype(float)
    df_dataset['Bwd Seg Size Avg'] = df_dataset['Bwd Seg Size Avg'].astype(float)
    df_dataset['Fwd Byts/b Avg'] = df_dataset['Fwd Byts/b Avg'].astype(int)
    df_dataset['Fwd Pkts/b Avg'] = df_dataset['Fwd Pkts/b Avg'].astype(int)
    df_dataset['Fwd Blk Rate Avg'] = df_dataset['Fwd Blk Rate Avg'].astype(int)
    df_dataset['Bwd Byts/b Avg'] = df_dataset['Bwd Byts/b Avg'].astype(int)
    df_dataset['Bwd Pkts/b Avg'] = df_dataset['Bwd Pkts/b Avg'].astype(int)
    df_dataset['Bwd Blk Rate Avg'] = df_dataset['Bwd Blk Rate Avg'].astype(int)
    df_dataset['Subflow Fwd Pkts'] = df_dataset['Subflow Fwd Pkts'].astype(int)
    df_dataset['Subflow Fwd Byts'] = df_dataset['Subflow Fwd Byts'].astype(int)
    df_dataset['Subflow Bwd Pkts'] = df_dataset['Subflow Bwd Pkts'].astype(int)
    df_dataset['Subflow Bwd Byts'] = df_dataset['Subflow Bwd Byts'].astype(int)
    df_dataset['Init Fwd Win Byts'] = df_dataset['Init Fwd Win Byts'].astype(int)
    df_dataset['Init Bwd Win Byts'] = df_dataset['Init Bwd Win Byts'].astype(int)
    df_dataset['Fwd Act Data Pkts'] = df_dataset['Fwd Act Data Pkts'].astype(int)
    df_dataset['Fwd Seg Size Min'] = df_dataset['Fwd Seg Size Min'].astype(int)
    df_dataset['Active Mean'] = df_dataset['Active Mean'].astype(float)
    df_dataset['Active Std'] = df_dataset['Active Std'].astype(float)
    df_dataset['Active Max'] = df_dataset['Active Max'].astype(int)
    df_dataset['Active Min'] = df_dataset['Active Min'].astype(int)
    df_dataset['Idle Mean'] = df_dataset['Idle Mean'].astype(float)
    df_dataset['Idle Std'] = df_dataset['Idle Std'].astype(float)
    df_dataset['Idle Max'] = df_dataset['Idle Max'].astype(int)
    df_dataset['Idle Min'] = df_dataset['Idle Min'].astype(int)
    
    return df_dataset


def preprocess_dataset():
    data1 = pd.read_csv("/Users/perera/Desktop/FAL/FAL/dataset/CSE-CIC-IDS2018/02-28-2018.csv", low_memory=False)
    data2 = pd.read_csv("/Users/perera/Desktop/FAL/FAL/dataset/CSE-CIC-IDS2018/03-01-2018.csv", low_memory=False)
    data3 = pd.read_csv("/Users/perera/Desktop/FAL/FAL/dataset/CSE-CIC-IDS2018/02-16-2018.csv", low_memory=False)
    data4 = pd.read_csv("/Users/perera/Desktop/FAL/FAL/dataset/CSE-CIC-IDS2018/02-15-2018.csv", low_memory=False)
    data5 = pd.read_csv("/Users/perera/Desktop/FAL/FAL/dataset/CSE-CIC-IDS2018/02-21-2018.csv", low_memory=False)
    data6 = pd.read_csv("/Users/perera/Desktop/FAL/FAL/dataset/CSE-CIC-IDS2018/03-02-2018.csv", low_memory=False)
    data7 = pd.read_csv("/Users/perera/Desktop/FAL/FAL/dataset/CSE-CIC-IDS2018/02-22-2018.csv", low_memory=False)
    data8 = pd.read_csv("/Users/perera/Desktop/FAL/FAL/dataset/CSE-CIC-IDS2018/02-20-2018.csv", low_memory=False)
    data9 = pd.read_csv("/Users/perera/Desktop/FAL/FAL/dataset/CSE-CIC-IDS2018/02-14-2018.csv", low_memory=False)
    data10 = pd.read_csv("/Users/perera/Desktop/FAL/FAL/dataset/CSE-CIC-IDS2018/02-23-2018.csv", low_memory=False)

    data8.drop(columns=['Flow ID', 'Src IP', 'Src Port', 'Dst IP'], inplace=True)

    data = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10], ignore_index=True)

    print(data.info())

    data.drop(columns=['Timestamp'], inplace = True)
    data = data.sample(frac = 0.25, random_state = 1)

    # replace +ve and -ve infinity with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace = True)

    data = fixDataType(data)
    data = shuffle(data)
    print(data.info())

    mask = data['Label'] != 'Label'
    df_dataset = data.drop(data[~mask].index)

    print(data.duplicated().sum())
    data.drop_duplicates(inplace = True)
    print(data.duplicated().sum())

    attacks = {'Benign': 0, 'DDOS attack-HOIC': 1, 'DDoS attacks-LOIC-HTTP': 2, 'DoS attacks-Hulk': 3, 
            'Bot': 4, 'FTP-BruteForce': 5,
                'SSH-Bruteforce': 6, 'Infilteration': 7, 'DoS attacks-SlowHTTPTest': 8, 
                'DoS attacks-GoldenEye': 9,
                'DoS attacks-Slowloris': 10, 'DDOS attack-LOIC-UDP': 11, 
                'Brute Force -Web': 12, 'Brute Force -XSS': 13, 'SQL Injection': 14}
    data['Label'] = data['Label'].map(attacks)

    return data

def split_dataset(df, seed, size, labeled_data_ratio):
    y = df['Attack_type']
    X = df.drop(['Attack_type'], axis=1)

    X_train, X_break, y_train, y_break = train_test_split(X, y, random_state=seed, test_size=size)
    X_test, X_val, y_test, y_val = train_test_split(X_break, y_break, random_state=seed+1, test_size=0.5)

    print("Train set size: ", len(X_train))
    print("Validation set size: ", len(X_val))
    print("Test set size: ", len(X_test))

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


df = preprocess_dataset()
print(df.info())
