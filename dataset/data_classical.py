import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
sns.set()
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
from numpy import where
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def load_dataset(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    return df


def drop_columns(df, columns):
    df = df.drop(columns, axis=1)
    return df


def drop_missing_values(df):
    df = df.dropna(axis=0, how='any')
    return df


def drop_duplicates(df):
    df = df.drop_duplicates(subset=None, keep="first")
    return df


def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    category_columns = []
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
        category_columns.append(dummy_name)
    df.drop(name, axis=1, inplace=True)
    return df


def preprocess_dataset(file_path):
    # print(file_path)
    df = load_dataset(file_path)
    print(df['Attack_type'].value_counts())
    df.info()

    df4 = df[['frame.time', 'ip.src_host', 'ip.dst_host', 'arp.dst.proto_ipv4',
              'arp.opcode', 'arp.hw.size', 'arp.src.proto_ipv4', 'icmp.checksum',
              'icmp.seq_le', 'icmp.transmit_timestamp', 'icmp.unused',
              'http.file_data', 'http.content_length', 'http.request.uri.query',
              'http.request.method', 'http.referer', 'http.request.full_uri',
              'http.request.version', 'http.response', 'http.tls_port', 'tcp.ack',
              'tcp.ack_raw', 'tcp.checksum', 'tcp.connection.fin',
              'tcp.connection.rst', 'tcp.connection.syn', 'tcp.connection.synack',
              'tcp.dstport', 'tcp.flags', 'tcp.flags.ack', 'tcp.len', 'tcp.options',
              'tcp.payload', 'tcp.seq', 'tcp.srcport', 'udp.port', 'udp.stream',
              'udp.time_delta', 'dns.qry.name', 'dns.qry.name.len', 'dns.qry.qu',
              'dns.qry.type', 'dns.retransmission', 'dns.retransmit_request',
              'dns.retransmit_request_in', 'mqtt.conack.flags',
              'mqtt.conflag.cleansess', 'mqtt.conflags', 'mqtt.hdrflags', 'mqtt.len',
              'mqtt.msg_decoded_as', 'mqtt.msg', 'mqtt.msgtype', 'mqtt.proto_len',
              'mqtt.protoname', 'mqtt.topic', 'mqtt.topic_len', 'mqtt.ver',
              'mbtcp.len', 'mbtcp.trans_id', 'mbtcp.unit_id',
              'Attack_type']]

    columns_to_drop = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
                       "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp", "http.request.uri.query",
                       "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg"]

    # drop unnecessary flow features columns
    df = drop_columns(df4, columns_to_drop)

    # feat = ['http.request.method-PUT',
    #         'http.referer-TESTING_PURPOSES_ONLY',
    #         'http.request.version-> HTTP/1.1',
    #         'http.request.version-By Dr HTTP/1.1',
    #         'dns.qry.name.len-_googlecast._tcp.local',
    #         'dns.qry.name.len-null-null.local',
    #         'mqtt.conack.flags-1461073',
    #         'mqtt.conack.flags-1461074',
    #         'mqtt.conack.flags-1461383',
    #         'mqtt.conack.flags-1461384',
    #         'mqtt.conack.flags-1461589',
    #         'mqtt.conack.flags-1461591',
    #         'mqtt.conack.flags-1471198',
    #         'mqtt.conack.flags-1471199',
    #         'mqtt.conack.flags-1574358',
    #         'mqtt.conack.flags-1574359']

    # df = df.drop(feat, axis=1)

    # replace INF values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # drop rows with NaN values
    df = drop_missing_values(df)

    # drop duplicate rows
    df = drop_duplicates(df)
    
    df = shuffle(df)

    print(df.info())
    

    encode_text_dummy(df, 'http.request.method')
    encode_text_dummy(df, 'http.referer')
    encode_text_dummy(df, "http.request.version")
    encode_text_dummy(df, "dns.qry.name.len")
    encode_text_dummy(df, "mqtt.conack.flags")
    encode_text_dummy(df, "mqtt.protoname")
    encode_text_dummy(df, "mqtt.topic")

    attacks = {'Normal': 0, 'MITM': 1, 'Uploading': 2, 'Ransomware': 3, 'SQL_injection': 4, 'DDoS_HTTP': 5,
               'DDoS_TCP': 6, 'Password': 7, 'Port_Scanning': 8, 'Vulnerability_scanner': 9, 'Backdoor': 10,
               'XSS': 11, 'Fingerprinting': 12, 'DDoS_UDP': 13, 'DDoS_ICMP': 14}
    df['Attack_type'] = df['Attack_type'].map(attacks)

    corr_features = ['mqtt.topic-Temperature_and_Humidity']
    df = df.drop(corr_features, axis=1)

    print(df.info())

    print(df['Attack_type'].value_counts())
    df.to_csv('preprocessed_ML.csv', encoding='utf-8', index=False)

    return df


def split_dataset(df):
    sns.set(font_scale=0.8)
    y = df['Attack_type']
    X = df.drop(['Attack_type'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    print("Train set size: ", len(X_train))
    print("Test set size: ", len(X_test))

#    # Calculate the correlation matrix
#     corr_matrix = X_train.corr().round(1)

#     # Create a heatmap
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(corr_matrix, cmap='coolwarm', linewidths=0.5)
#     plt.title('Correlation Heatmap')
#     plt.xticks(fontsize=6, rotation=90)
#     plt.yticks(fontsize=6, rotation=0)
#     plt.tight_layout()
#     plt.show()

    # corr_features = correlation(X_train, 0.5)
    # print(corr_features)

    # f_p_values=chi2(X_train,y_train)

    # p_values=pd.Series(f_p_values[1])
    # p_values.index=X_train.columns
    # print(p_values)


    # feat=[]
    # for i in X_train.columns:
    #     if p_values[i] >=0.05 :
    #         feat.append(i)

    # print(feat)

    # df = df.drop(corr_features, axis=1)
    # df = df.drop(feat, axis=1)
    
    # Feature scaling using min-max scaling
    scaler = MinMaxScaler()
    scaled_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    scaled_X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    X_train = scaled_X_train
    X_test = scaled_X_test
        
    return X_train, X_test, y_train, y_test

def correlation(dataset, threshold):
    col_corr = set()  
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: colname = corr_matrix.columns[i]                  
    col_corr.add(colname)
    return col_corr