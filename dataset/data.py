import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

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

    print(df.info())

    corr_features = ['mqtt.topic-Temperature_and_Humidity']
    feat = ['http.request.method-PUT',
            'http.referer-TESTING_PURPOSES_ONLY',
            'http.request.version-> HTTP/1.1',
            'http.request.version-By Dr HTTP/1.1',
            'dns.qry.name.len-_googlecast._tcp.local',
            'dns.qry.name.len-null-null.local',
            'mqtt.conack.flags-1461073',
            'mqtt.conack.flags-1461074',
            'mqtt.conack.flags-1461383',
            'mqtt.conack.flags-1461384',
            'mqtt.conack.flags-1461589',
            'mqtt.conack.flags-1461591',
            'mqtt.conack.flags-1471198',
            'mqtt.conack.flags-1471199',
            'mqtt.conack.flags-1574358',
            'mqtt.conack.flags-1574359']

    df = df.drop(corr_features, axis=1)
    df = df.drop(feat, axis=1)
    df.to_csv('preprocessed_DNN.csv', encoding='utf-8', index=False)
    print(df['Attack_type'].value_counts())
    return df


def split_dataset(df, seed, size, labeled_data_ratio):
    # print(df['Attack_type'].value_counts())

    y = df['Attack_type']
    X = df.drop(['Attack_type'], axis=1)

    X_train, X_break, y_train, y_break = train_test_split(X, y, random_state=seed, test_size=size)
    X_test, X_val, y_test, y_val = train_test_split(X_break, y_break, random_state=seed+1, test_size=0.5)

    print("Train set size: ", len(X_train))
    print("Validation set size: ", len(X_val))
    print("Test set size: ", len(X_test))

    # # Plot a bar graph to visualize class percentages
    # plt.figure(figsize=(10, 6))
    # plt.bar(unique_classes, class_sample_percentages)
    # plt.xlabel('Class')
    # plt.ylabel('Percentage (%)')
    # plt.title('Percentage of Each Sample Class in y_train')
    # plt.xticks(rotation=45)
    # plt.show()

    # smote = SMOTE(sampling_strategy='auto', random_state=42)
    # X_train, y_train = smote.fit_resample(X_train, y_train)

    # # Calculate the counts and percentages of each class in y_train
    # class_sample_counts = [np.sum(y_train == cls) for cls in unique_classes]
    # class_sample_percentages = [(count / len(y_train)) * 100 for count in class_sample_counts]

    # for cls, count, percentage in zip(unique_classes, class_sample_counts, class_sample_percentages):
    #     print(f"Class {cls}: {count} samples ({percentage:.2f}%)")

    # # Plot a bar graph to visualize class percentages
    # plt.figure(figsize=(10, 6))
    # plt.bar(unique_classes, class_sample_percentages)
    # plt.xlabel('Class')
    # plt.ylabel('Percentage (%)')
    # plt.title('Percentage of Each Sample Class in y_train')
    # plt.xticks(rotation=45)
    # plt.show()

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


# df = preprocess_dataset(file_path)
# X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, seed=1, size=0.2)

# Print the shapes of the resulting datasets
# print("Training set shape:", X_train.shape)
# print("Validation set shape:", X_val.shape)
# print("Test set shape:", X_test.shape)

# Print additional information if needed
# X_train.info()
