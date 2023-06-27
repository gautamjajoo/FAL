import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


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
    df = load_dataset(file_path)
    print(df['Attack_type'].value_counts())

    columns_to_drop = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
                       "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp", "http.request.uri.query",
                       "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg"]

    df = drop_columns(df, columns_to_drop)
    df = drop_missing_values(df)
    df = drop_duplicates(df)

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

    return df


def split_dataset(df, seed, size):
    y = df['Attack_type']
    X = df.drop(['Attack_type'], axis=1)

    X_train, X_break, y_train, y_break = train_test_split(X, y, random_state=seed, test_size=size)
    X_test, X_val, y_test, y_val = train_test_split(X_break, y_break, random_state=seed+1, test_size=0.5)

    return X_train, X_val, X_test, y_train, y_val, y_test


# Test the data pre processing part
# file_path = '/Users/gautamjajoo/Desktop/FAL/dataset/Edge-IIoTset/DNN-EdgeIIoT-dataset.csv'

# df = preprocess_dataset(file_path)
# X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, seed=1, size=0.2)

# Print the shapes of the resulting datasets
# print("Training set shape:", X_train.shape)
# print("Validation set shape:", X_val.shape)
# print("Test set shape:", X_test.shape)

# Print additional information if needed
# X_train.info()