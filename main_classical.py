import torch
from config import args_parser
from dataset.data_classical import preprocess_dataset, split_dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB  
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

file_path = '/Users/gautamjajoo/Desktop/FAL/dataset/Edge-IIoTset/ML-EdgeIIoT-dataset.csv'


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(args.seed)

    df = preprocess_dataset(file_path)
    num_classes = df['Attack_type'].nunique()
    input_features = df.drop(['Attack_type'], axis=1).shape[1]
    print("Number of classes:", num_classes)
    print("Number of input features:", input_features)

    X_train, X_test, y_train, y_test = split_dataset(df)

    unique_classes = y_train.unique()
    class_sample_counts = [np.sum(y_train == cls) for cls in unique_classes]
    for cls, count in zip(unique_classes, class_sample_counts):
        print(f"Class {cls}: {count} samples")

    unique_classes = y_test.unique()
    class_sample_counts = [np.sum(y_test == cls) for cls in unique_classes]
    for cls, count in zip(unique_classes, class_sample_counts):
        print(f"Class {cls}: {count} samples")

    # Feature scaling using min-max scaling
    scaler = MinMaxScaler()
    scaled_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    scaled_X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    X_train = scaled_X_train
    X_test = scaled_X_test

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    print("Length of train size: ")

    # Print the shapes of the resulting datasets
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    print("Done...")

    # Initialize an empty classification_reports dictionary
    classification_reports = {}
    
    def plot_confusion_matrix(cm, model_name):
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.title(f'{model_name} - Confusion Matrix')
        plt.show()

    def plot_accuracy_bar(accuracies):
        plt.figure(figsize=(8, 5))
        plt.bar(accuracies.keys(), accuracies.values())
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison Across Models')
        plt.ylim([0, 1])
        # plt.xticks(rotation=45)
        for model, accuracy in accuracies.items():
            plt.text(model, accuracy, f'{accuracy:.4f}', ha='center', va='bottom')

        plt.show()

    # Logistic Regression
    classifier = LogisticRegression(random_state=42, max_iter=5000)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    acc_logistic = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    classification_reports["logistic"] = report

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "Logistic Regression")

    print(f'Accuracy: {acc_logistic * 100:.6f}%')
    print("\nClassification Report:\n", report)

    # Naive Bayes
    classifier = GaussianNB()  
    classifier.fit(X_train, y_train) 

    y_pred = classifier.predict(X_test)

    acc_naive_bayes = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "Naive Bayes")

    classification_reports["naive_bayes"] = report

    print(f'Accuracy: {acc_naive_bayes * 100:.6f}%')
    print("\nClassification Report:\n", report)

    # Decision Tree
    clf = DecisionTreeClassifier(criterion = 'entropy', random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc_decision_tree = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    classification_reports["decision_tree"] = report

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "Decision Tree")

    print(f'Accuracy: {acc_decision_tree * 100:.6f}%')
    print("\nClassification Report:\n", report)

    # Random Forest
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    acc_random_forest = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    classification_reports["random_forest"] = report


    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "Random Forest")

    print(f'Accuracy: {acc_random_forest * 100:.6f}%')
    print("\nClassification Report:\n", report)

    # SVM
    classifier = SVC(kernel='linear', random_state=0)  
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    acc_svm = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    classification_reports["svm"] = report

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "SVM")

    print(f'Accuracy: {acc_svm * 100:.6f}% \n')
    print("Classification Report:\n", report)

    # KNN
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)

    acc_knn = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    classification_reports["knn"] = report

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "KNN")

    print(f'Accuracy: {acc_knn * 100:.6f}%')
    print("\nClassification Report:\n \n", report)

    accuracies = {
        "Logistic Regression": acc_logistic,
        "Naive Bayes": acc_naive_bayes,
        "Decision Tree": acc_decision_tree,
        "Random Forest": acc_random_forest,
        "SVM": acc_svm,
        "KNN": acc_knn
    }

    plot_accuracy_bar(accuracies)

    class_mapping = {
        'Normal': 0,
        'MITM': 1,
        'Uploading': 2,
        'Ransomware': 3,
        'SQL_injection': 4,
        'DDoS_HTTP': 5,
        'DDoS_TCP': 6,
        'Password': 7,
        'Port_Scanning': 8,
        'Vulnerability_scanner': 9,
        'Backdoor': 10,
        'XSS': 11,
        'Fingerprinting': 12,
        'DDoS_UDP': 13,
        'DDoS_ICMP': 14
    }

    # Define class names
    class_names = [
        'Normal',
        'MITM',
        'Uploading',
        'Ransomware',
        'SQL_injection',
        'DDoS_HTTP',
        'DDoS_TCP',
        'Password',
        'Port_Scanning',
        'Vulnerability_scanner',
        'Backdoor',
        'XSS',
        'Fingerprinting',
        'DDoS_UDP',
        'DDoS_ICMP',
    ]

    # Create a figure for precision
    fig_precision, axs_precision = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(right=0.8) 
    axs_precision.set_title('Precision')
    axs_precision.set_xlabel('Model')
    axs_precision.set_ylabel('Precision')

    # Create a figure for recall
    fig_recall, axs_recall = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(right=0.8) 
    axs_recall.set_title('Recall')
    axs_recall.set_xlabel('Model')
    axs_recall.set_ylabel('Recall')

    # Create a figure for F1-score
    fig_f1, axs_f1 = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(right=0.8) 
    axs_f1.set_title('F1-score')
    axs_f1.set_xlabel('Model')
    axs_f1.set_ylabel('F1-score')

    # Create empty arrays for precision, recall, and f1-score for each model
    precision_scores = [[] for _ in range(len(classification_reports))]
    recall_scores = [[] for _ in range(len(classification_reports))]
    f1_scores = [[] for _ in range(len(classification_reports))]

    # Extract precision, recall, and f1-score for each class and model
    for i, (model_name, report_data) in enumerate(classification_reports.items()):
        for class_name in class_names:
            print(model_name)  # This will print the model name
            class_id = class_mapping[class_name]
            precision_scores[i].append(report_data[str(class_id)]['precision'])
            recall_scores[i].append(report_data[str(class_id)]['recall'])
            f1_scores[i].append(report_data[str(class_id)]['f1-score'])

    # Transpose the data
    precision_scores = np.array(precision_scores).T
    recall_scores = np.array(recall_scores).T
    f1_scores = np.array(f1_scores).T

    # Create x-axis values for models
    x_models = np.arange(len(classification_reports))

    # Reduce the bar width
    bar_width = 0.05

    # Create bar spacing
    bar_spacing = 0.0

    # Define custom colors for bars
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon', 'lightseagreen', 'lightpink']

    # Plot precision, recall, and F1-score for each class as separate bars
    for i, class_name in enumerate(class_names):
        x_values = x_models + (i * (bar_width + bar_spacing))

        axs_precision.bar(x_values, precision_scores[i], width=bar_width, label=class_name, color=colors[i % len(colors)])
        axs_recall.bar(x_values, recall_scores[i], width=bar_width, label=class_name, color=colors[i % len(colors)])
        axs_f1.bar(x_values, f1_scores[i], width=bar_width, label=class_name, color=colors[i % len(colors)])


    model_names = [model_name for model_name, _ in classification_reports.items()]

    # Set x-axis labels and positions
    print(classification_reports)
    axs_precision.set_xticks(x_models + ((bar_width + bar_spacing) * (len(classification_reports) - 1)) / 2)
    axs_precision.set_xticklabels(model_names, rotation=0, ha='center')
    axs_recall.set_xticks(x_models + ((bar_width + bar_spacing) * (len(classification_reports) - 1)) / 2)
    axs_recall.set_xticklabels(model_names, rotation=0, ha='center')
    axs_f1.set_xticks(x_models + ((bar_width + bar_spacing) * (len(classification_reports) - 1)) / 2)
    axs_f1.set_xticklabels(model_names, rotation=0, ha='center')

    # Set legend and adjust layout
    axs_precision.legend(loc='upper left', bbox_to_anchor=(1, 0.8))
    axs_recall.legend(loc='upper left', bbox_to_anchor=(1, 0.8))
    axs_f1.legend(loc='upper left', bbox_to_anchor=(1, 0.8))
    plt.tight_layout()

    # Show the three figures
    plt.show()



