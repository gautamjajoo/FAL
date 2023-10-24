import matplotlib.pyplot as plt
import numpy as np


# Define the classification reports for each model
classification_reports = [
    {
        'model': 'Logistic',
        'report': {
            '0': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0},
            '1': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0},
            '2': {'precision': 0.6462585034013606, 'recall': 0.39034411915767847, 'f1-score': 0.48671149535702857},
            '3': {'precision': 0.5338541666666666, 'recall': 0.32625994694960214, 'f1-score': 0.405004939084623},
            '4': {'precision': 0.4319955406911929, 'recall': 0.37241710716001925, 'f1-score': 0.4},
            '5': {'precision': 0.8765957446808511, 'recall': 0.5894134477825465, 'f1-score': 0.7048759623609923},
            '6': {'precision': 0.5779636005902608, 'recall': 0.5842864246643461, 'f1-score': 0.5811078140454995},
            '7': {'precision': 0.41130604288499023, 'recall': 0.6267326732673267, 'f1-score': 0.4966653589642997},
            '8': {'precision': 0.4768392370572207, 'recall': 0.5017201834862385, 'f1-score': 0.4889633975970942},
            '9': {'precision': 0.8470254957507082, 'recall': 0.8555078683834049, 'f1-score': 0.8512455516014236},
            '10': {'precision': 0.5381372173246455, 'recall': 0.7270844122216468, 'f1-score': 0.6185022026431718},
            '11': {'precision': 0.5745814307458144, 'recall': 0.7696228338430173, 'f1-score': 0.6579520697167757},
            '12': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0},
            '13': {'precision': 0.9850948509485095, 'recall': 0.9941880341880341, 'f1-score': 0.9896205547047814},
            '14': {'precision': 0.9560885608856089, 'recall': 0.9881769641495042, 'f1-score': 0.9718679669917479}
        },
        'accuracy': 0.7179040735873851,
    },
    {
        'model': 'Naive Bayes',
        'report': {
            '0': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0},
            '1': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0},
            '2': {'precision': 0.612952968388589, 'recall': 0.4083204930662558, 'f1-score': 0.49013563501849566},
            '3': {'precision': 0.6947535771065183, 'recall': 0.23183023872679046, 'f1-score': 0.34765314240254575},
            '4': {'precision': 0.4119337016574586, 'recall': 0.8957232099951946, 'f1-score': 0.5643354526188314},
            '5': {'precision': 0.4544863459037711, 'recall': 1.0, 'f1-score': 0.6249441215914171},
            '6': {'precision': 0.33860919346691365, 'recall': 1.0, 'f1-score': 0.5059119496855347},
            '7': {'precision': 0.995575221238938, 'recall': 0.11138613861386139, 'f1-score': 0.20035618878005343},
            '8': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0},
            '9': {'precision': 0.9275461380724539, 'recall': 0.6471149260848832, 'f1-score': 0.7623595505617978},
            '10': {'precision': 0.6522148916116871, 'recall': 0.35836354220611083, 'f1-score': 0.46256684491978617},
            '11': {'precision': 0.9873417721518988, 'recall': 0.039755351681957186, 'f1-score': 0.07643312101910828},
            '12': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0},
            '13': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0},
            '14': {'precision': 0.957983193277311, 'recall': 1.0, 'f1-score': 0.9785407725321889}
        },
        'accuracy': 0.6585742444152431,
    },
    {
        'model': 'Random Forest',
        'report': {
            '0': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0},
            '1': {'precision': 1.0, 'recall': 0.9863013698630136, 'f1-score': 0.993103448275862},
            '2': {'precision': 0.7422779922779923, 'recall': 0.7899332306111967, 'f1-score': 0.7653645185369494},
            '3': {'precision': 0.9322751322751323, 'recall': 0.9347480106100796, 'f1-score': 0.9335099337748346},
            '4': {'precision': 0.7591312931885489, 'recall': 0.739067755886593, 'f1-score': 0.7489651813976139},
            '5': {'precision': 0.889927429072138, 'recall': 0.8362797936014613, 'f1-score': 0.8622881355932204},
            '6': {'precision': 0.7178104554398442, 'recall': 0.7476679841897233, 'f1-score': 0.7324022346368715},
            '7': {'precision': 0.688929889298893, 'recall': 0.7004950495049505, 'f1-score': 0.6946692607003891},
            '8': {'precision': 0.6861614497528833, 'recall': 0.6773700305810397, 'f1-score': 0.6817412720217252},
            '9': {'precision': 0.8742761092342569, 'recall': 0.8467658902442797, 'f1-score': 0.8602958177771044},
            '10': {'precision': 0.7673748461665537, 'recall': 0.769144684294024, 'f1-score': 0.7682593233896199},
            '11': {'precision': 0.8895734597156398, 'recall': 0.8912665486437444, 'f1-score': 0.8904198918362474},
            '12': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0},
            '13': {'precision': 0.9959415584415584, 'recall': 0.9982905982905983, 'f1-score': 0.9971153846153846},
            '14': {'precision': 0.9990079365079365, 'recall': 0.9943708609271523, 'f1-score': 0.9966832504145933}
        },
        'accuracy': 0.87884363,
    },
    {
        'model': 'SVM',
        'report': {
            '0': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0},
            '1': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0},
            '2': {'precision': 0.9857808857808858, 'recall': 0.9306848319885222, 'f1-score': 0.9573876799058493},
            '3': {'precision': 0.9828707317073171, 'recall': 0.9343868659597769, 'f1-score': 0.9589956412256689},
            '4': {'precision': 0.9515121501340483, 'recall': 0.9074970916022945, 'f1-score': 0.9299785299800997},
            '5': {'precision': 0.9959677419354839, 'recall': 0.9665740740740741, 'f1-score': 0.9810442930153322},
            '6': {'precision': 0.9778869778869779, 'recall': 0.9794788273615636, 'f1-score': 0.9786821705426356},
            '7': {'precision': 0.9935391241923905, 'recall': 0.9613861386138614, 'f1-score': 0.9771778362720403},
            '8': {'precision': 0.9788190613457543, 'recall': 0.9824528630729517, 'f1-score': 0.9806322425059101},
            '9': {'precision': 0.9352014011546313, 'recall': 0.9505425357699805, 'f1-score': 0.942807315523936},
            '10': {'precision': 0.979468253968254, 'recall': 0.9633802816901409, 'f1-score': 0.9713483146067417},
            '11': {'precision': 0.9875389408099688, 'recall': 0.9889485666804162, 'f1-score': 0.988243109079748},
            '12': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0},
            '13': {'precision': 0.9986301369863014, 'recall': 0.9991452991452992, 'f1-score': 0.9988876529477197},
            '14': {'precision': 0.9959514170040485, 'recall': 0.9968035190615836, 'f1-score': 0.996377672209026}
        },
        'accuracy': 0.71727989,
    },
    {
        'model': 'KNN',
        'report': {
            '0': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0},
            '1': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0},
            '2': {'precision': 0.7104256876855904, 'recall': 0.8566630152302599, 'f1-score': 0.776790559153524},
            '3': {'precision': 0.9201535508607199, 'recall': 0.7750281214848141, 'f1-score': 0.8413708392065087},
            '4': {'precision': 0.8923444976076555, 'recall': 0.6981974710171085, 'f1-score': 0.7824523036881696},
            '5': {'precision': 0.8918918918918919, 'recall': 0.9876543209876543, 'f1-score': 0.9379518072289157},
            '6': {'precision': 0.7817298347910598, 'recall': 0.868219197951048, 'f1-score': 0.8222222222222223},
            '7': {'precision': 0.6744516678451668, 'recall': 0.7772277227722773, 'f1-score': 0.7226027397260274},
            '8': {'precision': 0.7877358490566038, 'recall': 0.6793692506925208, 'f1-score': 0.7291666666666667},
            '9': {'precision': 0.8491431492842537, 'recall': 0.9043631744156967, 'f1-score': 0.8758238253032325},
            '10': {'precision': 0.7471910112359551, 'recall': 0.852112676056338, 'f1-score': 0.7957171556002997},
            '11': {'precision': 0.94996632996633, 'recall': 0.9775410021636234, 'f1-score': 0.963548309178744},
            '12': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0},
            '13': {'precision': 0.9974747474747475, 'recall': 0.9982905982905983, 'f1-score': 0.9978820192084146},
            '14': {'precision': 0.9943133583021223, 'recall': 0.992781568850115, 'f1-score': 0.9935467505042666}
        },
        'accuracy': 0.82573174,
    },
]

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
for i, report_data in enumerate(classification_reports):
    for class_name in class_names:
        precision_scores[i].append(report_data['report'][str(class_mapping[class_name])]['precision'])
        recall_scores[i].append(report_data['report'][str(class_mapping[class_name])]['recall'])
        f1_scores[i].append(report_data['report'][str(class_mapping[class_name])]['f1-score'])

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

# Set x-axis labels and positions
axs_precision.set_xticks(x_models + ((bar_width + bar_spacing) * (len(classification_reports) - 1)) / 2)
axs_precision.set_xticklabels([report_data['model'] for report_data in classification_reports], rotation=0, ha='center')
axs_recall.set_xticks(x_models + ((bar_width + bar_spacing) * (len(classification_reports) - 1)) / 2)
axs_recall.set_xticklabels([report_data['model'] for report_data in classification_reports], rotation=0, ha='center')
axs_f1.set_xticks(x_models + ((bar_width + bar_spacing) * (len(classification_reports) - 1)) / 2)
axs_f1.set_xticklabels([report_data['model'] for report_data in classification_reports], rotation=0, ha='center')

# Set legend and adjust layout
axs_precision.legend(loc='upper left', bbox_to_anchor=(1, 0.8))
axs_recall.legend(loc='upper left', bbox_to_anchor=(1, 0.8))
axs_f1.legend(loc='upper left', bbox_to_anchor=(1, 0.8))
plt.tight_layout()

# Show the three figures
plt.show()



