# METALS : seMi-supervised fEderaTed Active Learning for intrusion detection Systems

Due to limited communication resources, sending the raw data to the central server for model training is no longer practical. It is difficult to get labeled data because data labeling is expensive in terms of time. We develop a semi-supervised federated active learning for IDS, called (METALS). This model takes advantage of Federated Learning (FL) and Active Learning (AL) to reduce the need for a large number of labeled data by actively choosing the instances that should be labeled and keeping the data where it was generated. Specifically, FL trains the model locally and communicates the model parameters instead of the raw data. At the same time, AL allows the model located on the devices to automatically choose and label part of the traffic without involving manual inspection of each training sample.

## Installation

1. Create a python conda evironment and activate it: 
```
conda create -n metals python=3.9
```
```
conda activate metals
```

2. Clone this repository into your  local machine using the following command line:
```
git clone https://github.com/gautamjajoo/FAL.git
```

3. Install the required packages:
```
conda create -n myenv –file package-list.txt
```

4. Add EdgeIIOT dataset in the dataset/EdgeIIOT file. The dataset can be found [here](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot).


5. To run METALS:
```
python main.py
```

Note: configure the parameters by passing them as arguments above

