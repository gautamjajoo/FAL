import matplotlib.pyplot as plt

# Data
rounds = list(range(1, 11))
fedavg = [0.934654, 0.939456, 0.933952, 0.9422, 0.947795, 0.947562, 0.9444451, 0.947852, 0.948334, 0.947931]
fedprox_03 = [0.854329, 0.939372, 0.940733, 0.947266, 0.947808, 0.947826, 0.948337, 0.948360, 0.948371, 0.948316]
fedprox_05 = [0.935162, 0.937908, 0.939898, 0.946802, 0.947638, 0.948190, 0.948373, 0.948049, 0.947868, 0.948538]
fedprox_08 = [0.839871, 0.934219, 0.938126, 0.940736, 0.946745, 0.943956, 0.948164, 0.937518, 0.948224, 0.948211]
fedprox_1 = [0.933410, 0.907424, 0.933591, 0.911312, 0.944909, 0.932059, 0.947117, 0.947766, 0.948300, 0.948352]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(rounds, fedavg, marker='o', label='FedAvg')
plt.plot(rounds, fedprox_03, marker='o', label='FedProx(0.3)')
plt.plot(rounds, fedprox_05, marker='o', label='FedProx(0.5)')
plt.plot(rounds, fedprox_08, marker='o', label='FedProx(0.8)')
plt.plot(rounds, fedprox_1, marker='o', label='FedProx(1)')

# Add labels and title
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Round(Classical Federated Learning)')
plt.legend()

# Set y-axis limits with some padding for the outlier values
plt.ylim(min(fedprox_08) - 0.005, max(fedavg) + 0.005)

# Show plot
plt.grid(True)
plt.show()

