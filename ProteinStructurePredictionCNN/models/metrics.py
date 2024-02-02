import torch 

def accuracy(real,pred):
    correct = 0
    total = 0

    for i in range(real.size(0)):  # per element in the batch
        for j in range(real.size(1)):  # per amino acid residue
            true_labels = real[i, j, :]
            predicted_labels = (pred[i, j, :] > 0.5).to(torch.float32)  # Use threshold of 0.5 for binary classification
            # Calculate accuracy for this position
            accuracy = (true_labels == predicted_labels).all().item()  # Check if all labels match
            correct += accuracy
            total += 1

    return correct / total

def accuracy_max_value(real,pred):
    pass