

# Define your custom loss functions here if needed
# loss_fn = nn.CrossEntropyLoss()  # For now: Cross-Entropy Loss
# loss_fn = nn.BCEWithLogitsLoss()

import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import accuracy


class SSAccuracyLoss1(nn.Module):
    def __init__(self):
        super(SSAccuracyLoss, self).__init__()

    def forward(self, real, pred):
        total = real.size(0) * real.size(1)
        correct = 0

        for i in range(real.size(0)):  # per element in the batch
            for j in range(real.size(1)):  # per amino acid residue
                if torch.sum(real[i, j, :]) == 0:  # if it is padding
                    total = total - 1
                else:
                    if real[i, j, torch.argmax(pred[i, j, :])] > 0:
                        correct = correct + 1

        accuracy = correct / total
        loss = 1 - accuracy  # You can use 1 - accuracy as a loss
        print (real, pred, loss)
        return torch.tensor(loss, requires_grad=True)
    
    
class SSAccuracyLoss2(nn.Module):
    def __init__(self):
        super(SSAccuracyLoss, self).__init__()

    def forward(self, real, pred):
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

        accuracy = correct / total
        loss = 1 - accuracy  # You can use 1 - accuracy as a loss

        return torch.tensor(loss, requires_grad=True)
    

class SSAccuracyLoss3(nn.Module):
    def __init__(self):
        super(SSAccuracyLoss, self).__init__()

    def forward(self, real, pred):
        correct_max_value = 0
        correct_labels = 0
        total = 0

        for i in range(real.size(0)):  # per element in the batch
            for j in range(real.size(1)):  # per amino acid residue
                true_labels = real[i, j, :]
                max_value_index = torch.argmax(pred[i, j, :]).item()
                predicted_labels = (pred[i, j, :] > 0.5).to(torch.float32)  # Use threshold of 0.5 for binary classification

                # Check if the maximum predicted value index matches the true labels
                correct_max_value += (max_value_index == torch.argmax(true_labels)).item()

                # Check if all labels match
                correct_labels += (true_labels == predicted_labels).all().item()

                total += 1

        accuracy_max_value = correct_max_value / total
        accuracy_labels = correct_labels / total

        # Calculate the combined penalty as a weighted sum of the two accuracies
        combined_loss = 0.5 * (1 - accuracy_max_value) + 0.5 * (1 - accuracy_labels)
        return torch.tensor(combined_loss, requires_grad=True)

class SSAccuracyLoss(nn.Module):
    def __init__(self):
        super(SSAccuracyLoss, self).__init__()

    def forward(self, real, pred):
        # correct_max_value = 0


        accuracy_labels = accuracy(real,pred)

        # accuracy_max_value = correct_max_value / total
        # accuracy_labels = correct / total

        # Calculate the combined penalty as a weighted sum of the two accuracies
        # combined_loss = 0.5 * (1 - accuracy_max_value) 
        combined_loss = + 0.5 * (1 - accuracy_labels)
        return torch.tensor(combined_loss, requires_grad=True)

loss_fn = SSAccuracyLoss()