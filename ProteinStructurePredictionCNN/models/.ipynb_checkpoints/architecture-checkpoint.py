import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Define your model architecture here
        self.conv1 = nn.Conv1d(in_channels=20, out_channels=128, kernel_size=11, padding=5)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=11, padding=5)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=num_classes, kernel_size=11, padding=5)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # Define the forward pass of your model
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.softmax(x)
        return x
