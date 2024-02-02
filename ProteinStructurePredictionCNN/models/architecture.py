import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, input_channels, 
                 output_classes, 
                 window_size=11,
                 padding=5,
                 
                 dropout_probability=0.3):
        super(CNNModel, self).__init__()
        
        # Define the convolutional layers and activation functions in a sequential manner
        self.dropout_probability = dropout_probability
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=256, kernel_size=window_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=window_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=window_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            
            nn.Conv1d(in_channels=64, out_channels=output_classes, kernel_size=window_size, padding=padding)
        )
        
        # Softmax activation for classification
        self.activation = nn.Sigmoid()#nn.Softmax(dim=2)

    def forward(self, x):
        # Forward pass through the layers defined in nn.Sequential
        x = self.model(x)
        x = self.activation(x)

        return x

class CNNModel0(nn.Module):
    def __init__(self, input_channels, 
                 output_classes, 
                 window_size=11,
                 padding=5,
                 
                 dropout_probability=0.3):
        super(CNNModel0, self).__init__()
        
        # Define the convolutional layers and activation functions in a sequential manner
        self.dropout_probability = dropout_probability
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=256, kernel_size=window_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=window_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=window_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=window_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            
            nn.Conv1d(in_channels=64, out_channels=output_classes, kernel_size=window_size, padding=padding)
        )
        
        # Softmax activation for classification
        self.activation = nn.Softmax(dim=2) #nn.Sigmoid()#Tanh()#

    def forward(self, x):
        # Forward pass through the layers defined in nn.Sequential
        x = self.model(x)
        x = self.activation(x)

        return x
