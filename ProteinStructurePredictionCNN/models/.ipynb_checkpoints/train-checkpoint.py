import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from models.architecture import CNNModel
from models.loss import loss_fn
from models.metrics import Q8_accuracy

do_summary = True

LR = 0.0005
drop_out = 0.3
batch_dim = 64
nn_epochs = 20

early_stop = False  # Early stopping is not directly available in PyTorch
checkpoint = True  # Save the best model

def main():
    # Load your dataset and preprocess as needed
    x, y = dataset.load_data()  # Replace with your data loading code
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_dim, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_dim, shuffle=False)

    model = CNNModel()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    if do_summary:
        print("\nHyper Parameters\n")
        print("Learning Rate: " + str(LR))
        print("Dropout: " + str(drop_out))
        print("Batch Dimension: " + str(batch_dim))
        print("Number of Epochs: " + str(nn_epochs))
        print("\nLoss: Cross-Entropy Loss\n")
        print(model)

    for epoch in range(nn_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0) * targets.size(1)
                    correct += (predicted == targets).sum().item()
            accuracy = correct / total
            print(f"Epoch {epoch}, Test Accuracy: {accuracy * 100:.2f}%")

        if checkpoint:
            torch.save(model.state_dict(), f"allCombsPDB-best-epoch{epoch}.pth")

if __name__ == '__main__':
    main()
