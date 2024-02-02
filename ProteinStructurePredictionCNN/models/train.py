import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from architecture import CNNModel
from loss import loss_fn
from metrics import accuracy

from dataset import load_dataset, get_data_labels
from sklearn.model_selection import train_test_split

do_summary = True

LR = 0.0001
drop_out = 0.3
batch_dim = 64
nn_epochs = 2000
loss_fn2 = torch.nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# You can print the detected device
print(f"Using device: {device}")
early_stop = False  # Early stopping is not directly available in PyTorch
checkpoint = True  # Save the best model
epoch_eval = nn_epochs/10
def load_data(data_file_path,
              use_labels=[],
              test_size=0.2, 
              random_state=42):
    data,num_Y = load_dataset(data_file_path,
                       use_labels=use_labels) #not in git
    X,Y = get_data_labels(data, 
                          num_Y,
                          only_pssm=True)
    
    # Split the dataset into train, validation, and test sets
    X_train, X_temp, Y_train, y_temp = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)
    
    return [X_train, X_test, X_val, Y_train, Y_test, Y_val]
    


def run(data, 
        model=None,
       tag='model'):
    # Load your dataset and preprocess as needed
    [x_train, x_test, x_val, y_train, y_test, y_val] = data
    
    # Modify the input data shape to (-1, 21, 500)
    x_train = x_train.transpose(0, 2, 1)  # shape (-1, 500, 21) --> (-1, 21, 500)
    y_train = y_train.transpose(0, 2, 1)  # shape (-1, 500, 8) --> (-1, 8, 500)
    x_test = x_test.transpose(0, 2, 1)    # shape (-1, 500, 21) --> (-1, 21, 500)
    y_test = y_test.transpose(0, 2, 1)    # shape (-1, 500, 8) --> (-1, 8, 500)

    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_dim, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_dim, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    if do_summary:
        print("\nHyper Parameters\n")
        print("Learning Rate: " + str(LR))
        print("Dropout: " + str(drop_out))
        print("Batch Dimension: " + str(batch_dim))
        print("Number of Epochs: " + str(nn_epochs))
        print("\nLoss: Custom Loss\n")
        print(model)

    model.to(device)

    for epoch in range(nn_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn2(outputs, targets)

            loss.backward()
            optimizer.step()

        if epoch % epoch_eval == 0:
            model.eval()
            total_accuracy = 0.0
            total_samples = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    predicted = model(inputs)
                    _accuracy = accuracy(targets, predicted)
                    correct += _accuracy
                    total += 1

                    average_accuracy = correct / total
                    total_accuracy += average_accuracy * targets.size(0)
                    total_samples += targets.size(0)

            overall_accuracy = total_accuracy / total_samples
            print(f"Epoch {epoch}: Loss: {loss:.6f}, Average Test Accuracy: {overall_accuracy * 100:.2f}%")

            # if checkpoint:
            #     torch.save(model.state_dict(), f"saved_models/{tag}_mdl_best-epoch{epoch}.pth")

    if checkpoint:
        torch.save(model.state_dict(), f"saved_models/{tag}_mdl_best-epoch{epoch}.pth")
    print ("Complete.\n")

if __name__ == '__main__':
    pass
