import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from lib import fa_linear
from lib import linear
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device =',device)


def train_model(model, optimizer, train_loader, num_epochs):
    modelName = model.__class__.__name__
    print("\nTraining", modelName)
    logger_train = open('logs/' + modelName + 'train_log.txt', 'w')
    for epoch in range(1, num_epochs+1):
        for idx_batch, (inputs, targets) in enumerate(train_loader):
            # flatten the inputs from square image to 1d vector
            inputs = inputs.view(BATCH_SIZE, -1)
            # wrap them into varaibles
            inputs, targets = Variable(inputs), Variable(targets)
            # get outputs from the model
            outputs = model(inputs.to(device))
            loss = loss_crossentropy(outputs, targets.to(device))

            model.zero_grad()
            loss.backward()
            optimizer.step()

            if (idx_batch + 1) % 10 == 0:
                train_log = 'epoch ' + str(epoch) + ' step ' + str(idx_batch + 1) + \
                            ' loss ' + str(loss.item())
                print(f"epoch {epoch};  step {idx_batch+1}          ", end='\r')
                logger_train.write(train_log + '\n')


def test_model(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            # Flatten the inputs from square image to 1d vector
            inputs = data.view(data.size(0), -1)
            # Wrap inputs and targets into variables
            inputs, targets = Variable(inputs), Variable(targets)
            # Get outputs from the model
            outputs = model(inputs.to(device))
            # Calculate loss
            loss = criterion(outputs, targets.to(device))
            # Sum up batch loss
            test_loss += loss.item() * inputs.size(0)
            # Get the predicted labels
            _, predicted = torch.max(outputs.data, 1)
            # Total number of samples
            total += targets.size(0)
            # Number of correct predictions
            correct += (predicted == targets.to(device)).sum().item()

    # Average test loss
    test_loss = test_loss / total
    # Accuracy
    accuracy = 100 * correct / total

    print('Test Loss: {:.4f}'.format(test_loss))
    print('Accuracy: {:.2f}%'.format(accuracy))



if __name__ == '__main__':
    BATCH_SIZE = 32
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))
                                            ])),
                            batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))
                                            ])),
                            batch_size=BATCH_SIZE, shuffle=True)

    model_fa = fa_linear.LinearFANetwork(in_features=784, num_layers=2, num_hidden_list=[1000, 10]).to(device)
    model_bp = linear.LinearNetwork(in_features=784, num_layers=2, num_hidden_list=[1000, 10]).to(device)
    optimizer_fa = torch.optim.SGD(model_fa.parameters(),lr=1e-4, momentum=0.9, weight_decay=0.001, nesterov=True)
    optimizer_bp = torch.optim.SGD(model_bp.parameters(),lr=1e-4, momentum=0.9, weight_decay=0.001, nesterov=True)
    loss_crossentropy = torch.nn.CrossEntropyLoss()

    train_model(model_bp, optimizer_bp, train_loader, num_epochs=5)
    print("Testing BackProp Linear Model...")
    test_model(model_bp, test_loader, loss_crossentropy)
    
    train_model(model_fa, optimizer_fa,train_loader, num_epochs=5)
    print("Testing Feedforward DFA Model...")
    test_model(model_fa, test_loader, loss_crossentropy)

    