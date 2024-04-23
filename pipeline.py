import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import FA
import backprop
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device =',device)


def angle_measure(FA, BP):

    flattenFA, flattenBP = torch.flatten(FA, BP)
    value = torch.norm(torch.transpose(flattenFA, 0 ,1) * flattenBP) / (torch.norm(flattenFA) * torch.norm(flattenBP))
    return torch.acos(value)



def train_model(model, optimizer, train_loader, num_epochs, criterion, batchSize=32):
    logName = f"{model.__class__.__name__}_{model.num_layers}"
    print("Training", logName)
    logger_train = open(f"logs/{logName}.txt", 'w')
    for epoch in range(1, num_epochs+1):
        for idx_batch, (inputs, targets) in enumerate(train_loader):
            targetVectors = F.one_hot(targets, 10)
            inputs = inputs.view(batchSize, -1)
            inputs, targetVectors = Variable(inputs), Variable(targetVectors)
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targetVectors.to(device).float())
            model.zero_grad()
            loss.backward()
            optimizer.step()

            if (idx_batch + 1) % 100 == 0:
                train_log = 'epoch ' + str(epoch) + ' step ' + str(idx_batch + 1) + \
                            ' loss ' + str(loss.item())
                print(f"epoch {epoch};  step {idx_batch+1}          ", end='\r')
                logger_train.write(train_log + '\n')


def train_2_models(modelFA, modelBP, optimizerFA,optimizerBP , train_loader, num_epochs, criterion, batchSize=32):
    logName = f"{modelFA.__class__.__name__}_{modelFA.num_layers}"
    print("Training", logName)
    logger_train = open(f"logs/{logName}.txt", 'w')
    for epoch in range(1, num_epochs+1):
        for idx_batch, (inputs, targets) in enumerate(train_loader):
            targetVectors = F.one_hot(targets, 10)
            inputs = inputs.view(batchSize, -1)
            inputs, targetVectors = Variable(inputs), Variable(targetVectors)
            outputsFA = modelFA(inputs.to(device))
            outputsBP = modelBP(inputs.to(device))
            lossFA = criterion(outputsFA, targetVectors.to(device).float())
            lossBP = criterion(outputsBP, targetVectors.to(device).float())
            modelFA.zero_grad()
            modelBP.zero_grad()
            lossFA.backward()
            lossBP.backward()
            optimizerFA.step()
            optimizerBP.step()

            matriceFA = modelFA.linear[-1].weight_fa * lossFA.item()
            matriceBP = modelBP.linear[-1].weight * lossBP.item()
            angle =   angle_measure(matriceFA, matriceBP)

            if (idx_batch + 1) % 100 == 0:
                train_log = 'epoch ' + str(epoch) + ' step ' + str(idx_batch + 1) + \
                            ' lossFA ' + str(lossFA.item()) + ' lossBP ' + str(lossFA.item()) + ' angle '+ str(angle.item())
                print(f"epoch {epoch};  step {idx_batch+1}          ", end='\r')
                logger_train.write(train_log + '\n')

                

                
                

def test_model(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            targetVectors = F.one_hot(targets, 10)
            inputs = data.view(data.size(0), -1)
            inputs, targetVectors, targets = Variable(inputs), Variable(targetVectors), Variable(targets)
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targetVectors.to(device).float())
            test_loss += loss.item() * inputs.size(0)
            total += targets.size(0)
            correct += (outputs.argmax(dim=1) == targets.to(device)).sum().item()

    test_loss = test_loss / total
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
    loss_crossentropy = torch.nn.CrossEntropyLoss()
    loss_mse = torch.nn.MSELoss()
    
    model_fa = FA.FANetwork(in_features=784, num_layers=4, num_hidden_list=[1000, 30, 20, 10], activation_function=F.tanh).to(device)
    optimizer_fa = torch.optim.SGD(model_fa.parameters(),lr=1e-4, momentum=0.9, weight_decay=0.001, nesterov=True)
    train_model(model_fa, optimizer_fa,train_loader, num_epochs=50, criterion=loss_mse, batchSize=BATCH_SIZE)
    print("Testing Feedforward DFA Model...")
    test_model(model_fa, test_loader, loss_mse)

    model_bp = backprop.BackPropNetwork(in_features=784, num_layers=4, num_hidden_list=[1000, 30, 20, 10]).to(device)
    optimizer_bp = torch.optim.SGD(model_bp.parameters(),lr=1e-4, momentum=0.9, weight_decay=0.001, nesterov=True)
    train_model(model_bp, optimizer_bp, train_loader, num_epochs=50, criterion=loss_mse, batchSize=BATCH_SIZE)
    print("Testing BackProp Model...")
    test_model(model_bp, test_loader, loss_mse)
    

    model_fa = FA.FANetwork(in_features=784, num_layers=4, num_hidden_list=[1000, 30, 20, 10], activation_function=F.tanh).to(device)
    optimizer_fa = torch.optim.SGD(model_fa.parameters(),lr=1e-4, momentum=0.9, weight_decay=0.001, nesterov=True)
    model_bp = backprop.BackPropNetwork(in_features=784, num_layers=4, num_hidden_list=[1000, 30, 20, 10]).to(device)
    optimizer_bp = torch.optim.SGD(model_bp.parameters(),lr=1e-4, momentum=0.9, weight_decay=0.001, nesterov=True)
    train_2_models(model_fa, model_bp, optimizer_fa, optimizer_bp,train_loader, num_epochs=50, criterion=loss_mse, batchSize=BATCH_SIZE)
    