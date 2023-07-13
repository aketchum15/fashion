import random
import torchvision
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

import matplotlib.pyplot as plt

class ConvFashion(nn.Module):
    def __init__(self, output_channels, kernel_dim):
        super(ConvFashion, self).__init__()

        #in: 1x28x28 out: output_channelsx28x28
        self.conv1 = nn.Conv2d(1, output_channels, kernel_dim)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        #in: output_channelsx28x28 out: output_channelsx28x28
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_dim)
        self.act2 = nn.ReLU()

        #in: output_channelsx28x28 out: output_channelsx12x12
        self.pool2 = nn.MaxPool2d(2);

        #in: output_channelsx12x12 out: output_channels*12*12
        self.flat = nn.Flatten()

        #in: output_channels*144 out: 512
        self.fc3 = nn.Linear(output_channels*144, 512)
        self.act3 = nn.ReLU()

        #in: 512 out: 10 
        self.fc4 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=1)



    def forward(self, x):
        #input: 1x28x28 output: 28x28x28
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        #input:  28x28x28 output: 28x28x28
        x = self.act2(self.conv2(x))
        #input: 28x28x28 output:28x27x27
        x = self.pool2(x)
        #input: 28x14x14 output: 5488
        x = self.flat(x)
        #input: 5488 output: 512
        x = self.act3(self.fc3(x))
        #input: 512 output:10
        x = self.softmax(self.fc4(x))
        return x 

def train(net: nn.Module, data: torch.utils.data.Dataset, max_epochs: int = 5, seed=1337):
    
    torch.manual_seed(seed)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    losses = []
        
    train_data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)


    print("================================ START TRAINING ================================")

    for epoch in range(max_epochs):
        running_loss = 0
        for i, data in  enumerate(train_data_loader):

            x_train, y_train = data

            optimizer.zero_grad()
            y_pred = net(x_train)
            train_loss = loss(y_pred, y_train)
            train_loss.backward()
            optimizer.step()

            # print statistics
            running_loss += train_loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                losses.append(running_loss/100)
                running_loss = 0.0

    print("================================= END TRAINING =================================")

    return net;

def test(data: torch.utils.data.Dataset, model, classes):

    dataloader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    total = 0
    correct = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    print("================================ START TESTING =================================")
    with torch.no_grad():
        for test in dataloader:
            images, labels = test
            out = model(images)
            _, y_pred = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (y_pred == labels).sum().item()

            for label, pred in zip(labels, y_pred):
                if label == pred:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print("================================= END TESTING ==================================")

    accuracy = correct / total
    accs_classes = {}

    print(f'overall accuracy: {accuracy}')
    for classname, correct_count in correct_pred.items():
        accs_classes[classname] = correct_count / total_pred[classname]
        print(f'accuracy for class {classname}: {float(correct_count) / total_pred[classname]}')

    
    return accuracy, accs_classes


def main():
    classes = ('T-shirt', 'Pants', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Shoe', 'Bag', 'Boot')
    D = FashionMNIST(root = './FashionMNIST', download = True, transform=ToTensor())
    T = FashionMNIST(root = './FashionMNIST_Test',  download=True, train=False, transform=ToTensor())
    
    accs = []
    accs_classes = {c: [] for c in classes}

    #change output channels
    for output_channels in range(100):

        model = ConvFashion(output_channels + 1, 3)
        trained_model = train(model, D)

        acc, acc_classes = test(T, trained_model, classes)
        accs.append(acc)

        for c in classes:
            accs_classes[c].append(acc_classes)


    plt.plot(accs)
    fig, ax = plt.subplots(len(classes)//2, 5, )
    for c, i in zip(classes, range(len(classes))):
        ax.flat[i].plot(accs_classes[c])


    plt.show()

if __name__ == '__main__':
    main()

