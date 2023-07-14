from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import torch
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
import numpy as np

import matplotlib.pyplot as plt

from model import ConvFashion 
#for conv channels plateaus at 16 ish
#channel ratio between conv1 and conv2 plateaus at about 4
def train(net: Module, data: Dataset, max_epochs: int = 5, seed=1337):
    
    torch.manual_seed(seed)

    loss = CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    losses = []
        
    train_data_loader = DataLoader(data, batch_size=64, shuffle=True)


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

def test(data: Dataset, model: Module, classes: list):

    dataloader = DataLoader(data, batch_size=64, shuffle=True)
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

def tuneKernel(testX: Dataset, X: Dataset, classes: list) -> int:
    accs = []
    accs_classes = {c: [] for c in classes}

    for kernel in range(7):

        model = ConvFashion(16, 16*4, kernel)
        trained_model = train(model, X)

        acc, acc_classes = test(testX, trained_model, classes)
        accs.append(acc)

        for c in classes:
            accs_classes[c].append(acc_classes)

    plt.plot(accs)
    return int(np.argmax(np.array(accs)) + 1)

def tuneChannelRatio(testX: Dataset, X: Dataset, classes: list) -> float:
    accs = []
    accs_classes = {c: [] for c in classes}

    ratios = [1/8, 1/4, 1/2, 1, 2, 4, 8]
    for ratio in ratios:

        model = ConvFashion(16, 16*ratio, 3)
        trained_model = train(model, X)

        acc, acc_classes = test(testX, trained_model, classes)
        accs.append(acc)

        for c in classes:
            accs_classes[c].append(acc_classes)

    plt.plot(accs)
    return ratios[np.argmax(np.array(accs))]

def main():
    classes = ['T-shirt', 'Pants', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Shoe', 'Bag', 'Boot']
    D = FashionMNIST(root = './FashionMNIST', download = True, transform=ToTensor())
    T = FashionMNIST(root = './FashionMNIST_Test',  download=True, train=False, transform=ToTensor())

    #ratio = tuneKernel(D, T, classes)
    #print(f'best Convolution Layers Channel Ratio: {ratio}')

    kernel = tuneKernel(D, T, classes)
    print(f'best Convolution Kernel Size: {kernel}')
    plt.show()

if __name__ == '__main__':
    main()

