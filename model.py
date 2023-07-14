import torch.nn as nn

class ConvFashion(nn.Module):
    def __init__(self, channels_dim1, channels_dim2, kernel_dim):
        super(ConvFashion, self).__init__()

        #in: 1x28x28 out: output_channelsx28x28
        self.conv1 = nn.Conv2d(1, channels_dim1, kernel_dim)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        #in: output_channelsx28x28 out: output_channelsx28x28
        self.conv2 = nn.Conv2d(channels_dim1, channels_dim2, kernel_dim)
        self.act2 = nn.ReLU()

        #in: output_channelsx28x28 out: output_channelsx12x12
        self.pool2 = nn.MaxPool2d(2);

        #in: output_channelsx12x12 out: output_channels*12*12
        self.flat = nn.Flatten()

        #in: output_channels*144 out: 512
        self.fc3 = nn.Linear(channels_dim2*144, 512)
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

