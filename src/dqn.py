import torch.nn as nn


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 6), stride=(1, 1))
        self.prelu1 = nn.PReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(4, 4))

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(2, 4), stride=(1, 1))
        self.prelu2 = nn.PReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(4, 4))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(2, 4), stride=(1, 1))
        self.prelu3 = nn.PReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 4))

        self.conv4 = nn.Conv2d(64, 64, kernel_size=(2, 4), stride=(1, 1))
        self.prelu4 = nn.PReLU()

        self.fc1 = nn.Linear(in_features=768, out_features=256)
        self.leaky1 = nn.LeakyReLU(0.01)

        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.leaky2 = nn.LeakyReLU(0.01)

        self.fc3 = nn.Linear(in_features=64, out_features=16)
        self.leaky3 = nn.LeakyReLU(0.01)

        self.out = nn.Linear(in_features=16, out_features=2)

    def forward(self, x):
        x = self.maxpool1(self.prelu1(self.conv1(x)))
        x = self.maxpool2(self.prelu2(self.conv2(x)))
        x = self.maxpool3(self.prelu3(self.conv3(x)))
        x = self.prelu4(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.leaky1(self.fc1(x))
        x = self.leaky2(self.fc2(x))
        x = self.leaky3(self.fc3(x))
        x = self.out(x)
        return x
