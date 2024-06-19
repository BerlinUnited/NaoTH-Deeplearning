import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=24, out_channels=16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.LeakyReLU(inplace=True)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=40, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(40)
        self.relu5 = nn.LeakyReLU(inplace=True)

        # Fully connected layers
        # helpful: https://stackoverflow.com/questions/62993470/how-to-specify-the-input-dimension-of-pytorch-nn-linear
        self.fc = nn.Linear(7 * 10 * 7, 1)  # Output size is 1, change it according to your task

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = x.view(x.size(0), -1)  # Flatten the output of the convolutional layers
        x = self.fc(x)
        return x


# Instantiate the model
model = CustomCNN()
