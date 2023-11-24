# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# testing stuff

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 16, 5)
        self.fc1 = nn.Linear(106624, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def test(epochs, train_loader, test_loader):
    net = Net().to(device).float()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        print(f"Epoch = {epoch + 1}")
        for i, data in tqdm(enumerate(train_loader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # Convert inputs and labels to Float
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:    # print every 10 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0
    