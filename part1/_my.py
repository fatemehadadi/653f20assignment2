import numpy as np
import matplotlib as mpl

mpl.use("TKAgg")
import matplotlib.pyplot as plt

import sys
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# personal import
import torch.nn as nn
import torch.nn.functional as func
from torch.optim.lr_scheduler import StepLR
from random import shuffle
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(28, 128)  # 1 layer:- 28 input 128 o/p
        self.activation1 = nn.ReLU()  # Defining Regular linear unit as activation
        self.layer2 = nn.Linear(128, 64)  # 2 Layer:- 128 Input and 64 O/p
        self.activation2 = nn.Tanh()  # Defining Regular linear unit as activation
        self.layer3 = nn.Linear(64, 10)  # 3 Layer:- 64 Input and 10 O/P as (0-9)
        self.activation3 = nn.LogSoftmax(dim=1)
        self.layer4 = nn.Linear(280, 10)  # 4 Layer:- 128 Input and 10 O/P as (0-9)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = torch.flatten(x, 2)
        x = torch.flatten(x, 1)
        x = self.layer4(x)
        output = func.log_softmax(x, dim=1)
        # print(output.shape)
        return output


class Buffer:
    def __init__(self, limit):
        self.b = []
        self.limit = limit

    def add(self, data):
        if len(self.b) < self.limit:
            self.b.append(data)
        else:
            self.b = self.b[1:] + [data]

def learn2(model, optimizer, buffer, epochs, n_batches):
    model.train()
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=60000, epochs=epochs)
    for epoch in range(epochs):
        list = buffer.b
        shuffle(list)
        n = len(list)
        m = int(n/n_batches)
        for j in range(n_batches):
            for x in range(m):
                optimizer.zero_grad()
                loss = func.nll_loss(list[j*m+x][1], list[j*m+x][0])
                loss.backward()
                optimizer.step()
    return

def main():
    # Task setup block starts
    # Do not change
    torch.manual_seed(1000)
    dataset = datasets.MNIST(
        root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    # Task setup block end

    # Learner setup block
    seed = 0 if len(sys.argv) == 1 else int(sys.argv[1])
    torch.manual_seed(seed)  # do not change. This is for learners randomization
    ####### Start
    lr = 1
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.7)
    epochs = 5
    n_batches = 20
    T = 200
    buffer = Buffer(T)
    ####### End

    # Experiment block starts
    errors = []
    checkpoint = 1000
    correct_pred = 0
    for idx, (image, label) in enumerate(loader):
        # Observe
        label = label.to(device=device)
        image = image.to(device=device)
        # Make a prediction of label
        ####### Start
        # Replace the following statement with your own code for
        # making label prediction
        model.eval()
        pred = model(image)
        pred_label = pred.argmax(dim=1, keepdim=True)
        ####### End

        # Evaluation
        correct_pred += (pred_label == label).sum()

        # Learn
        ####### Start
        # Here goes your learning update
        model.train()
        torch.autograd.set_detect_anomaly(True)
        buffer.add((label, image))
        losses = 0
        #print(idx)
        if idx % T == 0 and idx >= T:
            for epoch in range(epochs):
                list = buffer.b
                shuffle(list)
                b = len(list)   # buffer size
                size_batches = int(b / n_batches)    # size of each mini-batch
                for ind, (l, img) in enumerate(list):
                    model.train()
                    optimizer.zero_grad()
                    p = model(img)
                    loss = func.nll_loss(p, l)
                    losses = loss + losses
                    if (ind+1) % size_batches == 0:
                        losses.backward(retain_graph=True)
                        optimizer.step()
                        scheduler.step()
                        losses = 0


        ####### End

        # Log
        if (idx + 1) % checkpoint == 0:
            error = float(correct_pred) / float(checkpoint) * 100
            print(error)
            errors.append(error)
            correct_pred = 0

            plt.clf()
            plt.plot(range(checkpoint, (idx + 1) + checkpoint, checkpoint), errors)
            plt.ylim([0, 100])
            plt.pause(0.001)
    name = sys.argv[0].split('.')[-2].split('_')[-1]
    data = np.zeros((2, len(errors)))
    data[0] = range(checkpoint, 60000 + 1, checkpoint)
    data[1] = errors
    np.savetxt(name + str(seed) + ".txt", data)
    plt.show()


if __name__ == "__main__":
    main()
