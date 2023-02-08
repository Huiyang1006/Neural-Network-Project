# CS559 Neural Network
# Huiyang Zhao
# UIN 655490960
'''Please put all pictures in validation set in a folder, and put the folder in ./imgFolder.'''

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms

seed = 655490960
random.seed(seed)
path = '0502-655490960-Zhao.pt'
shapes = ['Circle', 'Square', 'Octagon', 'Heptagon', 'Nonagon', 'Star', 'Hexagon', 'Pentagon', 'Triangle']
shapes.sort()


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(16384, 128)
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.conv1(x)  # [3, 200, 200] -> [32, 198, 198]
        x = F.relu(x)
        x = F.max_pool2d(x, 3)  # down sample, [32, 198, 198] -> [32, 66, 66]
        x = self.conv2(x)  # [32, 66, 66] -> [64, 64, 64]
        x = F.relu(x)
        x = F.max_pool2d(x, 4)  # down sample, [64, 64, 64] -> [64, 16, 16]
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)  # [64, 16, 16] -> [16384]
        x = self.fc1(x)  # [16384] -> [128]
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)  # [128] -> [9]
        return x


def test(args, model, device, test_loader, testSet, myfile):
    model.eval()
    tot_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            startIndex = batch_idx * args.test_batch_size
            file = testSet.imgs[startIndex: startIndex + 500]
            data, target = data.to(device), target.to(device)
            output = model(data)
            tot_loss += torch.nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for i in range(len(output)):
                filename = file[i][0]
                output_y = pred[i][0].cpu().numpy()
                myfile.writelines(str(filename)+'\t'+shapes[output_y]+'\n')
            correct += pred.eq(target.view_as(pred)).sum().item()


def main():
    # Testing settings
    parser = argparse.ArgumentParser(description='CS559 hw5 evaluate model')
    parser.add_argument('--test-batch-size', type=int, default=500,
                        help='input batch size for testing (default: 500)')
    parser.add_argument('--seed', type=int, default=655490960, help='random seed (default: 655490960)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    '''Please put all pictures in validation set in a folder, and put the folder in ./imgFolder.'''
    testSet = datasets.ImageFolder('./imgFolder', transform=transform)
    test_loader = torch.utils.data.DataLoader(testSet, batch_size=args.test_batch_size)
    # Load model
    model = Network()
    model.load_state_dict(torch.load(path))
    if torch.cuda.is_available():
        model.cuda()
    # Provide the inference results for all images
    myfile = open('./results.txt', 'w')
    test(args, model, device, test_loader, testSet, myfile)
    myfile.close()


if __name__ == '__main__':
    main()
