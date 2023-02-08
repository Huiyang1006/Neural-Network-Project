# CS559 Neural Network
# Huiyang Zhao
# UIN 655490960

import numpy as np
import argparse
import random
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
random.seed(655490960)

EON = '/eon'
letters = []
for ch in 'abcdefghijklmnopqrstuvwxyz':
    letters.append(ch)
letters.append(EON)
print(letters)

letters_dict = {}
for key, value in enumerate(letters):
    letters_dict[key] = value
print(letters_dict)


# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.relu = nn.ReLU()

        self.hn = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        self.cn = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)

        self.lstm = nn.LSTM(
            batch_first=True,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

        self.fc = nn.Linear(self.hidden_size, self.output_size)
        # self.linear1 = nn.Linear(self.hidden_size, 64)
        # self.linear2 = nn.Linear(64, self.output_size)

    def initial_hidden(self):
        hc = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device),
              torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device))
        return hc

    def forward(self, x, hc):
        x, (h1, c1) = self.lstm(x.float(), hc)
        x = self.relu(x)
        x = self.fc(x)
        # x = self.linear1(x)
        # x = self.linear2(x)

        return x, (h1, c1)


def encode(letter):
    encoded = [0 for i in range(27)]
    index = list(letters_dict.values()).index(letter)
    encoded[index] = 1
    return encoded


def preprocess():
    file = open('names.txt', 'r')
    names = file.readlines()

    input_names = []
    output = []

    for name in names:
        name_list = []
        for ch in name.replace('\n', '').lower():
            name_list.append(ch)
        while len(name_list) < 11:
            name_list.append(EON)
        label = name_list[1:]
        label.append(EON)

        input_names.append(torch.tensor([encode(ch) for ch in name_list]))
        output.append(torch.tensor([encode(ch) for ch in label]))

    return input_names, output


def train(args, model, device, train_loader, optimizer, epoch, error_array):
    model.train()
    tot_loss = 0
    correct = 0
    hc = model.initial_hidden()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, hc = model(data, hc)
        hc = tuple([each.data for each in hc])
        target = target.argmax(axis=2)
        temp = torch.transpose(output, 2, 1)
        loss = torch.nn.CrossEntropyLoss()(temp, target)
        loss.backward(retain_graph=True)
        optimizer.step()

        pred = output.argmax(dim=2, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        tot_loss = tot_loss + loss.item()

    print('End of Epoch: {}'.format(epoch))
    print('Training Loss: {:.6f}'.format(tot_loss / (len(train_loader))))
    error_array.append(tot_loss / len(train_loader))


class lstm_dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label


def main():
    data, labels = preprocess()

    parser = argparse.ArgumentParser(description='CS559 hw7')
    parser.add_argument('--batch-size', type=int, default=100, help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100,
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=120, help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.6, help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=655490960, help='random seed (default: 655490960)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    dataset = lstm_dataset(data, labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    model = LSTM(input_size=27, hidden_size=64, output_size=27, num_layers=1, batch_size=args.batch_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=6, gamma=args.gamma)

    error_array = []

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, data_loader, optimizer, epoch, error_array)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "train.pt")

    epoch_array = range(args.epochs)

    plt.figure()
    plt.title('epoch vs. Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(epoch_array, error_array)
    plt.savefig('epoch_vs_loss')
    plt.show()


if __name__ == '__main__':
    main()
