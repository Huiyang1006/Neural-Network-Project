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


def encode(letter):
    encoded = [0 for i in range(27)]
    index = list(letters_dict.values()).index(letter)
    encoded[index] = 1
    return encoded


def generate(ch, model):
    input_ch = [encode(ch)]
    # input_ch = encode(ch)
    hc = model.initial_hidden()
    generated_name = ch

    for i in range(11):
        torch_input_ch = torch.tensor(input_ch)
        output, hc = model(torch_input_ch, hc)
        # print(output.shape)
        output = output[-1].detach()
        # print(output.shape)
        '''select 3 letters with highest possibilities'''
        indexes = list(np.argpartition(output, -3)[-3:].numpy())
        # print(indexes)
        '''randomly choose one letter'''
        chosen = np.random.choice(indexes)
        '''if the chosen one is EON then terminates, o.w. loop until length of name is 11.'''
        if chosen == 26 or len(generated_name) == 11:
            break
        else:
            generated_name += letters_dict.get(chosen)
            input_ch.append(encode(letters_dict.get(chosen)))

    return generated_name


# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        # self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()

        self.hn = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        self.cn = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)

        self.lstm = nn.LSTM(
            batch_first=True,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            # dropout=0.1,
        )

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def initial_hidden(self):
        hc = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device),
              torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device))
        return hc

    def forward(self, x, hc):
        x, (h1, c1) = self.lstm(x.float())
        x = self.relu(x)
        x = self.fc(x)

        return x, (h1, c1)


def main():
    parser = argparse.ArgumentParser(description='CS559 hw7')
    parser.add_argument('--batch-size', type=int, default=100, help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100,
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=655490960, help='random seed (default: 655490960)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    args = parser.parse_args()

    path = './0702-655490960-Zhao.pt'
    # path = './train.pt'

    model = LSTM(input_size=27, hidden_size=64, output_size=27, num_layers=1, batch_size=args.batch_size).to(device)
    saved = torch.load(path)
    model.load_state_dict(saved)

    model.eval()

    num_names = 20
    generated_name_a = []
    generated_name_e = []

    for i in range(num_names):
        generated_name_a.append(generate('a', model))
        generated_name_e.append(generate('x', model))

    print('Feed a: ' + str(generated_name_a))
    print('Feed x: ' + str(generated_name_e))

    inputted = input("Enter the letter here: ")

    print(inputted)
    generated_name_input = []
    for i in range(num_names):
        generated_name_input.append(generate(inputted, model))

    print('Feed ' + inputted + ':' + str(generated_name_input))


if __name__ == '__main__':
    main()
