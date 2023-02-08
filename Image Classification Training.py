# CS559 Neural Network
# Huiyang Zhao
# UIN 655490960

import argparse
import random
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os

seed = 655490960
random.seed(seed)
'''Please create an imgFolder and put the output folder that contains pictures in it.'''
file_dir = './imgFolder/output/'
shapes = {'Circle', 'Square', 'Octagon', 'Heptagon', 'Nonagon', 'Star', 'Hexagon', 'Pentagon', 'Triangle'}
train_error_array = []
test_error_array = []
train_accuracy_array = []
test_accuracy_array = []


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(16384, 128)
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.conv1(x)  # [3, 200, 200] -> [32, 198, 198]
        x = F.relu(x)
        x = F.max_pool2d(x, 3)  # down sample, [32, 198, 198] -> [32, 66, 66]
        x = self.conv2(x)  # [32, 66, 66] -> [64, 64, 64]
        x = F.relu(x)
        x = F.max_pool2d(x, 4)  # down sample, [64, 64, 64] -> [64, 16, 16]
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # [64, 16, 16] -> [16384]
        x = self.fc1(x)  # [16384] -> [128]
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)  # [128] -> [9]
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    tot_loss = 0
    correct = 0
    # for batch_idx, (data, target) in enumerate(train_loader):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # print(target)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        tot_loss = tot_loss + loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), tot_loss / (batch_idx + 1),
                       100.0 * correct / ((batch_idx + 1) * args.batch_size)))

    print('End of Epoch: {}'.format(epoch))
    print('Training Loss: {:.6f}, Training Accuracy: {:.2f}%'.format(
        tot_loss / (len(train_loader)), 100.0 * correct / (len(train_loader) * args.batch_size)))
    train_error_array.append(tot_loss / (len(train_loader)))
    train_accuracy_array.append(100.0 * correct / (len(train_loader) * args.batch_size))


def test(args, model, device, test_loader):
    model.eval()
    tot_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            tot_loss += torch.nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('Test Loss: {:.6f}, Test Accuracy: {:.2f}%'.format(
        tot_loss / (len(test_loader)), 100.0 * correct / (len(test_loader) * args.test_batch_size)))
    test_error_array.append(tot_loss / (len(test_loader)))
    test_accuracy_array.append(100.0 * correct / (len(test_loader) * args.test_batch_size))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='CS559 hw5')
    parser.add_argument('--batch-size', type=int, default=100, help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=500,
                        help='input batch size for testing (default: 500)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.6, help='Learning rate step gamma (default: 0.6)')
    parser.add_argument('--seed', type=int, default=655490960, help='random seed (default: 655490960)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    '''Comment this paragraph to load saved trainset and testset.'''
    trainset = datasets.ImageFolder('./trainSet', transform=transform)
    torch.save(trainset, 'trainSet.pt')
    testset = datasets.ImageFolder('./testSet', transform=transform)
    torch.save(testset, 'testSet.pt')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size)
    '''Uncomment this paragraph to load saved trainset and testset.'''
    # train_loader = torch.utils.data.DataLoader(torch.load('trainSet.pt'), batch_size=args.batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(torch.load('testset.pt'), batch_size=args.test_batch_size)

    model = Network().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=10, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        if test_error_array[-1] < 0.25:
            break
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "0502-655490960-Zhao.pt")
        # torch.save({
        #     'epoch': range(len(train_error_array)),
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'train loss': train_error_array,
        #     'test loss': test_error_array,
        #     'train accuracy': train_accuracy_array,
        #     'test accuracy': test_accuracy_array
        # }, "checkpoint.pt")


if __name__ == '__main__':
    '''Reads the files and creates variables for training and testing.'''
    '''Comment this paragraph to create separated train/test folders'''
    if not os.path.exists('trainSet/'):
        for shape in shapes:
            os.makedirs('trainSet/' + shape + '/')
    if not os.path.exists('testSet'):
        for shape in shapes:
            os.makedirs('testSet/' + shape + '/')
    for shape in shapes:
        shape_list = [s for s in os.listdir(file_dir) if s.startswith(shape)]
        random.shuffle(shape_list)
        for item in shape_list[:8000]:
            shutil.copy(file_dir + item, 'trainSet/' + shape + '/')
        for item in shape_list[8000:]:
            shutil.copy(file_dir + item, 'testSet/' + shape + '/')

    main()

    epoch_array = range(len(train_error_array))

    plt.figure()
    plt.title('epoch vs. loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(epoch_array, train_error_array, label='training set')
    plt.plot(epoch_array, test_error_array, label='testing set')
    plt.legend()
    plt.savefig('epoch vs. loss.pdf')
    plt.show()

    plt.figure()
    plt.title('epoch vs. accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(epoch_array, train_accuracy_array, label='training set')
    plt.plot(epoch_array, test_accuracy_array, label='testing set')
    plt.legend()
    plt.savefig('epoch vs. accuracy.pdf')
    plt.show()
