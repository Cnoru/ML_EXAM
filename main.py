import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as T
import matplotlib.pyplot as plt

import mltracker as ml



batch_size = 64
num_epochs = 30
learning_rate = 0.01
save_filename = 'inception_model'

mnist_train = dset.MNIST('data', train=True, download=True, transform=T.ToTensor())
loader_train = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
mnist_test = dset.MNIST('data', train=False, download=True, transform=T.ToTensor())
loader_test = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)


class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        # To be implemented
        self.incep1 = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.incep2 = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=1),
            nn.Conv2d(2, 3, kernel_size=3, padding=1)
        )
        self.incep3 = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=1),
            nn.Conv2d(2, 3, kernel_size=5, padding=2)
        )
        self.incep4 = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )

    def forward(self, x):
        # To be implemented
        x1 = self.incep1(x)
        x2 = self.incep2(x)
        x3 = self.incep3(x)
        x4 = self.incep4(x)
        return torch.cat([x1, x2, x3, x4], 1)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            InceptionA(5),

            nn.Conv2d(10, 12, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            InceptionA(12),

            Flatten(),
            nn.Linear(160, 10)
        )

    def forward(self, x):
        return self.net(x)	


    
def train(epoch, model, loss_fn, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(loader_train):
        optimizer.zero_grad()
        output = model(data.cuda())
        loss = loss_fn(output, target.cuda())
        loss.backward()
        optimizer.step()
        if batch_idx % 300 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader_train.dataset),
                100. * batch_idx / len(loader_train), loss))

def test(model, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in loader_test:
        output = model(data.cuda())
        # sum up batch loss
        test_loss += loss_fn(output, target.cuda())
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.cuda().data.view_as(pred)).cpu().sum()

    test_loss /= len(loader_test.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader_test.dataset),
        100. * correct / len(loader_test.dataset)))
    ml.log_metric('Recent loss', float(test_loss))
    ml.log_metric('Accuracy', int(100*correct/len(loader_test.dataset)))

def main(config):
    model = Model().cuda()
    if config.model_path != 'NONE':
        model.load_state_dict(torch.load(config.model_path))
        model.eval()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    
    ml.start_run(version='inception')

    for epoch in range(0, num_epochs):
        train(epoch, model, loss_fn, optimizer)
        ml.log_param('epoch', (epoch+1))
        ml.log_param('num_epochs', num_epochs)
        test(model, loss_fn)
        save_path = ('/gpfs-volume/{}{}'.format(save_filename, epoch+1))
        torch.save(model.state_dict(), save_path)
        ml.log_file(save_path)
        
    ml.end_run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='NONE', help='Reference model set')
    config = parser.parse_args()
    print(config)	
    main(config)
