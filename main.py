from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
from loss import FocalLoss

from torch.utils.tensorboard import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

### Data Initialization and Loading
from data import initialize_data, data_transforms # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set


train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Create SummaryWriter instance
tb = SummaryWriter()
images, labels = next(iter(train_loader))
images = images.to(device)
labels = labels.to(device)
grid = torchvision.utils.make_grid(images)

tb.add_image('images', grid)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script



##############
# CHANGE NET HERE
##############
from resnet_model_2 import Net
model = Net()
print(model)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)
tb.add_graph(model, images)
tb.close()

##############
# CHANGE OPTIMIZER 
##############
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("Using ", optimizer, " optimizer")

def train(epoch):
    total_loss = 0
    model.train()
    model.to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        optimizer.zero_grad()
        output = model(data)
        ##############
        # CHANGE LOSS FUNCTION 
        ##############
        # loss = F.nll_loss(output, target)
        loss_func = nn.CrossEntropyLoss()
        # loss_func = FocalLoss()

        loss = loss_func(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
        total_loss += loss.item()
        
    return total_loss

def validation():
    model.eval()
    model.to(device)
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data, volatile=True).to(device), Variable(target).to(device)
        output = model(data)
        # validation_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
        # loss_func = nn.CrossEntropyLoss()
        loss_func = FocalLoss()
        validation_loss += loss_func(output, target).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    accuracy = correct / len(val_loader.dataset) *100
    return correct, accuracy


for epoch in range(1, args.epochs + 1):
    epoch_loss = train(epoch)
    epoch_correct, epoch_accuracy = validation()

    # tensorboard data
    tb.add_scalar('Loss', epoch_loss, epoch)
    tb.add_scalar('Number Correct', epoch_correct, epoch)
    tb.add_scalar('Validation Accuracy', epoch_accuracy, epoch)

#################
# CHANGE MODEL DIRECTORY HERE
#################
    model_file = 'resnet34_data_aug/model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to '
                                                                                                         'generate the Kaggle formatted csv file')
