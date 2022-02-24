import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --- hyperparameters ---
N_EPOCHS = 15
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
BATCH_SIZE_DEV = 100
LR = 0.01


# --- fixed constants ---
NUM_CLASSES = 24
DATA_DIR = '../data/sign_mnist_%s'


# --- Dataset initialization ---

# We transform image files' contents to tensors
# Plus, we can add random transformations to the training data if we like
# Think on what kind of transformations may be meaningful for this data.
# Eg., horizontal-flip is definitely a bad idea for sign language data.
# You can use another transformation here if you find a better one.
train_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])

train_set = datasets.ImageFolder(DATA_DIR % 'train', transform=train_transform)
dev_set = datasets.ImageFolder(DATA_DIR % 'dev',   transform=test_transform)
test_set = datasets.ImageFolder(DATA_DIR % 'test',  transform=test_transform)


# Create Pytorch data loaders
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)
dev_loader = torch.utils.data.DataLoader(
    dataset=dev_set, batch_size=BATCH_SIZE_TEST, shuffle=True)


# --- model ---
class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout2d(0.3))
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout2d(0.3))
        )
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


# --- set up ---
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = CNN().to(device)


# WRITE CODE HERE
# L2-norm can be implemented with weight decay parameter
loss_function = nn.NLLLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)
# optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=1e-5)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)

lambda1 = .5

# --- training ---
for epoch in range(N_EPOCHS):
    train_loss = 0
    train_correct = 0
    total = 0

    best_score = None
    patience = 7
    counter = 0
    delta = 0.005

    for batch_num, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        out = model(data)
        loss = loss_function(out, target)
        _, predicted = torch.max(out, 1)

        # L1-norm implementation
        # all_linear1_params = torch.cat(
        #     [x.view(-1) for x in model.layer1.parameters()])
        # l1_regularization = lambda1 * torch.norm(all_linear1_params, 1)

        # loss += l1_regularization

        loss.backward()
        optimizer.step()

        corr = (predicted == target).sum().item()/target.size()[0]

        train_loss = loss.item()
        train_correct += corr
        total += 1

        print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' %
              (epoch, batch_num, len(train_loader), train_loss / (batch_num + 1),
               100. * train_correct / total, train_correct, total))

        score = -loss
        if best_score is None:
            best_score = score
        elif score < best_score + delta:
            counter += 1
            if counter >= patience:
                break
        # WRITE CODE HERE
        # Please implement early stopping here.
        # You can try different versions, simplest way is to calculate the dev error and
        # compare this with the previous dev error, stopping if the error has grown.

# --- test ---
test_loss = 0
test_correct = 0
total = 0

with torch.no_grad():
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        out = model(data)
        loss = loss_function(out, target)
        _, predicted = torch.max(out.data, 1)

        total += 1
        test_correct += (predicted == target).sum().item()/target.size()[0]
        test_loss = loss.item()

        print('Evaluating: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' %
              (batch_num, len(test_loader), test_loss / (batch_num + 1),
               100. * test_correct / total, test_correct, total))


# Optimizers
# SGD - 90%
# Adam - 80%
# RMSProp - 85%

# Regularization schemes
# L1-norm - 1% increase
# L2-norm - 5% increase
# dropout - 10% decrease
