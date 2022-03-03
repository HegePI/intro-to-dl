import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Hyperparameters
N_EPOCHS = 15
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
BATCH_SIZE_DEV = 100
LR = 0.01


# --- fixed constants ---
NUM_CLASSES = 24
DATA_DIR = '../data/sign_mnist_%s'


class Model(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
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

model = Model().to(device)

# WRITE CODE HERE

loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
