import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os

import news_dataset

# Hyperparameters
N_EPOCHS = 15
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
BATCH_SIZE_DEV = 100
LR = 0.01


# --- fixed constants ---
NUM_CLASSES = 24
DATA_DIR = "final-project/data"
TOPICS = "final-project/topic_codes.txt"

data = news_dataset.newsDataset(DATA_DIR, TOPICS)

train_data, test_data = torch.utils.data.random_split(
    data, [math.floor(len(data)*(2/3)), math.ceil(len(data)*(1/3))])


# Create Pytorch data loaders
train_data_loader = torch.utils.data.DataLoader(
    dataset=train_data, shuffle=True)

test_data_loader = train_data_loader = torch.utils.data.DataLoader(
    dataset=test_data, shuffle=True)


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

for batch_num, (data, target) in enumerate(train_data_loader):
    print(data)
