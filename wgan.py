import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from time import time
import argparse
from torch.utils.tensorboard import SummaryWriter


# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="RMSProp: learning rate")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--clipping_param", type=float, default = 0.01, help="extent of clipping")
parser.add_argument("--n_critic", type=int, default=5, help="number of iterations of the critic per generator iteration")
opt = parser.parse_args()

# data load & preprocess
data_dir = "./data/Low_High_CT_mat_slice"
data_subfolder = ['L067', 'L109', , 'L143', 'L192', 'L286', 'L291', 'L310', 'L333', 'L506']
# architecture : similar to lsgan for comparison
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(opt.latent_dim, 1024)
        self.conv1= nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        output = self.fc(z)
        output = output.view(output.size(0), 1024, 1, 1)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        return output

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(inplace = True)
        )
        self.conv2= nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace = True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace = True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 1, 4, 1, 0)
        )
    def forward(self, z):
        output = self.conv1(z)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        return output.view(-1, 1)

generator = Generator().to_device()
critic = Critic().to_device()
gen_optimizer = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
cri_optimizer = torch.optim.RMSprop(critic.parameters(), lr=opt.lr)

for epoch in range(opt.n_epochs):
    for i in range(opt.n_critic):
        
