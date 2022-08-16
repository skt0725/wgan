import os
from scipy import io
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from time import time
import argparse
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="RMSProp: learning rate")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--clipping_param", type=float, default = 0.01, help="extent of clipping")
parser.add_argument("--n_critic", type=int, default=5, help="number of iterations of the critic per generator iteration")
opt = parser.parse_args()

# data load & preprocess
data_dir = "./data/Low_High_CT_mat_slice"
img_path = glob.glob(os.path.join(data_dir, '**/*.mat'), recursive=True)
low = []
high = []
for path in img_path:
    img_mat = io.loadmat(path)
    img_low_high = img_mat.get('imdb')
    low.append(img_low_high[0][0][0])
    high.append(img_low_high[0][0][1])

class AAPM(Dataset):
    def __init__(self, transform = None):
        self.class_len = len(low)
        self.data = low+high
        self.transform = transform


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.transform is not None:
            img = self.transform(self.data[idx])
            img_min, img_max = img.min(), img.max()
            img = (img-img_min)/(img_max-img_min) # min-max scale
        if idx < self.class_len:
            return img, low
        else:
            return img, high

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(opt.img_size),
    transforms.ToTensor(),
])

dataset = AAPM(transform = transform)
dataloader = DataLoader(dataset=dataset, batch_size = opt.batch_size, shuffle=True, drop_last = True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(opt.latent_dim, 1024)
        self.conv1= nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 8, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True)
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        output = self.fc(z)
        output = output.view(output.size(0), 1024, 1, 1)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        return output

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(inplace = True)
        )
        self.conv2= nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace = True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace = True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace = True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 1, 4, 1, 0),
        )

    def forward(self, z):
        output = self.conv1(z)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        return output.squeeze()

generator = Generator().to(device)
critic = Critic().to(device)
gen_optimizer = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
cri_optimizer = torch.optim.RMSprop(critic.parameters(), lr=opt.lr)
summary = SummaryWriter('logs/')

for epoch in range(opt.n_epochs):
    start_time = time()
    for i, (image, label) in enumerate(dataloader):
        real_image = image.to(device)
        z = torch.randn((opt.batch_size, opt.latent_dim)).to(device)
        fake_image = generator(z).detach()
        # train critic
        cri_optimizer.zero_grad()
        real_output = torch.sum(critic(real_image))
        fake_output = torch.sum(critic(fake_image))
        loss_c = (-real_output + fake_output)/opt.batch_size
        loss_c.backward()
        cri_optimizer.step()
        for param in critic.parameters():
            param.data.clamp_(-opt.clipping_param, opt.clipping_param)
        
        # train generator
        if i%opt.n_critic:
            gen_optimizer.zero_grad()
            z = torch.rand((opt.batch_size, opt.latent_dim)).to(device)
            loss_g = -torch.sum(critic(generator(z)))/opt.batch_size
            loss_g.backward()
            gen_optimizer.step()
        if i%100 == 0:
            print(i)

    result_save_dir = f"./result"
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
    if epoch == 0:
        fake_image = fake_image.view(opt.batch_size, 1, opt.img_size,opt.img_size
        real_image = real_image.view(opt.batch_size, 1, opt.img_size,opt.img_size)
        save_image(fake_image, "./fake.png", nrow=4)
        save_image(real_image, "./real.png", nrow=4)
    if (epoch+1) % 2 == 0:
        fake_image = fake_image.view(opt.batch_size, 1, opt.img_size,opt.img_size)
        save_image(fake_image, os.path.join(result_save_dir, f"{epoch}.png"), nrow=4)
    t = time()-start_time
    print(f'Epoch {epoch}/{opt.n_epochs} || discriminator loss={loss_c:.4f}  || generator loss={loss_g:.4f} || time {t:.3f}')
    summary.add_scalar("cri", loss_c, epoch)
    summary.add_scalar("gen", loss_g, epoch)
summary.close()
torch.save(critic.state_dict(), os.path.join(result_save_dir, "critic.pth"))
torch.save(generator.state_dict(), os.path.join(result_save_dir, "generator.pth"))



        
