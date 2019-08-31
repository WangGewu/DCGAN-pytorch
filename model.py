#定义了生成器和判别器的类

import torch
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self,opt):
        super(Generator, self).__init__()
        ngf=opt.ngf
        self.main=nn.Sequential(
            nn.ConvTranspose2d(opt.nz,ngf*8,4,1,0,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2,ngf,4,2,1,bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf,3,5,3,1,bias=False),
            nn.Tanh()
            # 输出形状：3 x 96 x 96
        )
    def forward(self, input):
        output=self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self,opt):
        ndf=opt.ndf
        super(Discriminator, self).__init__()
        self.main=nn.Sequential(
            nn.Conv2d(3,ndf,5,3,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*2,ndf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*4,ndf*8,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*8,1,4,1,0,bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        output=self.main(input).view(-1)
        return output