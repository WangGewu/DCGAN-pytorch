import torch
import os
import time
import torchvision.transforms as transforms
from torch_tutorial.dcgan.model import Generator,Discriminator
import torchvision
import torch.nn as nn
from tqdm import tqdm
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Config:
    data_path = 'data/'
    num_workers = 4 # 多进程加载数据所用的进程数
    image_size = 96
    batch_size = 256
    max_epoch = 100
    lr1 = 2e-4
    lr2 = 2e-4
    ngf = 64    # 生成器 feature map 数
    ndf = 64    # 判别器 feature map 数
    nz = 100    # 噪声维度
    beta1 = 0.5 # Adam优化器的参数

    save_path='imgs/'
    g_every = 5 # 每 5个batch训练一次生成器
    d_every = 1 # 每 1个batch训练一次判别器
    save_every = 10 # 每 10个batch保存一次
    netd_path = 'checkpoints/discriminator_99.pth'
    netg_path = 'checkpoints/generator_99.pth'

    # 测试
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0  # 噪声均值
    gen_std = 1   # 噪声方差
    gen_img = 'result.png'

opt = Config()

def train(**kwargs):
    print("Start training...")
    tic = time.time()
    for k, v in kwargs.items():
        setattr(opt, k, v)

    transform=transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.CenterCrop(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    dataset=torchvision.datasets.ImageFolder(opt.data_path,transform=transform)
    data_loader=torch.utils.data.DataLoader(dataset,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            drop_last=True
                                            )
    fake_labels = torch.zeros(opt.batch_size).to(device)
    g=Generator(opt).to(device)
    d=Discriminator(opt).to(device)
    optimizer_g=torch.optim.Adam(g.parameters(),lr=opt.lr1,betas=(opt.beta1, 0.999))
    optimizer_d=torch.optim.Adam(d.parameters(),lr=opt.lr2,betas=(opt.beta1, 0.999))
    criterion=nn.BCELoss()

    fix_noises=torch.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    noises=torch.randn(opt.batch_size,opt.nz,1,1).to(device)

    for epoch in range(100):
        total_index=len(data_loader)
        print('epoch: %d'%(epoch+1))
        for i,(img,_) in enumerate(tqdm(data_loader)):
            true_labels = torch.ones(img.size()[0]).to(device)
            real_img=img.to(device)
            if(i%opt.d_every)==0:
                optimizer_d.zero_grad()
                output=d(real_img)
                loss_true=criterion(output,true_labels)
                loss_true.backward()
                noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img=g(noises)
                output=d(fake_img)
                loss_false=criterion(output,fake_labels)
                loss_false.backward()
                optimizer_d.step()
            if(i%opt.g_every)==0:
                optimizer_g.zero_grad()
                noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = g(noises)
                output = d(fake_img)
                loss_g=criterion(output,true_labels)
                loss_g.backward()
                optimizer_g.step()
        if (epoch+1)%opt.save_every==0:
            fix_fake_imgs=g(fix_noises)
            torchvision.utils.save_image(fix_fake_imgs.data[:64], '%s/%s.png' % (opt.save_path, epoch), normalize=True,
                                range=(-1, 1))
            torch.save(d.state_dict(),'checkpoints/discriminator_%s.pth' % epoch)
            torch.save(g.state_dict(),'checkpoints/generator_%s.pth' % epoch)

def generate(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    g,d=Generator(opt),Discriminator(opt)
    noises = torch.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std)
    noises = noises.to(device)
    map_location = lambda storage, loc: storage
    g.load_state_dict(torch.load(opt.netg_path))
    d.load_state_dict(torch.load(opt.netd_path))
    g=g.to(device)
    d=d.to(device)
    fake_img=g(noises)
    scores=d(fake_img)
    indexs=scores.topk(opt.gen_num)[1]
    result=[]
    for i in indexs:
        result.append(fake_img.data[i])
    result=torch.stack(result)
    torchvision.utils.save_image(result,filename=opt.gen_img,normalize=True, range=(-1, 1))
if __name__ == '__main__':
    train()
    #generate()