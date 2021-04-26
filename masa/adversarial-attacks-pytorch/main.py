# https://github.com/RobustBench/robustbench
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
from utils import l2_distance
from models import *

# https://github.com/Harry24k/adversarial-attacks-pytorch
import torchattacks
print("torchattacks %s"%(torchattacks.__version__))
from torchattacks.attack import Attack

import torchvision
#from cifar10_models.resnet import resnet18

# import random things
import datetime
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
warnings.filterwarnings(action='ignore')

# https://github.com/bethgelab/foolbox
import foolbox as fb
print("foolbox %s"%(fb.__version__))

# https://github.com/IBM/adversarial-robustness-toolbox
import art
import art.attacks.evasion as evasion
from art.classifiers import PyTorchClassifier
print("art %s"%(art.__version__))

# run auto attack test
if __name__ == '__main__':
    # PREDEFINED EPSILON
    epsilon = 8/255
    N = 64

    # define data
    images, labels = load_cifar10(n_examples=N)
    img = torchvision.utils.make_grid(images).permute(1,2,0)
    plt.imsave('results/original_cifar10',img)
    images = images.cuda()
    labels = labels.cuda()

    # define model
    model = ResNet18()
    model = model.to('cuda')
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['net'])

    # true accuracy
    start = datetime.datetime.now()
    acc = clean_accuracy(model, images.cuda(),labels.cuda())
    end = datetime.datetime.now()
    print('- True Acc: {} ({} ms)'.format(acc, int((end-start).total_seconds()*1000)))

    # define auto attack
    version = 'standard'
    print("- AutoAttack (%s)"%(version))
    atk = torchattacks.AutoAttack(model, norm='Linf', eps=epsilon, seed=5, version=version)
    start = datetime.datetime.now()
    adv_images = atk(images, labels)
    end = datetime.datetime.now()
    acc = clean_accuracy(model, adv_images, labels)
    print('- Robust Acc: {} ({} ms)'.format(acc, int((end-start).total_seconds()*1000)))
    auto_img = torchvision.utils.make_grid(adv_images).permute(1,2,0).detach().cpu().numpy()
    plt.imsave('results/auto_cifar10',auto_img)

    # define PGD attack
    version = 'standard'
    print("- PGD (%s)"%(version))
    atk = torchattacks.PGD(model, eps=epsilon)
    start = datetime.datetime.now()
    adv_images = atk(images, labels)
    end = datetime.datetime.now()
    acc = clean_accuracy(model, adv_images, labels)
    print('- Robust Acc: {} ({} ms)'.format(acc, int((end-start).total_seconds()*1000)))
    pgd_img = torchvision.utils.make_grid(adv_images).permute(1,2,0).detach().cpu().numpy()
    plt.imsave('results/pgd_cifar10',auto_img)

    # define CW attack
    version = 'standard'
    print("- CW (%s)"%(version))
    atk = torchattacks.CW(model)
    start = datetime.datetime.now()
    adv_images = atk(images, labels)
    end = datetime.datetime.now()
    acc = clean_accuracy(model, adv_images, labels)
    print('- Robust Acc: {} ({} ms)'.format(acc, int((end-start).total_seconds()*1000)))
    cw_img = torchvision.utils.make_grid(adv_images).permute(1,2,0).detach().cpu().numpy()
    plt.imsave('results/cw_cifar10',cw_img)



