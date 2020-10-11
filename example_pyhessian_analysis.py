#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

from __future__ import print_function

import json
import os
import sys

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import *
from density_plot import get_esd_plot
from models.resnet import resnet
from pyhessian import hessian

import pickle

# Model
class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        
        self.L1 = nn.Linear(784, 512)
        self.L2 = nn.Linear(512, 512)
        self.L3 = nn.Linear(512, 10)
        self.relu = nn.PReLU()
        self.tanh = nn.Tanh()
        self.Softmax = nn.Softmax(dim = 1)

        nn.init.orthogonal_(self.L1.weight)
        nn.init.orthogonal_(self.L2.weight)
        nn.init.orthogonal_(self.L3.weight)

    def forward(self, x):
        x = self.L1(x)
        x = self.relu(x)

        x = self.L2(x)
        x = self.relu(x)

        logits = self.L3(x)
        probs = self.Softmax(logits)

        return probs

def GetHessianEig(path, hessian_dataloader):
    
    criterion = nn.CrossEntropyLoss()  # label loss
    
    # get model
    model = MLP()

    if args.cuda:
        model = model.cuda()
    model = torch.nn.DataParallel(model)
    check_point = torch.load(path)
    model.load_state_dict(check_point.state_dict)
    
    ######################################################
    # Begin the computation
    ######################################################

    # turn model to eval mode
    model.eval()
    if batch_num == 1:
        hessian_comp = hessian(model,
                               criterion,
                               data=hessian_dataloader,
                               cuda=args.cuda)
    else:
        hessian_comp = hessian(model,
                               criterion,
                               dataloader=hessian_dataloader,
                               cuda=args.cuda)

    print(
        '********** finish data londing and begin Hessian computation **********')

    top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=3)

    return top_eigenvalues

# Settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument(
    '--mini-hessian-batch-size',
    type=int,
    default=200,
    help='input batch size for mini-hessian batch (default: 200)')
parser.add_argument('--hessian-batch-size',
                    type=int,
                    default=200,
                    help='input batch size for hessian (default: 200)')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    help='random seed (default: 1)')
parser.add_argument('--batch-norm',
                    action='store_false',
                    help='do we need batch norm or not')
parser.add_argument('--residual',
                    action='store_false',
                    help='do we need residual connect or not')

parser.add_argument('--cuda',
                    action='store_false',
                    help='do we use gpu or not')
parser.add_argument('--resume',
                    type=str,
                    default='',
                    help='get the checkpoint')

args = parser.parse_args()
# set random seed to reproduce the work
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

for arg in vars(args):
    print(arg, getattr(args, arg))

# get dataset
transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean = (0.5), std = (0.5))
                    ])
train_dataset = datasets.MNIST(root = './', train = True, transform = transform, download = True)
test_dataset = datasets.MNIST(root = './', train = False, transform = transform, download = True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1000, shuffle = False)

##############
# Get the hessian data
##############
assert (args.hessian_batch_size % args.mini_hessian_batch_size == 0)
assert (60000 % args.hessian_batch_size == 0)
batch_num = args.hessian_batch_size // args.mini_hessian_batch_size

if batch_num == 1:
    for inputs, labels in train_loader:
        hessian_dataloader = (inputs, labels)
        break
else:
    hessian_dataloader = []
    for i, (inputs, labels) in enumerate(train_loader):
        hessian_dataloader.append((inputs, labels))
        if i == batch_num - 1:
            break

# Saved Models Path
model_path = './LinearnetworkCheckpoints/'
num_epochs = 15

Eig = {}
for i in range(num_epochs):
    curr_checkpoint_path = model_path+'model_'+str(i+1)+'.pth.tar'
    Eigenvalues = []
    for j in range(batch_num):
        Eigenvalues.append(GetHessianEig(curr_checkpoint_path, hessian_dataloader[j]))
    Eig[str(i+1)] = Eigenvalues
        
Eigenvalues = np.array(Eigenvalues)
EigFile = open('./EigenValues', 'wb') 
pickle.dump(Eig, EigFile) 
geeky_file.close()
