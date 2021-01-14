# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:27:28 2020

@author: ivpcl
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:36:27 2020

@author: ivpcl
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:20:44 2020

@author: ivpcl
"""

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from PIL import ImageOps
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from IPython.display import clear_output




def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')
    
    
    




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(107648, 2048)
        self.fc3 = nn.Linear(2048, 512)
        self.fc5 = nn.Linear(512, 16)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #print(x.size())
        x = x.view(-1, 107648)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, training=self.training)
        x = self.fc5(x)
        return x





# def train(epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(loader):
#         # data, target = data.cuda(), target.cuda()
#         data, target = data.to(torch.device('cuda')), target.to(torch.device('cuda'))
#         data, target = Variable(data), Variable(target)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = nn.CrossEntropyLoss()
#         output = loss(output, target)
#         output.backward()
#         optimizer.step()
#         if batch_idx %1700 == 0:
#             clear_output()
#         if batch_idx % 100 == 0:
#              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                  epoch, batch_idx * len(data), len(loader.dataset),
#                  100. * batch_idx / len(loader), output.data[0]))
#         #    print()
        
        
        
    
    

    
    
def main():

    
        # for ix in idx:
        #     root = "./03_emnist_data/"+ix
        #     for folder in os.listdir(root):
        #         path = root +'/'+folder
        #     if os.path.isdir(path):
        #         print(path)
        #         for name in os.listdir(path):
        #         ipath = os.path.join(path, name)
        #         #print(ipath)
        #         img = cv2.imread(ipath,0)
        #         img = 255 - img
        #         saveto = root+'/'+name
        #         cv2.imwrite(saveto, img)
        
    mydata = ImageFolder(root="D:\\Ugesh\\EMNIST_GITHUB_Hexadecemial\\", transform=ToTensor(), loader=pil_loader)
    loader = DataLoader(mydata, batch_size=10, shuffle=True, num_workers=2)
    
    if torch.cuda.is_available():
        print("GPU avalible")
        print("GPU Model: ", torch.cuda.get_device_name(0))
        print("Num GPU's: ", torch.cuda.device_count())
        print("Current GPU: ", torch.cuda.current_device())
    else:
        print("GPU not avalible")
    
    
    model = Net()
    model.to(torch.device('cuda'))
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, eps=1e-4)
    
    for epoch in range(5):
        model.train()
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(torch.device('cuda')), target.to(torch.device('cuda'))
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()
            output = loss(output, target)
            output.backward()
            optimizer.step()
            if batch_idx %1700 == 0:
                clear_output()
            if batch_idx % 100 == 0:
                  print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      epoch, batch_idx * len(data), len(loader.dataset), 100. * batch_idx / len(loader), output.data.item()))
                      # 100. * batch_idx / len(loader), output.data[0]))
                       
                  print('-----------------------------------')
        
    # torch.save(model.state_dict(), 'char_recognizer_emnist_11_29.pt')
    
    
if __name__ == '__main__':
# dataset()
# traning()
# testing()
    main()