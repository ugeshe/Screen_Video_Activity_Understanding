# Import required libraries 
import os
import cv2 as cv 
import copy as cp
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from PIL import ImageOps
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from IPython.display import clear_output

# Define Neural Network
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

# Predict Character 
def classifyChar(gray, model, idx):
    w = gray.size[0]
    h = gray.size[1]
    gray = gray.convert('L')
    gray = gray.point(lambda x: 0 if x<180 else 255, '1')
    x= int(64- (w/2))
    y = int(64- (h/2))
    canvas = Image.new('L', (128, 128), (255))
    canvas.paste(gray, box=(x, y))

    canvas = ImageOps.invert(canvas)
    canvas = np.array(canvas)
    canvas = canvas / 255.0
    
    # plt.imshow(canvas)
    # plt.show()

    test_output = model(Variable(torch.FloatTensor(canvas).unsqueeze(0).unsqueeze(0).cuda()))
    pred = test_output.data.max(1, keepdim=True)[1] 
    pred.cpu().numpy()
    
    print('Predicted Label: ', idx[pred])
    return idx[pred]
    
    
# Hexadecmial Digits
idx = ['0','1','2','3','4','5','6','7','8','9', 'A','B','C','D','E','F']

# Model 
model = Net()
model.load_state_dict(torch.load("path_to_model.pt"))
model.to(torch.device('cuda'))


# cv.namedWindow("Img", cv.WINDOW_AUTOSIZE)
# outVid = cv.VideoWriter('output.avi',cv.VideoWriter_fourcc('M','J','P','G'), 1, (1920,1080))

# Read Image 
img = cv.imread("path_to_image")
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY);
__, imgGray = cv.threshold(imgGray, 150, 255, cv.THRESH_BINARY_INV); 

# Find Rectangular Region in Image
vSE = cv.getStructuringElement(cv.MORPH_RECT, (1, 20)); 
vErodeImg = cv.erode(imgGray, vSE)
vDilateImg = cv.dilate(vErodeImg, vSE)

hSE = cv.getStructuringElement(cv.MORPH_RECT, (20, 1)); 
hErodeImg = cv.erode(imgGray, hSE);
hDilateImg = cv.dilate(hErodeImg, hSE) 

binaryImg = vDilateImg + hDilateImg; 

imgContours = cv.findContours(binaryImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for cnt in reversed(imgContours[0]):
    
    x,y,w,h = cv.boundingRect(cnt)
    
    area = cv.contourArea(cnt)
    imgDisp = cp.deepcopy(img)
    
    if w >= 100 and w <= 500 and area >= 1000 and h >= 10 and h <= 40: 
        imgPatch = imgGray[y:y+h, x:x+w] 
        _, imgPatchBinary = cv.threshold(imgPatch,100,255,cv.THRESH_BINARY_INV)
        
        imgPathContours = cv.findContours(imgPatchBinary,  cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        m = 0

        for cntPatch in reversed(imgPathContours[0]):
            xPatch, yPatch, wPatch, hPatch = cv.boundingRect(cntPatch)
            
            
            if  wPatch >= 8 and wPatch <= 25 and  hPatch >= 8 and hPatch <= 25:
                segChar = imgPatchBinary[yPatch : yPatch + hPatch, xPatch : xPatch + wPatch]
                # Send Image Patch to neural network model 
                segChar = cv.resize(segChar, (40, 40))
                imgPil = Image.fromarray(segChar)
                predictedChar = classifyChar(imgPil, model, idx)
                posIdx = 300 + (m-1)*30
                
                # Display output
                if m != 0:
                    cv.putText(imgDisp, str(predictedChar), (posIdx, 600), 
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
                    imgDisp[500:540, posIdx: posIdx + 40] = cv.merge([segChar, segChar, segChar])
                
                cv.rectangle(imgDisp, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv.rectangle(imgDisp, (x+ xPatch, y + yPatch), (x+ xPatch + wPatch, y + yPatch + hPatch), (0, 0, 255), 1)
                m += 1
        
