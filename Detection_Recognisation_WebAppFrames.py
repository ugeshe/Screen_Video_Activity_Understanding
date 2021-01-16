# Import required Libraries
import os
import cv2 as cv 
import copy as cp
import numpy as np
import math

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
    
    # print('Predicted Label: ', idx[pred])
    return idx[pred]

    
def matchedChars(a,b):
    if len(a) > len(b):
        return sum ( a[i] == b[i] for i in range(len(b)) )
    else: 
        return sum ( a[i] == b[i] for i in range(len(a)) )
    
# Hexadecmial Digits
idx = ['0','1','2','3','4','5','6','7','8','9', 'A','B','C','D','E','F']

# Model
model = Net()
model.load_state_dict(torch.load("D:/UNM/Thesis/ivPCL/Datsets/Project_Submission/char_recognizer_emnist_11_29.pt"))
model.to(torch.device('cuda'))


# cv.namedWindow("Img", cv.WINDOW_AUTOSIZE)
# cv.namedWindow("ImgPrev", cv.WINDOW_AUTOSIZE)

# Output Video Writer
# outVid = cv.VideoWriter('webAppFrames_reconginstion_12_07_Detection_Recognistion_on_entrieVideo.avi',cv.VideoWriter_fourcc('M','J','P','G'), 1, (1920,1080))


# Current Frame differencing method uses two consecutive frames but as project progress, 
# we would use more temporal frames

for fileName in range(5, 144):
    # Read Image
    img = cv.imread("D:/UNM/Thesis/ivPCL/RecordedVidoes/WebApp_GrayScale/Image_" + "{0:04}".format(fileName) + ".jpg")
    imgPrev = cv.imread("D:/UNM/Thesis/ivPCL/RecordedVidoes/WebApp_GrayScale/Image_" + '{0:04}'.format(fileName - 1) + ".jpg")
    
    img_ = cp.deepcopy(img)
    imgPrev_ = cp.deepcopy(imgPrev)
    
    # Binaryize Image 
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY);
    __, imgGray = cv.threshold(imgGray, 150, 255, cv.THRESH_BINARY_INV); 
    
    imgGrayPrev = cv.cvtColor(imgPrev, cv.COLOR_BGR2GRAY);
    __, imgGrayPrev = cv.threshold(imgGrayPrev, 150, 255, cv.THRESH_BINARY_INV); 
    
    # Find rectangular regions
    vSE = cv.getStructuringElement(cv.MORPH_RECT, (1, 20)); 
    vErodeImg = cv.erode(imgGray, vSE)
    vDilateImg = cv.dilate(vErodeImg, vSE)
    
    hSE = cv.getStructuringElement(cv.MORPH_RECT, (20, 1)); 
    hErodeImg = cv.erode(imgGray, hSE);
    hDilateImg = cv.dilate(hErodeImg, hSE) 
    
    vErodeImgPrev = cv.erode(imgGrayPrev, vSE)
    vDilateImgPrev = cv.dilate(vErodeImgPrev, vSE)
    
    hErodeImgPrev = cv.erode(imgGrayPrev, hSE);
    hDilateImgPrev = cv.dilate(hErodeImgPrev, hSE) 
    
    binaryImg = vDilateImg + hDilateImg; 
    binaryImgPrev = vDilateImgPrev + hDilateImgPrev; 
    
    imgContours = cv.findContours(binaryImg,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    imgContoursPrev = cv.findContours(binaryImgPrev,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    imgBBox = []
    imgBBoxPrev = []
    minBBox = []
    finalBBox = []
    finalBBoxPrev = []
    
    for cnt in reversed(imgContours[0]):
    
        x,y,w,h = cv.boundingRect(cnt)
        area = cv.contourArea(cnt)
        
        if w >= 100 and w <= 500 and area >= 1000 and h >= 10 and h <= 40: 
            imgBBox.append( [x, y, x+w, y + h])
            # cv.rectangle(img, (x, y), (x+ w, y + h), (0, 0, 255), 1)
            
    
    
    for cnt in reversed(imgContoursPrev[0]):
    
        x,y,w,h = cv.boundingRect(cnt)
        area = cv.contourArea(cnt)
        
        if w >= 100 and w <= 500 and area >= 1000 and h >= 10 and h <= 40: 
            imgBBoxPrev.append( [x, y, x+w, y + h])
            # cv.rectangle(imgPrev, (x, y), (x+ w, y + h), (0, 255, 255), 1)
            
    distRect = np.zeros([len(imgBBox), len(imgBBoxPrev)])
    
    for m in range(len(imgBBox)):
        for n in range(len(imgBBoxPrev)):
            centrod1 = [(imgBBox[m][0] + imgBBox[m][2])/2, (imgBBox[m][1] + imgBBox[m][3])/2]
            centrod2 = [(imgBBoxPrev[n][0] + imgBBoxPrev[n][2])/2, (imgBBoxPrev[n][1] + imgBBoxPrev[n][3])/2]
            distRect[m,  n] = math.dist(centrod1, centrod2)
    
    # Locate region with changes with in two frames
    for m in range(len(imgBBox)):
        if distRect.size > 0: 
            minBBox.append([m, np.argmin(distRect[m, :])])
          
    multiPatch = 0
    for m in range(len(minBBox)):
       
        imgPatch = imgGray[imgBBox[minBBox[m][0]][1] : imgBBox[minBBox[m][0]][3], imgBBox[minBBox[m][0]][0] : imgBBox[minBBox[m][0]][2]]
        imgPatchPrev = imgGrayPrev[imgBBoxPrev[minBBox[m][1]][1] : imgBBoxPrev[minBBox[m][1]][3], imgBBoxPrev[minBBox[m][1]][0] : imgBBoxPrev[minBBox[m][1]][2]]
        
        imgPatchPrev_ = cp.deepcopy(imgPatchPrev)
        imgPatchPrev_ = cv.resize(imgPatchPrev_, (imgPatch.shape[1], imgPatch.shape[0]))
   

        nnzPixels = np.count_nonzero(imgPatch - imgPatchPrev_)
        
        if nnzPixels > 10:
            _, imgPatchBinary = cv.threshold(imgPatch,100,255,cv.THRESH_BINARY_INV)
            imgPathContours = cv.findContours(imgPatchBinary,  cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            
            _, imgPatchBinaryPrev = cv.threshold(imgPatchPrev,100,255,cv.THRESH_BINARY_INV)
            imgPathContoursPrev = cv.findContours(imgPatchBinaryPrev,  cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        
            charInPatch = []
            charInPatchPrev = []
            finalChar = []
            finalCharBBox = []
            eliminateHash = 0
            
            for cntPatch in reversed(imgPathContours[0]):
                xPatch, yPatch, wPatch, hPatch = cv.boundingRect(cntPatch)
                
                if  wPatch >= 8 and wPatch <= 25 and  hPatch >= 8 and hPatch <= 25:
                    if eliminateHash > 0:
                        finalCharBBox.append([xPatch, yPatch, wPatch, hPatch])
                        # Recognise characters in patchs
                        segChar = imgPatchBinary[yPatch : yPatch + hPatch, xPatch : xPatch + wPatch]
                        if segChar.size > 0: 
                            segChar = cv.resize(segChar, (40, 40))
                            imgPil = Image.fromarray(segChar)
                            predictedChar = classifyChar(imgPil, model, idx)
                            charInPatch.append(predictedChar)
                        
                        segCharPrev = imgPatchBinaryPrev[yPatch : yPatch + hPatch, xPatch : xPatch + wPatch]
                        if segCharPrev.size > 0: 
                            segCharPrev = cv.resize(segCharPrev, (40, 40))
                            imgPilPrev = Image.fromarray(segCharPrev)
                            predictedCharPrev = classifyChar(imgPilPrev, model, idx)
                            charInPatchPrev.append(predictedCharPrev)
                    
                    eliminateHash += 1
                    
                     
                    # cv.rectangle(imgDisp, (x+ xPatch, y + yPatch), (x+ xPatch + wPatch, y + yPatch + hPatch), (0, 0, 255), 1)
                    
            # Find number of chacters matched between frames
            if matchedChars(charInPatchPrev, charInPatch) > 0 and matchedChars(charInPatchPrev, charInPatch) < 7: 
                for i in range(len(finalCharBBox)):
                    # img_ = cv.rectangle(img_,(imgBBox[minBBox[m][0]][0]+ xPatch, imgBBox[minBBox[m][0]][1] + yPatch), (imgBBox[minBBox[m][0]][0]+ xPatch + wPatch, imgBBox[minBBox[m][0]][1] + yPatch + hPatch), (0, 255, 0), 1)
                    xPatch = finalCharBBox[i][0]
                    yPatch = finalCharBBox[i][1]
                    wPatch = finalCharBBox[i][2]
                    hPatch = finalCharBBox[i][3]
                    img_ = cv.rectangle(img_,(imgBBox[minBBox[m][0]][0]+ xPatch, imgBBox[minBBox[m][0]][1] + yPatch), (imgBBox[minBBox[m][0]][0]+ xPatch + wPatch, imgBBox[minBBox[m][0]][1] + yPatch + hPatch), (0, 255, 0), 1)
               
                finalChar.append(charInPatch)
                # img_ = cv.putText(img_, str(charInPatch), (300, 600), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
            
            # Display Output 
            for chars in range(len(finalChar)):
                yPosIdx = 500 + multiPatch*50
                multiPatch += 1
                img_ = cv.putText(img_, str(finalChar[chars]), (100, yPosIdx), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
            charInPatch = []
            charInPatchPrev = []
            finalChar = []
    
            # cv.imshow("Img", img_)
            # cv.waitKey(1)
            
    # outVid.write(img_)
    print('FrameNumber: ' + str(fileName))
    # cv.destroyAllWindows()
    
    
# outVid.release()
