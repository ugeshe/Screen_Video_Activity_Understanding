
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


def predict_char(gray, model, idx):
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
    
    plt.imshow(canvas)
    plt.show()

    test_output = model(Variable(torch.FloatTensor(canvas).unsqueeze(0).unsqueeze(0).cuda()))
    pred = test_output.data.max(1, keepdim=True)[1] 
    pred.cpu().numpy()
    
    print('Predicted Label: ', idx[pred])
    
    




def testing():
    
    idx = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
        
    model = Net()
    model.load_state_dict(torch.load("./char_recognizer_emnist_11_29.pt"))
    model.to(torch.device('cuda'))

    pil_im =  Image.open(r"D:\Ugesh\CroppedLetters\0363.jpg")
    new_width  = 40
    new_height = 40
    pil_im = pil_im.resize((new_width, new_height), Image.BICUBIC)
    predict_char(pil_im, model, idx)

if __name__ == '__main__':
    testing()
