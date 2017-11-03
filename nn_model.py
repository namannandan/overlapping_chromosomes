import torch as t
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torch.optim as optim
import pickle

class segnet_model1(nn.Module):

    def __init__(self):
        super(segnet_model1, self).__init__()
        #Encoder
        #encoder conv layer (EC1) has 1 input channel and generates 5 output channels
        #dimensions of the input image = (96x96x1)
        #padding = 1
        #dimensions of the input image after padding = (98x98x1)
        #kernel size = (3x3)
        #dimensions of the output feature maps after convolution = (96x96x2)
        self.EC1 = nn.Conv2d(1, 5, 3, stride = 1, padding=1)
        #max pool (M1)
        #kernel size = (2x2)
        #stride = (2)
        #output size (48x48x5)
        self.M1 = nn.MaxPool2d(2, stride=None, padding=0, dilation=1, return_indices=True)
        #encoder conv layer (EC2) has 5 input channels and generates 10 output channels
        self.EC2 = nn.Conv2d(5, 10, 3, stride = 1, padding=1)
        #max pool (M2)
        #output size (24x24x10)
        self.M2 = nn.MaxPool2d(2, stride=None, padding=0, dilation=1, return_indices=True)
        #encoder conv layer (EC3) has 10 input channels and generates 20 output channels
        self.EC3 = nn.Conv2d(10, 20, 3, stride = 1, padding=1)
        #max pool (M3)
        #output size (12x12x20)
        self.M3 = nn.MaxPool2d(2, stride=None, padding=0, dilation=1, return_indices=True)
        #encoder conv layer (EC4) has 20 input channels and generates 40 output channels
        self.EC4 = nn.Conv2d(20, 40, 3, stride = 1, padding=1)
        #max pool (M4)
        #output size (6x6x40)
        self.M4 = nn.MaxPool2d(2, stride=None, padding=0, dilation=1, return_indices=True)
        #encoder conv layer (EC5) has 40 input channels and generates 80 output channels
        self.EC5 = nn.Conv2d(40, 80, 3, stride = 1, padding=1)
        #max pool (M5)
        #output size (3x3x80)
        self.M5 = nn.MaxPool2d(2, stride=None, padding=0, dilation=1, return_indices=True)

        #Decoder
        #max unpool layer (MU1)
        #output size (6x6x80)
        self.MU1 = nn.MaxUnpool2d(2, stride=2)
        #decoder conv layer (DC1) has 80 input channels and generates 40 output channels
        self.DC1 = nn.Conv2d(80, 40, 3, stride = 1, padding = 1)
        #max unpool layer (MU2)
        #output size (12x12x40)
        self.MU2 = nn.MaxUnpool2d(2, stride=2)
        #conv layer (DC2) has 40 input channels and generates 20 output channels
        self.DC2 = nn.Conv2d(40, 20, 3, stride = 1, padding = 1)
        #max unpool layer (MU3)
        #output size (24x24x20)
        self.MU3 = nn.MaxUnpool2d(2, stride=2)
        #conv layer (DC3) has 20 input channels and generates 10 output channels
        self.DC3 = nn.Conv2d(20, 10, 3, stride = 1, padding = 1)
        #max unpool layer (MU4)
        #output size (48x48x10)
        self.MU4 = nn.MaxUnpool2d(2, stride=2)
        #conv layer (DC4) has 10 input channels and generates 5 output channels
        self.DC4 = nn.Conv2d(10, 5, 3, stride = 1, padding = 1)
        #max unpool layer (MU5)
        #output size (96x96x5)
        self.MU5 = nn.MaxUnpool2d(2, stride=2)
        #conv layer (DC5) has 5 input channels and generates 5 output channels
        self.DC5 = nn.Conv2d(5, 5, 3, stride = 1, padding = 1)
        #log softmax
        #applies logarithm of the softmax determined at each pixel position in an image
        #(applied across all pixels, across all channels at a given spatial pixel location)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        #Encoder
        x = self.EC1(x)
        x = F.relu(x)
        x, indices1 = self.M1(x)
        x = self.EC2(x)
        x = F.relu(x)
        x, indices2 = self.M2(x)
        x = self.EC3(x)
        x = F.relu(x)
        x, indices3 = self.M3(x)
        x = self.EC4(x)
        x = F.relu(x)
        x, indices4 = self.M4(x)
        x = self.EC5(x)
        x = F.relu(x)
        x, indices5 = self.M5(x)
        #decoder
        x = self.MU1(x, indices5)
        x = self.DC1(x)
        x = self.MU2(x, indices4)
        x = self.DC2(x)
        x = self.MU3(x, indices3)
        x = self.DC3(x)
        x = self.MU4(x, indices2)
        x = self.DC4(x)
        x = self.MU5(x, indices1)
        x = self.DC5(x)
        #log softmax
        x = self.logsoftmax(x)
        return (x)
