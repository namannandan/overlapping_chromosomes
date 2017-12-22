import torch as t
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torch.optim as optim
import pickle

class nn_model(nn.Module):

    def __init__(self):
        super(nn_model, self).__init__()
        #Encoder
        #increse feature maps
        self.EC1_1 = nn.Conv2d(1, 20, 3, stride = 1, padding=1)
        #bottleneck
        self.EC1_2 = nn.Conv2d(20, 5, 1, stride = 1, padding=0)
        self.EC1_3 = nn.Conv2d(5, 5, 3, stride = 1, padding=1)
        self.EC1_4 = nn.Conv2d(5, 20, 1, stride = 1, padding=0)
        #convolution for the residual connection
        self.RES1 = nn.Conv2d(1, 20, 1, stride = 1, padding = 0)
        #batch normalization (BN1)
        #number of feature maps = 5
        self.BN1_1 = nn.BatchNorm2d(20)
        self.BN1_2 = nn.BatchNorm2d(5)
        self.BN1_3 = nn.BatchNorm2d(5)
        self.BN1_4 = nn.BatchNorm2d(20)
        self.BN1_5 = nn.BatchNorm2d(20)
        #max pool
        self.M1 = nn.MaxPool2d(2, stride=None, padding=0, dilation=1, return_indices=True)
        #Encoder
        #increse feature maps
        self.EC2_1 = nn.Conv2d(20, 60, 3, stride = 1, padding=1)
        #bottleneck
        self.EC2_2 = nn.Conv2d(60, 15, 1, stride = 1, padding=0)
        self.EC2_3 = nn.Conv2d(15, 15, 3, stride = 1, padding=1)
        self.EC2_4 = nn.Conv2d(15, 60, 1, stride = 1, padding=0)
        #convolution for the residual connection
        self.RES2 = nn.Conv2d(20, 60, 1, stride = 1, padding = 0)
        #batch normalization (BN1)
        #number of feature maps = 5
        self.BN2_1 = nn.BatchNorm2d(60)
        self.BN2_2 = nn.BatchNorm2d(15)
        self.BN2_3 = nn.BatchNorm2d(15)
        self.BN2_4 = nn.BatchNorm2d(60)
        self.BN2_5 = nn.BatchNorm2d(60)
        #max pool
        self.M2 = nn.MaxPool2d(2, stride=None, padding=0, dilation=1, return_indices=True)
        #Encoder
        #increse feature maps
        self.EC3_1 = nn.Conv2d(60, 180, 3, stride = 1, padding=1)
        #bottleneck
        self.EC3_2 = nn.Conv2d(180, 45, 1, stride = 1, padding=0)
        self.EC3_3 = nn.Conv2d(45, 45, 3, stride = 1, padding=1)
        self.EC3_4 = nn.Conv2d(45, 180, 1, stride = 1, padding=0)
        #convolution for the residual connection
        self.RES3 = nn.Conv2d(60, 180, 1, stride = 1, padding = 0)
        #batch normalization (BN1)
        #number of feature maps = 5
        self.BN3_1 = nn.BatchNorm2d(180)
        self.BN3_2 = nn.BatchNorm2d(45)
        self.BN3_3 = nn.BatchNorm2d(45)
        self.BN3_4 = nn.BatchNorm2d(180)
        self.BN3_5 = nn.BatchNorm2d(180)
        #max pool
        self.M3 = nn.MaxPool2d(2, stride=None, padding=0, dilation=1, return_indices=True)

        #Decoder
        #max unpool layer (MU1)
        #output size (6x6x80)
        # self.MU1 = nn.MaxUnpool2d(2, stride=2)
        # #decoder conv layer (DC1) has 80 input channels and generates 40 output channels
        # self.DC1 = nn.Conv2d(120, 40, 3, stride = 1, padding = 1)
        # #max unpool layer (MU2)
        # #output size (12x12x40)
        # self.MU2 = nn.MaxUnpool2d(2, stride=2)
        # #conv layer (DC2) has 40 input channels and generates 20 output channels
        # self.DC2 = nn.Conv2d(60, 20, 3, stride = 1, padding = 1)
        # #max unpool layer (MU3)
        # #output size (24x24x20)
        self.MU3 = nn.MaxUnpool2d(2, stride=2)
        #conv layer (DC3) has 20 input channels and generates 10 output channels
        self.DC3 = nn.Conv2d(180, 60, 3, stride = 1, padding = 1)
        self.DBN3 = nn.BatchNorm2d(60)
        #max unpool layer (MU4)
        #output size (48x48x10)
        self.MU4 = nn.MaxUnpool2d(2, stride=2)
        #conv layer (DC4) has 10 input channels and generates 5 output channels
        self.DC4 = nn.Conv2d(60, 20, 3, stride = 1, padding = 1)
        self.DBN4 = nn.BatchNorm2d(20)
        #max unpool layer (MU5)
        #output size (96x96x5)
        self.MU5 = nn.MaxUnpool2d(2, stride=2)
        #conv layer (DC5) has 5 input channels and generates 5 output channels
        self.DC5 = nn.Conv2d(20, 5, 3, stride = 1, padding = 1)
        self.DBN5 = nn.BatchNorm2d(5)
        #log softmax
        #applies logarithm of the softmax determined at each pixel position in an image
        #(applied across all pixels, across all channels at a given spatial pixel location)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        #Encoder
        out1 = self.EC1_1(x)
        out1 = self.BN1_1(out1)
        out1 = F.relu(out1)
        out1 = self.EC1_2(out1)
        out1 = self.BN1_2(out1)
        out1 = F.relu(out1)
        out1 = self.EC1_3(out1)
        out1 = self.BN1_3(out1)
        out1 = F.relu(out1)
        out1 = self.EC1_4(out1)
        out1 = self.BN1_4(out1)
        res1 = self.RES1(x)
        res1 = self.BN1_5(res1)
        out1 = out1 + res1
        out1 = F.relu(out1)
        out1, indices1 = self.M1(out1)

        out2 = self.EC2_1(out1)
        out2 = self.BN2_1(out2)
        out2 = F.relu(out2)
        out2 = self.EC2_2(out2)
        out2 = self.BN2_2(out2)
        out2 = F.relu(out2)
        out2 = self.EC2_3(out2)
        out2 = self.BN2_3(out2)
        out2 = F.relu(out2)
        out2 = self.EC2_4(out2)
        out2 = self.BN2_4(out2)
        res2 = self.RES2(out1)
        res2 = self.BN2_5(res2)
        out2 = out2 + res2
        out2 = F.relu(out2)
        out2, indices2 = self.M2(out2)

        out3 = self.EC3_1(out2)
        out3 = self.BN3_1(out3)
        out3 = F.relu(out3)
        out3 = self.EC3_2(out3)
        out3 = self.BN3_2(out3)
        out3 = F.relu(out3)
        out3 = self.EC3_3(out3)
        out3 = self.BN3_3(out3)
        out3 = F.relu(out3)
        out3 = self.EC3_4(out3)
        out3 = self.BN3_4(out3)
        res3 = self.RES3(out2)
        res3 = self.BN3_5(out3)
        out3 = out3+res3
        out3 = F.relu(out3)
        out3, indices3 = self.M3(out3)
        # out4 = self.EC4_1(out3)
        # out4 = self.BN4_1(out4)
        # out4 = F.relu(out4)
        # out4 = self.EC4_2(out4)
        # out4 = self.BN4_2(out4)
        # res4 = self.RES4(out3)
        # out4 = out4+res4
        # out4 = F.relu(out4)
        # out4, indices4 = self.M4(out4)
        # out5 = self.EC5_1(out4)
        # out5 = self.BN5_1(out5)
        # out5 = F.relu(out5)
        # out5 = self.EC5_2(out5)
        # out5 = self.BN5_2(out5)
        # res5 = self.RES5(out4)
        # out5 = out5+res5
        # out5 = F.relu(out5)
        # out5, indices5 = self.M5(out5)
        # #decoder
        # out6 = self.MU1(out5, indices5)
        # out6 = t.cat((out6, out4), 1)
        # out6 = self.DC1(out6)
        # out7 = self.MU2(out6, indices4)
        # out7 = t.cat((out7, out3), 1)
        # out7 = self.DC2(out7)
        out8 = self.MU3(out3, indices3)
        #out8 = t.cat((out8, out2), 1)
        out8 = self.DC3(out8)
        out8 = self.DBN3(out8)
        out8 = F.relu(out8)
        out9 = self.MU4(out8, indices2)
        #out9 = t.cat((out9, out1), 1)
        out9 = self.DC4(out9)
        out9 = self.DBN4(out9)
        out9 = F.relu(out9)
        out10 = self.MU5(out9, indices1)
        #out10 = t.cat((out10, x), 1)
        out10 = self.DC5(out10)
        out10 = self.DBN5(out10)
        out10 = F.relu(out10)
        #log softmax
        out11 = self.logsoftmax(out10)
        return (out11)
