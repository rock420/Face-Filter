## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        
        self.conv1=nn.Conv2d(1,32,3)
        self.conv2=nn.Conv2d(32,64,3,padding=(1,1))
        self.conv3=nn.Conv2d(64,128,3,padding=(1,1))
        self.short= nn.Conv2d(32,128,1)   ## for resnet connection
        self.conv4=nn.Conv2d(128,256,5)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1= nn.Linear(25*25*256,1000)
        self.fc2 = nn.Linear(1000,500)
        self.output=nn.Linear(500,136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv1(x)))  ## layer 1 (conv+relu+pool)
        residual=self.short(x)  ## short connection for resnet (conv to match the size of 3rd layer output)
        
        #print(x.shape)
        x = F.relu(self.conv2(x))  ## layer 2 (conv+relu)

        #print(x.shape)         
        x = self.conv3(x)          ## layer 3(conv)
        x = self.pool(F.relu(x+residual))   ## add the residual got from fist layer to the output of 3rd layer (add+relu+pool)
        x = F.dropout(x,p=0.2)
        
        #print(x.shape)
        x = self.pool(F.relu(self.conv4(x)))   ## layer 4(conv+relu+pool)
        x = F.dropout(x,p=0.4)
        
        #print(x.shape)
        x = x.view(x.size(0),-1)  ## Flatten the output for fc layers
        
        #print(x.shape)
        x = F.relu(self.fc1(x))  ## first fullly connected layer of 1000 node + dropout
        x = F.dropout(x,p=0.4)
        
        #print(x.shape)
        x = F.relu(self.fc2(x))  ## second fully connected layer of 500 node + dropout
        x = F.dropout(x,p=0.2)
        
        #print(x.shape)
        x = self.output(x)   ## final output layer of 136 node
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x