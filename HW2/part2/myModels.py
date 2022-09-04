
# Modelzoo for usage 
# Feel free to add any model you like for your final result
# Note : Pretrained model is allowed iff it pretrained on ImageNet

from audioop import bias
from pickle import TRUE
from pydoc import classname
from turtle import forward
import torch
import torch.nn as nn
from torchvision.models import resnet50

class mypretrain(nn.Module):
    def __init__(self, num_out = 10):
        super(mypretrain, self).__init__()
        self.resnet = resnet50(pretrained= TRUE)
        self.fc1 = nn.Linear(in_features= self.resnet.fc.out_features, out_features= 512)
        self.fc2 = nn.Linear(in_features= 512, out_features= num_out)
    
    def forward(self, x):
        out = self.resnet(x)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class myLeNet(nn.Module):
    def __init__(self, num_out):
        super(myLeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120,84), nn.ReLU())
        self.fc3 = nn.Linear(84,num_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        # It is important to check your shape here so that you know how manys nodes are there in first FC in_features
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        out = x
        return out

    
class residual_block(nn.Module):
    def __init__(self, in_channels):
        super(residual_block, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(in_channels))

        self.relu = nn.ReLU()
        
    def forward(self,x):
        ## TO DO ## 
        # Perform residaul network. 
        # You can refer to our ppt to build the block. It's ok if you want to do much more complicated one. 
        # i.e. pass identity to final result before activation function 
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = out + x
        out = self.relu(out)
        return out
        # pass

        
class myResnet(nn.Module):
    def __init__(self, in_channels=3, num_out=10):
        super(myResnet, self).__init__()
        
        self.stem_conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        # self.block = residual_block(in_channels= 64)
        self.layer = residual_block(in_channels= 64)
        self.cnn_layer1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
                                nn.BatchNorm2d(num_features=64),
                                nn.ReLU())
        self.cnn_layer2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, stride=1, kernel_size=3,padding=1),
                                        nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
                                        nn.BatchNorm2d(num_features=128),
                                        nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(in_features=8*8*128, out_features=512, bias=True),
                                nn.ReLU())
        self.fc2 = nn.Linear(in_features=512, out_features=num_out, bias=True)
        ## TO DO ##
        # Define your own residual network here. 
        # Note: You need to use the residual block you design. It can help you a lot in training.
        # If you have no idea how to design a model, check myLeNet provided by TA above.
        
        pass
        
    def forward(self,x):
        ## TO DO ## 
        # Define the data path yourself by using the network member you define.
        # Note : It's important to print the shape before you flatten all of your nodes into fc layers.
        # It help you to design your model a lot. 
        # x = x.flatten(x)
        # print(x.shape)
        out = self.stem_conv(x)
        out = self.layer(out)
        out = self.cnn_layer1(out)
        out = self.layer(out)
        out = self.cnn_layer2(out)
        flat = torch.flatten(out, start_dim=1, end_dim=-1)
        flat = self.fc1(flat)
        ans = self.fc2(flat)
        return ans
        pass