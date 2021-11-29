import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
  def __init__(self,input_channels, *args, **kwargs):
    
    super(ResBlock,self).__init__()

    self.dropout_value = 0.1
    self.input_channels = input_channels

    self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_value)
    )
    self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_value) 
    )

  def forward(self,x):
    
    x = self.conv1(x)
    x = self.conv2(x)

    return x

class block(nn.Module):
  def __init__(self,input_channels,output_channels, *args, **kwargs):
    super(block,self).__init__()

    self.input_channels = input_channels
    self.output_channels = output_channels
    self.dropout_value = 0.1

    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels= self.input_channels, out_channels=self.output_channels,
                  kernel_size = (3, 3) ,stride = 1, padding = 1, bias = False),
        nn.MaxPool2d(2,2),
        nn.BatchNorm2d(self.output_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(self.dropout_value)
    )
    self.resblock = ResBlock(self.output_channels)

  def forward(self,x):

    x = self.conv1(x)
    identity = x
    x = self.resblock(x)
    out = x.clone() + identity
    
    return x
class CustomResNet(nn.Module):
  def __init__(self,intitialise_weights = False):
    super(CustomResNet,self).__init__()

    self.dropout_value = 0.1

    self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels = 64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_value)
    )
    
    
    self.layer1 = block(64,128)

    self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels= 128, out_channels=256,
                  kernel_size = (3, 3) ,stride = 1, padding = 1, bias = False),
        nn.MaxPool2d(2,2),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(self.dropout_value)
    )

    self.layer3 = block(256,512)

    self.gmp = nn.MaxPool2d(4,4)

    self.fc1 = nn.Linear(in_features=512,out_features=256)
    self.fc2 = nn.Linear(in_features=256,out_features=64)
    self.fc3 = nn.Linear(in_features=64,out_features=10)

    if intitialise_weights:
        self.intialise_weight()
  
  def forward(self,x):

    x = self.prep_layer(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.gmp(x)
    
    # print(x.shape)

    x = x.view(-1,512)
    
    # print(x.shape)

    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    
    x = x.view(-1,10)

    return F.log_softmax(x)
  
  def intialise_weight(self):
    for m in self.modules():
      if isinstance(m,nn.BatchNorm2d):
        nn.init.constant_(m.weight,0.125)

