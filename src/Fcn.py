import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the convolutional block
class conv_block(nn.Module):
   def __init__(self, in_ch,out_ch):
      super().__init__()
      self.block= nn.Sequential(
         nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1),
         nn.ReLU(inplace=True),
         nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1),
         nn.ReLU(inplace=True),
      )


      def forward(self,x):
         return self.block(x)
      

#Encoder block
class encoder(nn.Module):
   def __init__(self,in_ch,out_ch):
      super().__init__()
      self.conv1=conv_block(3,64)
      self.pool1=nn.MaxPool2d(2)

      self.conv2=conv_block(64,128)
      self.pool2=nn.MaxPool2d(2)

      self.conv3=conv_block(128,256)
      self.pool3=nn.MaxPool2d(2)

      

   def forward(self,x):
      x1=self.conv1(x)
      p1=self.pool1(x1)
      x2=self.conv2(p1)
      p2=self.pool2(x2)
      x3=self.conv3(p2)
      p3=self.pool3(x3)

      return p3,x1,x2,x3

#decoder block






# Define the FCN model

class FCN