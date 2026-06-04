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

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

    def forward(self, x, x1, x2, x3):

        d3 = self.up3(x)
        d3 = torch.cat([d3, x3], dim=1)  # skip connection
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)

        return d1




# Define the FCN model

class FCN(nn.Module):
   def __init__(self,num_classes):
       super().__init__()
       self.encoder = encoder()
       self.decoder = Decoder()
       # classification pixel-wise
       self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

       def forward(self, x):

         x1, x2, x3, bottleneck = self.encoder(x)

         x = self.decoder(bottleneck, x1, x2, x3)

         x = self.classifier(x)

         return x
       



model = FCN(num_classes=4)

x = torch.randn(1, 3, 256, 256)

out = model(x)

print(out.shape)