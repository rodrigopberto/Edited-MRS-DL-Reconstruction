import torch.nn as nn
import torch
import torch.nn.functional as F


class OldUNET2D(nn.Module):

    def __init__(self,steps=4,initial_filters=16,input_channels=1):
        super(OldUNET2D,self).__init__()

        self.conv1d_1 = nn.Conv2d(input_channels,initial_filters,kernel_size=(5,1),padding=(2,0))
        self.conv1d_2 = nn.Conv2d(initial_filters,initial_filters,kernel_size=(5,1),padding=(2,0))

        self.enc = _UNET2DEncoder(steps,initial_filters,input_channels=initial_filters)
        self.dec = _UNET2DDecoder(steps,initial_filters)
        self.mid_block = UnetBlock(initial_filters*(2**(steps-1)),initial_filters*(2**(steps-1)),initial_filters*(2**(steps)))
        self.last_layer = nn.Conv2d(initial_filters,1,kernel_size=3,padding="same")

    def forward(self,x,ppm):
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)

        x,skips = self.enc(x)
        x = self.mid_block(x)
        x = self.dec(x,skips)
        #y = x.permute(0,2,3,1)
        y = self.last_layer(x)
        y = y.mean(axis=3).squeeze(1)

        #y = y.squeeze(1)#.squeeze(-1)

        return y

class _UNET2DEncoder(nn.Module):
    def __init__(self,steps=4,initial_filters=16,kernel_size=3,input_channels=1):
        super(_UNET2DEncoder,self).__init__()


        in_channels = [input_channels]+[initial_filters*(2**i) for i in range(steps-1)]
        out_channels = [initial_filters*(2**i) for i in range(steps)]
        
        self.enc_blocks = nn.ModuleList([UnetBlock(in_channels[i],out_channels[i],kernel_size=kernel_size) for i in range(steps)])
    
    def forward(self, x):
        outputs=[]
        for block in self.enc_blocks:
            x = block(x)
            outputs.append(x)
            x = F.max_pool2d(x,(2,1))
        
        
        return x,outputs

class _UNET2DDecoder(nn.Module):
    def __init__(self,steps=4,initial_filters=16,kernel_size=3):
        super(_UNET2DDecoder,self).__init__()
        in_channels = [initial_filters*(2**(i+1)) for i in range(steps)]
        out_channels = [initial_filters*(2**(i-1)) for i in range(steps)]
        in_channels.reverse()
        out_channels.reverse()


        out_channels[-1]=int(out_channels[-1]*2)

        
        self.dec_blocks = nn.ModuleList([UnetBlock(in_channels[i],out_channels[i],kernel_size=kernel_size) for i in range(steps)])
    
    def forward(self, x, skips):
        
        for i in range(len(skips)):
            x = F.interpolate(x,scale_factor=(2,1))
            x = torch.cat([x,skips[-(1+i)]],axis=1)
            x = self.dec_blocks[i](x)
            

        return x

class UnetBlock(nn.Module):

    def __init__(self,in_channels,out_channels,mid_channels=None,kernel_size=3):
        super(UnetBlock,self).__init__()

        if mid_channels==None:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels,mid_channels,kernel_size=kernel_size,stride=1,padding="same").float()
        self.conv2 = nn.Conv2d(mid_channels,out_channels,kernel_size=kernel_size,stride=1,padding="same").float()
        self.bacth_norm_1 = nn.BatchNorm2d(mid_channels)
        self.bacth_norm_2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        return F.relu(self.bacth_norm_2(self.conv2(F.relu(self.bacth_norm_1(self.conv1(x))))))
        #return F.relu(self.conv2(F.relu(self.conv1(x))))