import torch
import torch.nn as nn
import torch.nn.functional as F




class ResUnet2D(nn.Module):

    def __init__(self,steps=4,initial_filters=4,input_channels=2,kernel_size=3):

        super(ResUnet2D,self).__init__()

        self.encoder = _ResUnet2D()
        self.decoder = _ResUnet2D

    def forward(self,x,ppm):

        x_avg = torch.mean(x,axis=3,keepdim=True)[:,:1,:,:]

    
class _ResUnet2DEncoder(nn.Module):

    def __init__(self,steps=4,initial_filters=4,input_channels=2,kernel_size=3):
        super(_ResUnet2DEncoder,self).__init__()

        output_filters = [initial_filters*(2)**(i) for i in range(steps)]
        input_filters = [input_channels] + output_filters[:steps-1]

        self.blocks = nn.ModuleList([_ResUnet2DBlock(input_filters=input_filters[i],output_filters=output_filters[i],kernel_size=kernel_size) for i in range(steps)])
        self.steps=steps

    def forward(self,x,x_avg):
        skip_outputs = []
        for i in self.steps:
            x,x_avg = self.blocks[i](x,x_avg)
            skip_outputs.append(x)
            




class _ResUnet2DBlock(nn.Module):

    def __init__(self,input_filters=16,output_filters=16,middle_filters=None,kernel_size=3):
        super(_ResUnet2DBlock,self).__init__()

        if middle_filters==None:
            mid_filters=output_filters
        else:
            mid_filters=middle_filters

        self.batch_norm_1 = nn.BatchNorm2d(input_filters)
        self.conv_1 = nn.Conv2d(in_channels=input_filters,out_channels=mid_filters,kernel_size=kernel_size,padding="same")
        
        self.batch_norm_2 = nn.BatchNorm2d(mid_filters)
        self.conv_2 = nn.Conv2d(in_channels=input_filters,out_channels=mid_filters,kernel_size=kernel_size,padding="same")

        self.conv_3 = nn.Conv3d(output_filters,1,kernel_size=kernel_size,padding="same")

    def forward(self,x,x_avg):
        x  = x-x_avg
        y = self.conv_2(F.tanh(self.batch_norm_2(self.conv_1(F.tanh,self.batch_norm_2))))
        y_avg = x_avg+self.conv_3(y).mean(axis=3,keepdim=True)

        return y,y_avg


class Unet2D(nn.Module):

    def __init__(self,steps=4,initial_filters=4,input_channels=2,kernel_size=(3,3)):

        super(Unet2D,self).__init__()

        self.encoder = _Unet2DEncoder(steps,input_channels,initial_filters,kernel_size)
        self.decoder = _Unet2DDecoder(steps,initial_filters,kernel_size)

        self.mid_block = _Unet2DBlock(initial_filters*(2**(steps-1)),initial_filters*(2**steps),kernel_size=kernel_size)

        self.last_layer = nn.Conv2d(initial_filters,1,kernel_size=kernel_size,padding="same")

    def forward(self,x,ppm=None):

        y,output_list = self.encoder(x)
        y = self.mid_block(y)
        y = self.decoder(y,output_list)

        y = self.last_layer(y)

        y = y.mean(axis=3).squeeze(1)

        return y


class _Unet2DEncoder(nn.Module):

    def __init__(self, steps = 4, input_channels=2,initial_filters=4, kernel_size=(3,3)):
        super(_Unet2DEncoder,self).__init__()
        
        input_filters = [input_channels] + [initial_filters*(2**(i)) for i in range(steps-1)]
        out_filters = [initial_filters*(2**(i)) for i in range(steps)]

        self.steps = steps
        self.conv_blocks = nn.ModuleList([_Unet2DBlock(input_filters[i],out_filters[i],kernel_size) for i in range(steps)])
    
    def forward(self,x):
        
        output_list=[]
        y=x
        for i in range(self.steps):
            y = self.conv_blocks[i](y)
            output_list.append(y)
            y = F.max_pool2d(y,kernel_size=(2,1))
        
        return y,output_list

class _Unet2DDecoder(nn.Module):

    def __init__(self, steps = 4, initial_filters=2, kernel_size=(3,3)):
        super(_Unet2DDecoder,self).__init__()

        input_filters = [int(initial_filters*(2**(i)*3)) for i in range(steps)]
        out_filters = [initial_filters*(2**(i)) for i in range(steps)]

        input_filters.reverse()
        out_filters.reverse()

        self.conv_blocks = nn.ModuleList([_Unet2DBlock(input_filters[i],out_filters[i],kernel_size) for i in range(steps)])

        self.steps = steps
    
    def forward(self,x,input_list):

        y = x
        for i in range(self.steps):
            y = F.interpolate(y,scale_factor=(2,1))
            y = torch.cat([y,input_list[(self.steps-1-i)]],axis=1)
            y = self.conv_blocks[i](y)
        
        return y


class _Unet2DBlock(nn.Module):

    def __init__(self, initial_filters=10,end_filters=10,kernel_size=(3,3),in_middle_filters=None):
        super(_Unet2DBlock,self).__init__()

        if in_middle_filters is None:
            mid_filters=end_filters
        else:
            mid_filters=in_middle_filters

        self.conv_1 = nn.Conv2d(initial_filters,mid_filters,kernel_size=kernel_size,padding="same")
        self.conv_2 = nn.Conv2d(mid_filters,end_filters,kernel_size=kernel_size,padding="same")

        self.batch_norm_1 = nn.BatchNorm2d(mid_filters)
        self.batch_norm_2 = nn.BatchNorm2d(end_filters)
    
    def forward(self, x):
        return F.relu(self.batch_norm_2(self.conv_2(F.relu(self.batch_norm_1(self.conv_1(x))))))