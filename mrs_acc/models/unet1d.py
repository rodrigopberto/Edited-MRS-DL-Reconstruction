import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet1D(nn.Module):

    def __init__(self,steps=4,initial_filters=16,transients=40,input_channels=2,kernel_size=5):

        super(Unet1D,self).__init__()

        self.encoder = _Unet1DEncoder(steps,input_channels*transients,initial_filters,kernel_size)
        self.decoder = _Unet1DDecoder(steps,initial_filters,kernel_size)

        self.mid_block = _Unet1DBlock(initial_filters*(2**(steps-1)),initial_filters*(2**steps),kernel_size=kernel_size)

        self.last_layer = nn.Conv1d(initial_filters,1,kernel_size=kernel_size,padding="same")

    def forward(self,x,ppm=None):
        y = x.permute(0,1,3,2).reshape(x.shape[0],-1,x.shape[2])

        y,output_list = self.encoder(y)
        y = self.mid_block(y)
        y = self.decoder(y,output_list)

        y = self.last_layer(y)

        y = y.squeeze(1)

        return y


class _Unet1DEncoder(nn.Module):

    def __init__(self, steps = 4, input_channels=2,initial_filters=4, kernel_size=3):
        super(_Unet1DEncoder,self).__init__()
        
        input_filters = [input_channels] + [initial_filters*(2**(i)) for i in range(steps-1)]
        out_filters = [initial_filters*(2**(i)) for i in range(steps)]

        self.steps = steps
        self.conv_blocks = nn.ModuleList([_Unet1DBlock(input_filters[i],out_filters[i],kernel_size) for i in range(steps)])
    
    def forward(self,x):
        
        output_list=[]
        y=x
        for i in range(self.steps):
            y = self.conv_blocks[i](y)
            output_list.append(y)
            y = F.max_pool1d(y,kernel_size=(2))
        
        return y,output_list

class _Unet1DDecoder(nn.Module):

    def __init__(self, steps = 4, initial_filters=2, kernel_size=3):
        super(_Unet1DDecoder,self).__init__()

        input_filters = [int(initial_filters*(2**(i)*3)) for i in range(steps)]
        out_filters = [initial_filters*(2**(i)) for i in range(steps)]

        input_filters.reverse()
        out_filters.reverse()

        self.conv_blocks = nn.ModuleList([_Unet1DBlock(input_filters[i],out_filters[i],kernel_size) for i in range(steps)])

        self.steps = steps
    
    def forward(self,x,input_list):

        y = x
        for i in range(self.steps):
            y = F.interpolate(y,scale_factor=2)
            y = torch.cat([y,input_list[(self.steps-1-i)]],axis=1)
            y = self.conv_blocks[i](y)
        
        return y


class _Unet1DBlock(nn.Module):

    def __init__(self, initial_filters=10,end_filters=10,kernel_size=3,in_middle_filters=None):
        super(_Unet1DBlock,self).__init__()

        if in_middle_filters is None:
            mid_filters=end_filters
        else:
            mid_filters=in_middle_filters

        self.conv_1 = nn.Conv1d(initial_filters,mid_filters,kernel_size=kernel_size,padding="same")
        self.conv_2 = nn.Conv1d(mid_filters,end_filters,kernel_size=kernel_size,padding="same")

        self.batch_norm_1 = nn.BatchNorm1d(mid_filters)
        self.batch_norm_2 = nn.BatchNorm1d(end_filters)
    
    def forward(self, x):
        return F.relu(self.batch_norm_2(self.conv_2(F.relu(self.batch_norm_1(self.conv_1(x))))))