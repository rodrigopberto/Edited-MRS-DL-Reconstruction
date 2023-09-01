import torch
import torch.nn as nn
import torch.nn.functional as F




class PreDownUnet2DTanh(nn.Module):

    def __init__(self,initial_filters=16,transients=40,input_channels=2,kernel_size=3,
                 depth_1d=3,kernel_sizes_1d=[3,5,7,9],):

        super(PreDownUnet2DTanh,self).__init__()

        self.pre_cnn = CNN1DMultiChain(depth=depth_1d,input_filters=input_channels,output_filters=initial_filters,kernel_sizes=kernel_sizes_1d)

        self.encoder = _DownUnet2DEncoder(4,initial_filters*len(kernel_sizes_1d),initial_filters,kernel_size)
        self.decoder = _DownUnet2DDecoder(4,initial_filters,kernel_size)

        self.mid_block = _DownUnet2DBlock(initial_filters*(2**(4-1)),initial_filters*(2**4),kernel_size=kernel_size)

        self.last_layer = nn.Conv2d(initial_filters,initial_filters,kernel_size=(kernel_size,4),padding=(kernel_size//2,0))
        self.last_last_layer = nn.Conv2d(initial_filters,1,kernel_size=kernel_size,padding="same")

    def forward(self,x,ppm=None):
        y = self.pre_cnn(x)
        #print(y.shape)
        y,output_list = self.encoder(y)
        y = self.mid_block(y)
        y = self.decoder(y,output_list)

        y = self.last_last_layer(F.tanh(self.last_layer(y)))

        y = y.squeeze(-1).squeeze(1)

        return y

class CNN1DMultiChain(nn.Module):
    def __init__(self,depth=3,input_filters=2,output_filters=16,kernel_sizes=[3,5,7,9]):
        super(CNN1DMultiChain,self).__init__()

        self.chains = nn.ModuleList([CNN1DSingleChain(depth,input_filters,output_filters,kernel_size=kernel_sizes[i]) for i in range(len(kernel_sizes))])
        self.kernel_sizes_len = len(kernel_sizes)
    
    def forward(self,x):
        y_list=[]
        for i in range(self.kernel_sizes_len):
            y_list.append(self.chains[i](x))
        return torch.cat(y_list,axis=1)

        
class CNN1DSingleChain(nn.Module):
    def __init__(self,depth=3,input_filters=2,output_filters=16,kernel_size=3):
        super(CNN1DSingleChain,self).__init__()
        self.depth=depth
        input_filters = [input_filters]+[output_filters]*(depth-1)

        self.conv_blocks = nn.ModuleList([nn.Conv2d(input_filters[i],output_filters,kernel_size=(kernel_size,1),padding="same") for i in range(depth)])

    def forward(self,x):
        y=x
        for i in range(self.depth):
            y = F.tanh(self.conv_blocks[i](y))
        return y

class _DownUnet2DEncoder(nn.Module):

    def __init__(self, steps = 4, input_channels=2,initial_filters=4, kernel_size=3):
        super(_DownUnet2DEncoder,self).__init__()
        
        input_filters = [input_channels] + [initial_filters*(2**(i)) for i in range(steps-1)]
        out_filters = [initial_filters*(2**(i)) for i in range(steps)]

        self.steps = steps
        self.conv_blocks = nn.ModuleList([_DownUnet2DBlock
                                          (input_filters[i],out_filters[i],kernel_size) for i in range(steps)])
    
    def forward(self,x):
        
        output_list=[]
        y=x
        for i in range(self.steps):
            y = self.conv_blocks[i](y)
            output_list.append(y)
            y = F.max_pool2d(y,kernel_size=(2,1))
        
        return y,output_list

class _DownUnet2DDecoder(nn.Module):

    def __init__(self, steps = 4, initial_filters=2, kernel_size=3):
        super(_DownUnet2DDecoder,self).__init__()

        input_filters = [int(initial_filters*(2**(i)*3)) for i in range(steps)]
        out_filters = [initial_filters*(2**(i)) for i in range(steps)]

        input_filters.reverse()
        out_filters.reverse()

        
        self.conv_blocks = nn.ModuleList([_DownUnet2DBlock(input_filters[i],out_filters[i],kernel_size) for i in range(steps)])

        bridge_filters = [initial_filters*(2**(i)) for i in range(steps)]
        bridge_filters.reverse()
        bridge_kernel_size_y = [5,13,21,29]
        self.bridge_blocks = nn.ModuleList([_DownUnet2DBridgeBlock(bridge_filters[i],kernel_size,bridge_kernel_size_y[i]) for i in range(len(bridge_filters))])
        self.steps = steps
    
    def forward(self,x,input_list):

        y = x
        for i in range(self.steps):
            y = F.interpolate(y,scale_factor=(2,1))
            y = torch.cat([y,self.bridge_blocks[i](input_list[(self.steps-1-i)])],axis=1)
            y = self.conv_blocks[i](y)
        
        return y


class _DownUnet2DBlock(nn.Module):

    def __init__(self, initial_filters=10,end_filters=10,kernel_size=3,in_middle_filters=None):
        super(_DownUnet2DBlock,self).__init__()

        if in_middle_filters is None:
            mid_filters=end_filters
        else:
            mid_filters=in_middle_filters

        self.conv_1 = nn.Conv2d(initial_filters,mid_filters,kernel_size=(kernel_size,3),padding=(kernel_size//2,0))
        self.conv_2 = nn.Conv2d(mid_filters,end_filters,kernel_size=(kernel_size,3),padding=(kernel_size//2,0))

        self.batch_norm_1 = nn.BatchNorm2d(mid_filters)
        self.batch_norm_2 = nn.BatchNorm2d(end_filters)
    
    def forward(self, x):
        return F.tanh(self.batch_norm_2(self.conv_2(F.tanh(self.batch_norm_1(self.conv_1(x))))))
    
class _DownUnet2DBridgeBlock(nn.Module):

    def __init__(self, filters=10,kernel_size_x=3,kernel_size_y=3):
        super(_DownUnet2DBridgeBlock,self).__init__()

        self.conv = nn.Conv2d(filters,filters,kernel_size=(kernel_size_x,kernel_size_y),padding=(kernel_size_x//2,0))
        self.batch_norm = nn.BatchNorm2d(filters)
    
    def forward(self,x):
        y= F.tanh(self.batch_norm(self.conv(x)))
        return y

