import torch
import torch.nn as nn
import torch.nn.functional as F




class ResCNN2D(nn.Module):

    def __init__(self,steps=4,filters=4,input_channels=2,kernel_size=3,
                 depth_1d=3,kernel_sizes_1d=[3,5,7,9]):

        super(ResCNN2D,self).__init__()

        self.pre_cnn = CNN1DMultiChain(depth=depth_1d,input_filters=input_channels,output_filters=filters,kernel_sizes=kernel_sizes_1d)

        self.cnn_blocks = nn.ModuleList([_ResCNN2DBlock(filters*len(kernel_sizes_1d),filters*len(kernel_sizes_1d),kernel_size=kernel_size) for i in range(steps)])

        self.final_cnn = nn.Conv2d(filters*len(kernel_sizes_1d),1,kernel_size=1)

        self.steps = steps

    def forward(self,x,ppm):

        x_avg = torch.mean(x,axis=3,keepdim=True)

        x = self.pre_cnn(x)
        x_avg = self.pre_cnn(x_avg)

        for i in range(self.steps):
            x,x_avg = self.cnn_blocks[i](x,x_avg)
        
        y = self.final_cnn(x)

        return y.squeeze(1).squeeze(-1)
        
    

    

class _ResCNN2DBlock(nn.Module):

    def __init__(self,input_filters=16,output_filters=16,middle_filters=None,kernel_size=3):
        super(_ResCNN2DBlock,self).__init__()

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
            y = F.relu(self.conv_blocks[i](y))
        return y



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