import torch.nn as nn
import torch
import torch.nn.functional as F


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

class DownUnet2DMLPV3(nn.Module):

    def __init__(self,depth_1d=3,kernel_sizes_1d = [3,5,7,9],initial_filters=4,transients=40,input_channels=2,kernel_size=3):

        super(DownUnet2DMLPV3,self).__init__()

        self.steps=4
        self.zoom_size=256
        self.zoom_min_ind=1084

        self.regular_unet = _RegularDownUnet2D(initial_filters=initial_filters,transients=transients,input_channels=(initial_filters*len(kernel_sizes_1d)),kernel_size=kernel_size)
        self.zoom_mlp_unet = _ZoomMLPDownUnet2D(initial_filters=initial_filters,transients=transients,input_channels=(initial_filters*len(kernel_sizes_1d)),kernel_size=kernel_size,zoom_size=self.zoom_size)


        self.pre_cnn = CNN1DMultiChain(depth=depth_1d,input_filters=input_channels,output_filters=initial_filters,kernel_sizes=kernel_sizes_1d)

        self.last_layer = nn.Conv2d(initial_filters*2,initial_filters*4,kernel_size=(kernel_size,4),padding=(kernel_size//2,0))
        self.last_last_conv_layer = nn.Conv2d(initial_filters*4,1,kernel_size=kernel_size,padding="same")

        self.mlp_layer_1 = nn.Linear(self.zoom_size*initial_filters*4,self.zoom_size)
        self.mlp_layer_2 = nn.Linear(self.zoom_size,self.zoom_size)
        self.mlp_combine_layer = nn.Linear(self.zoom_size*2,self.zoom_size)
        self.rescaling_layer = nn.Linear(self.zoom_size,self.zoom_size)

        #self.last_layer = nn.Conv2d(initial_filters,initial_filters,kernel_size=(kernel_size,4),padding=(kernel_size//2,0))
        

    def forward(self,x,ppm=None):

        zoom_min_ind=1112
        zoom_max_ind=zoom_min_ind+self.zoom_size

        y = self.pre_cnn(x)

        y_regular = self.regular_unet(y)

        #print(f"y regular shape:{y_regular.shape}")


        y_zoom_input = y[:,:,self.zoom_min_ind:self.zoom_min_ind+self.zoom_size]

        y_zoom = self.zoom_mlp_unet(y_zoom_input)

        #print(f"y_zoom output:{y_regular.shape}")

        device = torch.get_device(y_zoom)

        y_zoom_larger = torch.cat([torch.zeros(y_zoom.shape[0],y_zoom.shape[1],self.zoom_min_ind,y_zoom.shape[3]).to(device),
                                   y_zoom,
                                   torch.zeros(y_zoom.shape[0],y_zoom.shape[1],y_regular.shape[2]-self.zoom_min_ind-self.zoom_size,y_zoom.shape[3]).to(device)],axis=2)

        y = torch.cat([y_regular,y_zoom_larger],axis=1)


        y = F.relu(self.last_layer(y))

        y_out_conv = self.last_last_conv_layer(y)

        y_out_conv_zoom = y_out_conv[:,:,zoom_min_ind:zoom_max_ind,:].reshape(y.shape[0],-1)

        y_zoom = y[:,:,zoom_min_ind:zoom_max_ind,:].reshape(y.shape[0],-1)
        y_out_mlp = F.relu(self.mlp_layer_2(F.relu(self.mlp_layer_1(y_zoom))))
        y_combine = self.rescaling_layer(F.relu(self.mlp_combine_layer(torch.cat([y_out_conv_zoom,y_out_mlp],axis=-1))))
        y_zoom_out = self.rescaling_layer(y_combine)#.reshape(y.shape[0],1,self.zoom_size,1)

        y = y_out_conv[:,0,:,0]
        y[:,zoom_min_ind:zoom_max_ind]=y_zoom_out

        #y = y.squeeze(-1).squeeze(1)

        return y


class _RegularDownUnet2D(nn.Module):

    def __init__(self,initial_filters=4,transients=40,input_channels=2,kernel_size=3):
        super(_RegularDownUnet2D,self).__init__()
        self.encoder = _DownUnet2DEncoder(4,input_channels,initial_filters,kernel_size)
        self.decoder = _DownUnet2DDecoder(4,initial_filters,kernel_size)

        self.mid_block_conv = _DownUnet2DBlock(initial_filters*(2**(3)),int(initial_filters*(2**4)),kernel_size=kernel_size)

    def forward(self,x,ppm=None):

        y,output_list = self.encoder(x)

        #print(y.shape)

        y = self.mid_block_conv(y)
        y = self.decoder(y,output_list)

        return y

class _ZoomMLPDownUnet2D(nn.Module):
    def __init__(self,initial_filters=4,transients=40,input_channels=2,kernel_size=3,zoom_size=300):
        super(_ZoomMLPDownUnet2D,self).__init__()

        self.zoom_size = zoom_size

        self.encoder = _DownUnet2DEncoder(4,input_channels,initial_filters,kernel_size)
        self.decoder = _DownUnet2DDecoder(4,initial_filters,kernel_size)

        self.mid_block_conv = _DownUnet2DBlock(initial_filters*(2**(3)),int(initial_filters*(2**4)*(3/4)),kernel_size=kernel_size)
        self.mid_block_mlp_1 = nn.Linear(int(self.zoom_size/2)*int(initial_filters)*24,int(self.zoom_size*initial_filters*20/4))

    def forward(self,x,ppm=None):

        y,output_list = self.encoder(x)

        #print(f"within zoom net enc out: {y.shape}")

        y_1 = self.mid_block_conv(y)

        #print(f"within zoom net y_1: {y_1.shape}")

        y_flat = y.reshape(y.shape[0],-1)
        y_out_1 = F.relu(self.mid_block_mlp_1(y_flat))
        y_2 = y_out_1.reshape(y.shape[0],int(y.shape[1]/2),y.shape[2],y.shape[3]-4)

        #y_2 = F.relu(self.mid_block_mlp(y.reshape(y.shape[0],-1))).reshape(y.shape[0],y_1.shape[1],y_1.shape[2],y_2.shape[3])

        #print("-0-0")
        #print(y_1.shape)
        #print(y_2.shape)

        y = torch.cat([y_1,y_2],axis=1)

        y = self.decoder(y,output_list)

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
            #print("----")
            #print(y.shape)
            #print(input_list[(self.steps-1-i)].shape)
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
        return F.relu(self.batch_norm_2(self.conv_2(F.relu(self.batch_norm_1(self.conv_1(x))))))
    
class _DownUnet2DBridgeBlock(nn.Module):

    def __init__(self, filters=10,kernel_size_x=3,kernel_size_y=3):
        super(_DownUnet2DBridgeBlock,self).__init__()

        self.conv = nn.Conv2d(filters,filters,kernel_size=(kernel_size_x,kernel_size_y),padding=(kernel_size_x//2,0))
        self.batch_norm = nn.BatchNorm2d(filters)
    
    def forward(self,x):
        y= F.relu(self.batch_norm(self.conv(x)))
        return y




