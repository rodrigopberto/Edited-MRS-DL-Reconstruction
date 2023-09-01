import torch
import torch.nn as nn
import torch.nn.functional as F


class CombineCNNZoom(nn.Module):

    def __init__(self,cnn_steps=4,filters=4,input_channels=3,kernel_sizes=[3,5,7,9]):

        super(CombineCNNZoom,self).__init__()

        self.size=2048
        self.zoom_size=256
        self.zoom_min_ind=1084

        self.ccn_chain_zoom = CNN1DMultiChain(depth=cnn_steps,input_filters=input_channels,output_filters=filters,kernel_sizes=kernel_sizes)

        self.ccn_chain_diff = CNN1DMultiChain(depth=cnn_steps,input_filters=1,output_filters=filters,kernel_sizes=kernel_sizes)

        self.cnn_diff_prep = nn.Conv1d(len(kernel_sizes)*filters,1,kernel_size=5,padding="same")

        self.mlp_1 = nn.Linear((len(kernel_sizes)*filters+1)*self.zoom_size,self.zoom_size*filters)
        self.mlp_2 = nn.Linear(self.zoom_size*filters,self.zoom_size*2)
        self.mlp_3 = nn.Linear(self.zoom_size*2,self.zoom_size)

        self.final_combine = nn.Linear(self.zoom_size+2048,2048)


    def forward(self,x,ppm=None):

        y_zoom = x[:,:,self.zoom_min_ind:self.zoom_min_ind+self.zoom_size]
        y_zoom_ccn_out = self.ccn_chain_zoom(y_zoom)
        y_zoom_mlp_in = torch.cat([y_zoom[:,2,:],y_zoom_ccn_out.reshape(x.shape[0],-1)],axis=1)

        y_zoom = self.mlp_3(F.relu(self.mlp_2(F.relu(self.mlp_1(y_zoom_mlp_in)))))

        y_cnn = self.ccn_chain_diff(x[:,2:,:])
        y_cnn_out = self.cnn_diff_prep(y_cnn).reshape(x.shape[0],-1)

        y_combine = torch.cat([y_zoom,y_cnn_out],axis=1)
        y_out = self.final_combine(y_combine)
    
        return y_out


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

        self.conv_blocks = nn.ModuleList([nn.Conv1d(input_filters[i],output_filters,kernel_size=kernel_size,padding="same") for i in range(depth)])

    def forward(self,x):
        y=x
        for i in range(self.depth):
            y = F.relu(self.conv_blocks[i](y))
        return y

