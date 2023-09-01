import torch
import torch.nn as nn
import torch.nn.functional as F


class CombineCNN(nn.Module):

    def __init__(self,cnn_steps=4,filters=4,input_channels=3,kernel_sizes=[3,5,7,9]):

        super(CombineCNN,self).__init__()

        self.size=2048

        self.ccn_chain = CNN1DMultiChain(depth=cnn_steps,input_filters=input_channels,output_filters=filters,kernel_sizes=kernel_sizes)

        self.cnn_combine = nn.Conv1d(len(kernel_sizes)*filters,3,kernel_size=5,padding="same")

        self.mlp_1 = nn.Linear(4*self.size,self.size*4)
        self.mlp_2 = nn.Linear(self.size*4,self.size*2)
        self.mlp_3 = nn.Linear(self.size*2,self.size)


    def forward(self,x,ppm=None):

        y = F.relu(self.cnn_combine(self.ccn_chain(x))).reshape(x.shape[0],-1)
        y = torch.cat([y,x[:,2,:]],axis=1)
        y = F.relu(self.mlp_1(y))
        y = F.relu(self.mlp_2(y))
        y = self.mlp_3(y)

        y = y + x[:,2,:]
    
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

        self.conv_blocks = nn.ModuleList([nn.Conv1d(input_filters[i],output_filters,kernel_size=kernel_size,padding="same") for i in range(depth)])

    def forward(self,x):
        y=x
        for i in range(self.depth):
            y = F.relu(self.conv_blocks[i](y))
        return y

