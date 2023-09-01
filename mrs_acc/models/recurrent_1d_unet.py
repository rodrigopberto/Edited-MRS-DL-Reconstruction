import torch
import torch.nn as nn
import torch.nn.functional as F


class Recurrent1DUnet(nn.Module):

    def __init__(self,steps=4,zoom_filters=16,long_filters=16,transients=40,input_channels=4,kernel_size=5):

        super(Recurrent1DUnet,self).__init__()

        self.steps=steps
        self.size=2048
        self.zoom_size=256
        self.zoom_min_ind=1084
        self.zoom_filters=zoom_filters


        self.encoder = _Unet1DEncoder(steps=steps,initial_filters=zoom_filters,input_channels=zoom_filters*2,kernel_size=kernel_size)
        self.decoder = _Unet1DDecoder(steps=steps,initial_filters=zoom_filters,kernel_size=kernel_size)

        self.mid_block = _Unet1DBlock(zoom_filters*(2**(steps-1)*3),zoom_filters*(2**steps),kernel_size=kernel_size)

        #self.mlp_output = nn.Linear(self.zoom_size*zoom_filters,self.zoom_size*zoom_filters)
        

        #self.unet_1d = Unet1D(steps=steps,initial_filters=zoom_filters,input_channels=zoom_filters*2)
        self.pre_conv = nn.Sequential(nn.Conv1d(in_channels=input_channels,out_channels=zoom_filters,kernel_size=kernel_size,padding="same"),
                                       nn.BatchNorm1d(zoom_filters))
        
        self.large_unet_1d = Unet1D(steps=steps,initial_filters=long_filters,input_channels=input_channels*transients)

        self.final_layer = nn.Conv1d(in_channels=zoom_filters,out_channels=1,kernel_size=kernel_size,padding="same")
        #self.final_layer = nn.Linear(2048*(zoom_filters+long_filters),2048)

    def forward(self,x,ppm=None):
        
        zoom_x = x[:,:,self.zoom_min_ind:self.zoom_min_ind+self.zoom_size]
        steps = x.shape[3]

        og_device = torch.get_device(x)
        

        connecting_output = torch.zeros(zoom_x.shape[0],self.zoom_filters,self.zoom_size).to(og_device)
        hidden_input = torch.zeros(zoom_x.shape[0],self.zoom_filters*(2**self.steps),self.zoom_size//(2**self.steps)).to(og_device)
        for step in range(steps):
            #print(hidden_input.shape)
            pre_x = F.relu(self.pre_conv(zoom_x[:,:,:,step]))
            
            #print(pre_x.shape)
            recu_input = torch.cat([connecting_output,pre_x],axis=1)
           # print(recu_input.shape)
            y,output_list = self.encoder(recu_input)
            #print("--")
            #print(y.shape)
            #print(hidden_input.shape)
            y_mid_input = torch.cat([y,hidden_input],axis=1)
            hidden_input = self.mid_block(y_mid_input)
            connecting_output = self.decoder(hidden_input,output_list)

            #print(recu_input.shape)
            #connecting_output = self.unet_1d(recu_input)
        
        #og_device = torch.get_device(x)
        #og_device="cpu"
        #print(og_device)
        #recu_out = torch.cat([torch.zeros(size=(x.shape[0],connecting_output.shape[1],self.zoom_min_ind)).to(og_device),
        #                     connecting_output,
        #                     torch.zeros(size=(x.shape[0],connecting_output.shape[1],x.shape[2]-self.zoom_size-self.zoom_min_ind)).to(og_device)]
        #                     ,axis=2)
        large_unet_input = x.permute(0,1,3,2).reshape(x.shape[0],-1,x.shape[2])
        long_out = self.large_unet_1d(large_unet_input)
        
        #print(recu_out.shape)
        #print(long_out.shape)


        #joint_out = torch.cat([recu_out,long_out],axis=1)
        joint_out = long_out
        #print(joint_out.shape)
        #print(connecting_output.shape)
        joint_out[:,:,self.zoom_min_ind:self.zoom_min_ind+self.zoom_size] = connecting_output[:,:,:]

        #print(joint_out.shape)
        
        output = self.final_layer(joint_out).squeeze(1)
        
        #print(output.shape)
        #raise Exception("stop")

        #print(output.shape)

        #print(output)
        return output

            





class Unet1D(nn.Module):

    def __init__(self,steps=4,initial_filters=16,input_channels=2,kernel_size=5):

        super(Unet1D,self).__init__()

        self.encoder = _Unet1DEncoder(steps,input_channels,initial_filters,kernel_size)
        self.decoder = _Unet1DDecoder(steps,initial_filters,kernel_size)

        self.mid_block = _Unet1DBlock(initial_filters*(2**(steps-1)),initial_filters*(2**steps),kernel_size=kernel_size)

        #self.last_layer = nn.Conv1d(initial_filters,1,kernel_size=kernel_size,padding="same")

    def forward(self,x,ppm=None):
        y = x
        #y = x.permute(0,1,3,2).reshape(x.shape[0],-1,x.shape[2])

        y,output_list = self.encoder(y)
        y = self.mid_block(y)
        y = self.decoder(y,output_list)

        #y = self.last_layer(y)

        #y = y.squeeze(1)

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