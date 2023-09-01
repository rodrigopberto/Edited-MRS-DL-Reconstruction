import copy
import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F


class ConvBlock_b(nn.Module):
    def __init__(self, ch_in, ch_out, k_size=5, stride=1, padding=1, bias=True):
        super(ConvBlock_b,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=k_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=k_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class UpBlock_b(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(UpBlock_b,self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.tconv1d = nn.ConvTranspose1d(ch_in, ch_in, 4, 2, 1, bias=False)
        self.o_conv = ConvBlock_b(ch_in+ch_out, ch_out, k_size=3, stride=1, padding=1, bias=False)

    def forward(self, in1, in2):
        in1 = self.tconv1d(in1)
        x = torch.cat((in1, in2), dim=1)
        x = self.o_conv(x)
        return x


class Backbone(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Backbone, self).__init__()
        self.Maxpool = nn.MaxPool1d(kernel_size=2,stride=2)
        self.ConvBlock_b4 = ConvBlock_b(ch_in, ch_in*2)
        self.ConvBlock_cc = ConvBlock_b(ch_in*2, ch_in*2)
        self.UpBlock_b4 = UpBlock_b(ch_in*2, ch_out)

    def forward(self, x):
        e4 = self.ConvBlock_b4(x)
        cc = self.Maxpool(e4)
        cc = self.ConvBlock_cc(cc)
        d4 = self.UpBlock_b4(cc, e4)
        return d4


class R_UNet(nn.Module):
    def __init__(self):
        super(R_UNet, self).__init__()
        self.prev_t = None
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.ConvBlock_b1 = ConvBlock_b(6, 8)
        self.ConvBlock_b2 = ConvBlock_b(8, 16)
        self.ConvBlock_b3 = ConvBlock_b(16, 32)
        self.SRU = SRU(32, 64, 3, backnet=Backbone(32, 64))
        self.UpBlock_b3 = UpBlock_b(32, 32)
        self.UpBlock_b2 = UpBlock_b(32, 16)
        self.UpBlock_b1 = UpBlock_b(16, 8)
        self.ConvBlock_f = ConvBlock_b(8, 3)

    def forward(self, x):
        if self.prev_t == None:
            self.prev_t = torch.zeros(x.size())
        x = torch.cat((x, self.prev_t), dim=1)
        e1 = self.ConvBlock_b1(x)
        e2 = self.Maxpool(e1)
        e2 = self.ConvBlock_b2(e2)
        e3 = self.Maxpool(e2)
        e3 = self.ConvBlock_b3(e3)
        e4 = self.Maxpool(e3)
        e4 = e4.unsqueeze(0)
        d4, _ = self.SRU(e4)
        d3 = self.UpBlock_b3(d4[0].squeeze(1), e3)
        d2 = self.UpBlock_b2(d3, e2)
        d1 = self.UpBlock_b1(d2, e1)
        x = self.ConvBlock_f(d1)
        self.prev_t = x

        return x
        

"""if __name__ == "__main__":
    device = torch.device("cuda:0")
    model = R_UNet().cuda()
    summary(model, torch.zeros((1, 3, 512, 512)).cuda(), show_input=True)
    summary(model, torch.zeros((1, 3, 512, 512)).cuda(), show_input=False)

    y = model(torch.zeros((1, 3, 512, 512)).cuda())"""



#####
class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, k_size=3, stride=1, padding=1, bias=True):
        super(ConvBlock,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=k_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class SRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, backnet):
        super(SRUCell, self).__init__()
        self.input_dim  = in_channels
        self.hidden_dim = hidden_channels
        self.update_gate = backnet
        self.out_gate = backnet

    def forward(self, input_tensor, cur_state):
        h_cur = cur_state
        x_in = input_tensor
        update = torch.sigmoid(self.update_gate(x_in))
        x_out = torch.tanh(self.out_gate(x_in))
        h_new = h_cur + x_out * (1 - update)
        return h_new

    def init_hidden(self, b, h, w):
        return torch.zeros(b, self.hidden_dim, h, w).cuda(1)


class SRU(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, batch_first=False, return_all_layers=False, backnet=None):
        super(SRU, self).__init__()
        hidden_channels = self._extend_for_multilayer(hidden_channels, num_layers)
        self.input_dim  = in_channels
        self.hidden_dim = hidden_channels
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers
        self.backnet = backnet
        self.conv = ConvBlock(64, 32)

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            cell_list.append(SRUCell(in_channels=cur_input_dim, hidden_channels=self.hidden_dim[i], backnet=self.backnet))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful DRU
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            b, _, _, h, w = input_tensor.shape
            hidden_state = self._init_hidden(b, h, w)

        layer_output_list = []
        last_state_list   = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :], cur_state=h)
                output_inner.append(h)

            output_inner[0] = self.conv(output_inner[0])
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, b, h, w):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(b, h, w))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param