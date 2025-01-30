import torch
from torch import nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        
        self.seg_size = configs.seg_size
        self.num_map = configs.num_map
        
        self.kernel_size = configs.kernel_size
        self.conv_stride = configs.conv_stride
        self.c = configs.c
        
        self.mappings = nn.ModuleList(
            [nn.Linear(configs.seq_len // self.seg_size, configs.pred_len // self.seg_size) for _ in range(self.num_map + 1)]
        )
        
        # for elec & traffic
        period = configs.period
        stride = period // configs.seg_size
        new_weights = torch.zeros(configs.pred_len // self.seg_size, configs.seq_len // self.seg_size)
        
        for i in range(0, configs.pred_len // self.seg_size):
            for j in range(configs.seq_len // self.seg_size - stride, 0, -stride):
                if j + i < configs.seq_len // self.seg_size:
                    new_weights[i, j+i] = period / configs.seq_len
        
        self.mappings[0].weight.data = new_weights
        self.mappings[0].bias.data = torch.zeros(configs.pred_len // self.seg_size)
        # self.mappings[0].weight.requires_grad = False
        
        self.conv_dim = (configs.seq_len - self.kernel_size) // self.conv_stride + 1
        
        # == MoE ==
        # self.ds_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, self.kernel_size), stride=(1, self.conv_stride))
        # self.gates = nn.Linear(self.conv_dim, self.num_map)
        
        # == MMoE ==
        self.ds_convs = nn.ModuleList(
            [nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.kernel_size, stride=self.conv_stride)
            for _ in range(self.c)]
        )
        
        self.gates = nn.Linear(self.conv_dim, self.num_map)
        
        self.dropout = nn.Dropout(configs.dropout)
        
        self.topk = configs.topk
        
    def forward(self, x):
        
        # input size: b, seq_len, c
        x = x.transpose(-2, -1)
        
        # input size: b, c, seq_len
        means = x.mean(2, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=2, keepdim=True, unbiased=False) + 1e-10)
        x /= stdev
        
        # conv_out = self.ds_conv(x.unsqueeze(1))
        conv_outs = [
            self.ds_convs[i](x[:,i,:].unsqueeze(1))
            for i in range(self.c)
        ]
        
        conv_out = torch.cat(conv_outs, dim=1)
        
        # [b,c,num_map]
        gates_out = self.gates(conv_out.squeeze(1))
        gates_out = F.softmax(gates_out, dim=-1)
        
        # if self.topk > 0:
        #     gates_out, topk_indice = torch.topk(gates_out, self.topk, dim=-1)
        #     # can test the effect of softmax
        #     gates_out = F.softmax(gates_out, dim=-1)
            
        
        bs, c, _ = x.shape
        x_ = x.reshape(bs, c, -1, self.seg_size).transpose(2, 3)
        
        # if self.topk > 0:
        #     new_gates_out = torch.zeros(bs, c, self.num_map).to(x.device)
        #     new_gates_out.scatter_(-1, topk_indice, gates_out)
        #     gates_out = new_gates_out
    
        # [b,c,seg_size]
        x_out = [
            self.mappings[i](x_).transpose(2, 3).flatten(start_dim=2)
            for i in range(self.num_map)
        ]
        
        # [b,c,num_map,seg_size]
        x_out = torch.stack(x_out, dim=2)
        
        x = torch.einsum("bcns,bcn->bcs", x_out, gates_out)
        
        x = x * stdev
        x = x + means
        
        x = x.transpose(-2, -1)
        
        return x