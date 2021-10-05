import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from copy import deepcopy
from sklearn.metrics import f1_score
import scipy.sparse as sp
from MRGraph.utils import *

class GraphConv(Module):
    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
    
    
    
    

class ResGraphConv(Module):
    def __init__(self, in_features, out_features, with_bias=True):
        super(ResGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        
        identity = sp.eye(in_features)
        self.identity = sparse_mx_to_torch_sparse_tensor(identity)

        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if input.data.is_sparse:
            support = torch.spmm(input, (self.weight+self.identity))
        else:
            support = torch.mm(input, (self.weight+self.identity))
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    
class DenseGraphConv(Module):
    def __init__(self, nblocks,in_features, out_features, with_bias=True):
        super(DenseGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nblocks=nblocks
        
        blocks =[]
        for i in range(nblocks):
            weight = Parameter(torch.FloatTensor(in_features, out_features))
            identity = sp.eye(in_features)
            identity = sparse_mx_to_torch_sparse_tensor(identity)
            weight = weight+identity
            blocks.append(weight)
        
        self.weight = blocks[0]
        self.blocks = blocks
    
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if input.data.is_sparse:
            support = torch.spmm(input, self.blocks[0])
            for i in range(self.nblocks-1):
                support = torch.spmm(support, self.blocks[i+1])
        else:
            support = torch.mm(input, self.blocks[0])
            for i in range(self.nblocks-1):
                support = torch.mm(support, self.blocks[i+1])
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
    
    
class MomGraphConv(Module):
    def __init__(self, nblocks,nnodes,in_features, out_features, device, with_bias=True):
        super(MomGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nblocks=nblocks
        self.device=device
        self.velocity = torch.ones(nnodes,in_features)
#         self.velocity = velocity.to(self.device)
#         self.velocity = sparse_mx_to_torch_sparse_tensor(velocity)
        self.gamma = .1#gamma
        
        blocks =[]
        for i in range(nblocks):
            weight = Parameter(torch.FloatTensor(in_features, out_features))
#             weight = weight.to(self.device)
            blocks.append(weight)
        
        self.weight = blocks[0]
        self.blocks = blocks
    
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
#         print('device = ',self.device)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        velocity = self.velocity.to(device)
#         print('before loop, velocity on cuda? ',velocity.is_cuda)
        if input.data.is_sparse:
            
            for i in range(self.nblocks):
                velocity = (1 - self.gamma) * torch.spmm(input, self.blocks[i]) + self.gamma * velocity
                x = velocity+input
        else:
            for i in range(self.nblocks):
                velocity = self.gamma * velocity
#                 print('velocity type: ', type(velocity))
#                 print('velocity shape: ', velocity.shape)
#                 print('is velocity cuda? ',velocity.is_cuda)
#                 print('velo')
                test = (1 - self.gamma) * torch.spmm(input, self.blocks[i])
#                 exit()
#                 print('test type: ',type(test))
#                 print('test shape: ',test.shape)
#                 print('is test cuda? ',test.is_cuda)
#                 print()
#                 velocity = velocity + (1 - self.gamma) * torch.spmm(input, self.blocks[i])
#                 exit()
                velocity= velocity + test
#                 exit()
                x = velocity+input
#                 exit()
        x = torch.spmm(adj,x)
        exit()
        if self.bias is not None:
            return x + self.bias
        else:
            return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'