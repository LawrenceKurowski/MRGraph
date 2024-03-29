import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import MRGraph.utils as utils
from copy import deepcopy
from sklearn.metrics import f1_score
from MRGraph.layers import ResGraphConv, GraphConv,DenseGraphConv,MomGraphConv




class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, clip,dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, device=None):

        super(GCN, self).__init__()

        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GraphConv(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConv(nhid, nclass, with_bias=with_bias)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        
        self.clip = clip

    def forward(self, x, adj):
        
        if self.with_relu:
            x = F.relu(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=True, normalize=True, patience=500, **kwargs):
#         sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
#         dataloader = dgl.dataloading.NodeDataLoader(g, train_nids, sampler,batch_size=1024,shuffle=True,drop_last=False,num_workers=4)
#         self.device = self.gc1.weight.device
        self.to(self.device)
    
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        adj_norm = adj_norm
        self.adj_norm = adj_norm.to(self.device) 
        self.features = features
        self.labels = labels

        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            
            loss_train.backward()
            
            
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)

            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

#         scheduler = optim.ExponentialLR(optimizer, gamma=0.9)
        
        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)

            optimizer.step()
#             scheduler.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            
            loss_train.backward()
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
            
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self, idx_test):
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    def predict(self, features=None, adj=None):

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)



class ResGCN(GCN):
    def __init__(self, nfeat, nhid, nclass, clip,dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, device=None):

        super(ResGCN, self).__init__(nfeat,nhid,nclass)
        self.gc1 = ResGraphConv(nfeat, nfeat, with_bias=with_bias)
        self.gc2 = ResGraphConv(nhid, nhid, with_bias=with_bias)
        self.fc1 = torch.nn.Linear(nfeat, nhid)
        self.fc2 = torch.nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias

    def forward(self, x, adj):
        if self.with_relu:
            x = F.relu(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)
        x = self.fc1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    
class DenseGCN(GCN):
    def __init__(self, nblocks,nfeat, nhid, nclass,clip,dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, device=None):

        super(DenseGCN, self).__init__(nblocks,nfeat,nhid,nclass)
        
        self.gc1 = DenseGraphConv(nblocks,nfeat, nfeat, with_bias=with_bias)
        self.gc2 = DenseGraphConv(nblocks,nhid, nhid, with_bias=with_bias)
        
        self.fc1 = torch.nn.Linear(nfeat, nhid)
        self.fc2 = torch.nn.Linear(nhid, nclass)
        
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias

    def forward(self, x, adj):
        
        x = F.relu(self.gc1(x, adj)+x)
        x = self.fc1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj)+x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    
class MomGCN(GCN):
    def __init__(self, nblocks,nnodes,nfeat, nhid, nclass,clip,device,dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True):

        super(MomGCN, self).__init__(nblocks,nnodes,nfeat,nhid,nclass)
        self.gc1 = MomGraphConv(nblocks,nnodes,nfeat, nfeat, device,with_bias=with_bias)
        self.gc2 = MomGraphConv(nblocks,nnodes,nhid, nhid, device,with_bias=with_bias)
        
        self.fc1 = torch.nn.Linear(nfeat, nhid)
        self.fc2 = torch.nn.Linear(nhid, nclass)
        
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias

    def forward(self, x, adj):
#         x = x.to(self.device)
#         adj = adj.to(self.device)
        x = F.relu(self.gc1(x, adj)+x)
        x = self.fc1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj)+x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)