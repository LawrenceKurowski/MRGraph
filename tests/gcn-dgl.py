import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import argparse
import dgl.nn

dataset = dgl.data.CoraGraphDataset()
g=dataset[0]

# class GCN(nn.Module):
#     def __init__(self,
#                  g,
#                  in_feats,
#                  n_hidden,
#                  n_classes,
#                  n_layers,
#                  activation,
#                  dropout):
#         super(GCN, self).__init__()
#         self.g = g
#         self.layers = nn.ModuleList()
#         # input layer
#         self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
#         # hidden layers
#         for i in range(n_layers - 1):
#             self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
#         # output layer
#         self.layers.append(GraphConv(n_hidden, n_classes))
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, features):
#         h = features
#         for i, layer in enumerate(self.layers):
#             if i != 0:
#                 h = self.dropout(h)
#             h = layer(self.g, h)
#         return h

class GCN(nn.Module):
    def __init__(self,in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = dgl.nn.GraphConv(in_features, hidden_features)
        self.conv2 = dgl.nn.GraphConv(hidden_features, out_features)

    def forward(self, blocks, x):
        x = F.relu(self.conv1(blocks[0], x))
        x = F.relu(self.conv2(blocks[1], x))
        return x
    
in_feats = g
# Create the model with given dimensions
model = GCN(in_features=g.ndata['feat'].shape[1], hidden_features=16, out_features=dataset.num_classes)
opt = torch.optim.Adam(model.parameters())


def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.NodeDataLoader(g, features, sampler,batch_size=1024,shuffle=True,drop_last=False,num_workers=4)
    
    for e in range(200):
        
        for input_nodes, output_nodes, blocks in dataloader:
            
            blocks = [b.to(torch.device('cuda')) for b in blocks]
            
            input_features = blocks[0].srcdata['features']
            
            output_labels = blocks[-1].dstdata['label']
            
            output_predictions = model(blocks, input_features)
            output = output_predictions.argmax(1)
            
            loss = F.cross_entropy(output[train_mask], labels[train_mask])
            
            opt.zero_grad()
            
            loss.backward()
            
            opt.step()
        
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc


        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))

if __name__ == '__main__':
    train(g,model)
    
