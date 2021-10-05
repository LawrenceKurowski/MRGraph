import torch
import numpy as np
import torch.nn.functional as F
from MRGraph.models import GCN,ResGCN,DenseGCN,MomGCN
from MRGraph.utils import *
from MRGraph.dataset import Dataset
import argparse

# import dgl
# import dgl.nn as dglnn


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--model', type=str, default='GCN',  choices=['GCN','ResGCN','DenseGCN','MomGCN'],help='model')
parser.add_argument('--clip', type=float, default=0.5, help='Gradient clip')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


clip = args.clip
# import wandb

# wandb.init(project="MRGraph", config={
#     "model": args.model,
#     "dataset": args.dataset})
# config = wandb.config


accuracies = []
seeds = [0,15,30,45,60]#[args.seed]



for seed in seeds:
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    
    data = Dataset(root='./data/', name=args.dataset, setting='gcn')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    
#     adj = adj.to(device)
#     features = features.to(device)
#     labels = labels.to(device)
    
    
    if args.model=='ResGCN':
        nfeat = features.shape[1]
        nhid = 16
        nclass = labels.shape[0]
        model = ResGCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1,clip=clip, dropout=0.5, lr=0.001, weight_decay=5e-4,with_relu=True, with_bias=True, device=device)
    elif args.model=='DenseGCN':
        model = DenseGCN(nblocks=2,nfeat=features.shape[1], nhid=16, nclass=labels.max()+1,clip=clip, device=device,lr=0.001)
    elif args.model=='MomGCN':
        model = MomGCN(nblocks=2,nnodes = features.shape[0], nfeat=features.shape[1], nhid=32, nclass=labels.max()+1,clip=clip, device=device,lr=0.001)
    else:
        model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1,clip=clip, device=device)
    
    adj,features,labels = preprocess(adj, features, labels, preprocess_adj=True, preprocess_feature=True, device=device)
    model = model.to(device)

    
    
    model.fit(features, adj, labels, idx_train, train_iters=275, verbose=False)
#     model.fit(features.to(device), adj.to(device), labels.to(device), idx_train, train_iters=200, verbose=False)
    model.eval()
    acc = model.test(idx_test)
    accuracies.append(acc)
#     wandb.log({"acc":acc})
    
print("(Test) accuracies: ",accuracies)
print("(Test) accuracy over", len(seeds),"runs: ",np.mean(accuracies))
# wandb.finish()
#, "loss":loss})