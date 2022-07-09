import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import networkx as nx
import random

plt.rc('font',family='Times New Roman') 
plt.rcParams['axes.unicode_minus'] = False
from pylab import mpl
mpl.rcParams['font.size'] = 18

def make_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def train(epoch,model,train_loader,optimizer,criterion,device,adj=None):
    model.train()
    running_loss = 0.0
    for data, label in tqdm(train_loader):
        label=label.to(device)
        
        if adj is not None:
            adj=adj.to(device)
            data=data.to(device)
            outputs=model(data,adj)
        else:
            data=data.to(device)
            outputs=model(data)

        loss = criterion(outputs, label)
        loss=loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()

    print("train epoch[{}] loss:{:.3f}".format(epoch+1,running_loss/len(train_loader)))


def test(epoch,model,test_loader,criterion,n_class,device,adj=None):
    t_start=time.perf_counter()
    model.eval()
    acc=0.0
    running_loss = 0.0
    confusion_matrix = torch.zeros(n_class, n_class)
    with torch.no_grad():
        for data, label in tqdm(test_loader):
            label=label.to(device)
            if adj is not None:
                adj=adj.to(device)
                data=data.to(device)
                outputs=model(data,adj)
            else:
                data=data.to(device)
                outputs=model(data)
            
            loss = criterion(outputs, label)
            loss=loss.mean()
            running_loss+=loss.item()
        
            predict_y=torch.max(outputs,dim=1)[1]
            label_y=torch.argmax(label)
            confusion_matrix[label_y.long(), predict_y.long()] += 1
            if predict_y[0] == label_y:
                acc += 1
    t_end=time.perf_counter()
    t_mean=(t_end-t_start)/len(test_loader)

    val_acc=acc/len(test_loader)
    confusion_matrix=confusion_matrix.detach().cpu().numpy()
    confusion_matrix=np.rint(100*confusion_matrix/confusion_matrix.sum(axis=1)[:, np.newaxis])

    print("test epoch[{}] loss:{:.3f}".format(epoch+1,running_loss/len(test_loader)))
    
    return confusion_matrix,val_acc,t_mean


def v_confusion_matrix(cm,class_list,title=None,save_path=None):
    cm=pd.DataFrame(cm,index=class_list,columns=class_list)

    plt.figure(figsize=(6,6))
    sns_plot=sns.heatmap(cm,annot=True,linewidth=0.5,fmt=".4g",cmap="binary",cbar=False)#cmap:'Reds/Blues','binary',YlGnBu','RdBu_r'
    sns_plot.tick_params(labelsize=16,direction='out')
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path,bbox_inches="tight",dpi=300)  

def select_channel(n,items):
    N = len(items)
    set_all=[]
    for i in range(2**N):
        combo = []  
        for j in range(N):   
            if(i >> j ) % 2 == 1:  
                combo.append(items[j])
        if len(combo) == n:
            set_all.append(combo)
    return set_all

def get_adjmatrix(args):
    sensor_loc_list=np.array([3,11,2,12,10,9,15,16,8,5,7,1,4,6,13,14])-1#minus 1:crresponding to index
    group_list=[sensor_loc_list[:6],sensor_loc_list[6:12],sensor_loc_list[12:]]

    graph_dict={}
    for group in  group_list:
        for i in group:
            if i in args.channels:
                connect_node=[]
                for j in group:
                    if j!=i and j in args.channels:#no circle
                        connect_node.append(j)
                graph_dict[i]=connect_node
    graph_dict=dict(sorted(graph_dict.items(),key=lambda item:item[0]))#sort the graph_dict by key
    # print(graph_dict)

    adj=nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict)).toarray()
    adj=preprocess_adj(adj)

    return adj

def preprocess_adj(adj):
    '''
    Pre-process adjacency matrix
    :param A: adjacency matrix
    :return:
    '''
    I = np.eye(adj.shape[0])
    A_hat = adj + I # add self-loops
    D_hat_diag = np.sum(A_hat, axis=1)
    D_hat_diag_inv_sqrt = np.power(D_hat_diag, -0.5)
    D_hat_diag_inv_sqrt[np.isinf(D_hat_diag_inv_sqrt)] = 0.
    D_hat_inv_sqrt = np.diag(D_hat_diag_inv_sqrt)
    A=np.dot(np.dot(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)
    return torch.from_numpy(A).float()
    
def sparse_dropout(x, rate, noise_shape):
    """
    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()
    dropout_mask=dropout_mask.bool()
    i = x._indices() 
    v = x._values() 

    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
    out = out * (1./ (1-rate))

    return out


