import os
import utils
import numpy as np
import torch
import torch.nn as nn
from lda import LDAmodel
from model import LSTMnet,CNNnet,ANNnet,GCNnet
from dataset import FMGdataset
from model_config import build_args
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 

def main(args):
    os.environ['CUDA_VIVIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    args.device=device

    adj=None
    if args.model_name == "LSTM":
        model=LSTMnet(in_dim=len(args.channels),hidden_dim=args.hidden_dim,n_layer=args.n_layer,n_class=args.n_class)
    elif args.model_name == "ANN":
        model=ANNnet(in_dim=len(args.channels),hidden_dim=args.hidden_dim,n_layer=args.n_layer,n_class=args.n_class)
    elif args.model_name == "CNN":
        model=CNNnet(in_dim=len(args.channels),hidden_dim=args.hidden_dim,n_layer=args.n_layer,n_class=args.n_class)
    elif args.model_name == "GCN":
        model=GCNnet(input_dim=args.L_win,hidden_dim=args.hidden_dim,output_dim=args.n_class,num_channel=len(args.channels))
        adj=utils.get_adjmatrix(args)
    else:
        print("Model's name is not in the list!");return -1
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss(reduction='none')

    #train_val
    print("TrainVal_subjects_index:{}".format(args.subindex))
    train_dataset=FMGdataset(args,test_ratio=args.test_ratio,phase="train")
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=1)
    test_dataset=FMGdataset(args,test_ratio=args.test_ratio,phase="test")
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=1)

    if args.mode == "train":
        save_path="./models"
        utils.make_dir(save_path)
        best_acc = 0.0
        for epoch in range(args.epochs):
            utils.train(epoch,model,train_loader,optimizer,criterion,device,adj)
            _,acc,_=utils.test(epoch,model,test_loader,criterion,args.n_class,device,adj)
            if acc > best_acc:
                best_acc=acc
                torch.save(model.state_dict(),os.path.join(save_path,"best{}_{}_S{}.pth".format(args.model_name,args.dataset,args.subindex)))
            print("Current accuracy:{:.3f},Best accuracy:{:.3f}".format(acc,best_acc))
            print("#-----------------------------------------------------------------------#")
        return best_acc
    elif args.mode == "inference":
        print("inference start")
        model.load_state_dict(torch.load("./models/best{}_{}_S{}.pth".format(args.model_name,args.dataset,args.subindex)))
        cm,acc,t_mean=utils.test(0,model,test_loader,criterion,args.n_class,device,adj)
        print("Current accuracy:{:.3f}".format(acc))
        return cm,acc,t_mean

def get_model_performance():
    utils.set_seed(0)
    for model_name in ["LDA","ANN","CNN","LSTM","GCN"]:
        args=build_args(model_name)
        CM=np.zeros((5,args.n_class,args.n_class))
        ACC=np.zeros(5)
        for i in range(5):
            print("#-----------------------------------------------------------------------#")
            print(model_name,i)
            args.subindex=i
            if model_name == "LDA":
                cm,acc,_=LDAmodel(args)
            else:
                args.mode = "train"
                main(args)
                args.mode = "inference"
                cm,acc,_=main(args)
            CM[i,:,:]=cm
            ACC[i]=acc
        save_dir=os.path.join(args.output_root,"model_performance")
        utils.make_dir(save_dir)
        np.save(save_dir+"/{}_CM.npy".format(model_name),CM)
        np.save(save_dir+"/{}_ACC.npy".format(model_name),ACC)

def get_time_delay():
    for model_name in ["LDA","ANN","CNN","LSTM","GCN"]:
        print(model_name)
        args=build_args(model_name)
        Time=np.zeros(5)
        for i in range(5):
            args.subindex=i
            if model_name == "LDA":
                _,_,t_mean=LDAmodel(args)
            else:
                args.mode = "inference"
                _,_,t_mean=main(args)
            Time[i]=t_mean+args.L_win/1000
        save_dir=os.path.join(args.output_root,"model_usedtime")
        utils.make_dir(save_dir)
        np.save(save_dir+"/{}_Time.npy".format(model_name),Time)
        print(Time)
if __name__ == "__main__":
    
    # Normal Function
    utils.set_seed(0)
    model_name="GCN"
    args=build_args(model_name)
    args.subindex=0 #index of subjects
    if model_name == "LDA":
        cm,acc,t_mean=LDAmodel(args)
    else:
        args.mode = "train"
        main(args)
        args.mode = "inference"
        cm,acc,t_mean=main(args)
    print("Average inference time:{}".format(t_mean))
    title="Confision matrix of "+args.model_name
    utils.v_confusion_matrix(cm,args.part_actions,title=title,save_path="./figure/{}_CM.pdf".format(args.model_name))
    plt.show()

    #model_performance
    # get_model_performance()

    #Time_delay Analysis
    # get_time_delay()
