import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset

class FMGdataset(Dataset):
    def __init__(self,args,test_ratio=0.3,phase="train"):
        self.model_name=args.model_name
        self.channels=args.channels
        self.actions=args.part_actions
        self.action_dict={action_cls:idx for idx, action_cls in enumerate(self.actions)}

        self.subjects=args.subindex
        self.L_win=args.L_win
        self.stride=args.stride
        self.test_ratio=test_ratio
        self.phase=phase

        self.get_basedata(args)
    def get_basedata(self,args):
        if not isinstance(self.subjects,list):
            self.subjects=[self.subjects]
        data_paths=[os.path.join(args.data_root,"L{}_s{}".format(self.L_win,self.stride),
                                "S{}_A{}_I{}".format(sub+1,args.class_dict[act]+1,i+1))+".npy"
                                 for sub in self.subjects for act in self.actions for i in range(args.repeat)]
        datalist=[]
        labellist=[]
        for data_path in data_paths:
            a_id=int(data_path.split("_")[-2][1:])-1
            a_class=args.class_name[a_id]
            datas=np.load(data_path)
            for data in datas:
                datalist.append(data)
                labellist.append(a_class)
        total_num=len(datalist)

        self.train_datalist,self.train_labellist=[],[]
        self.test_datalist,self.test_labellist=[],[]
        random.seed(0)
        test_idx=random.sample(range(0,total_num),int(total_num*self.test_ratio))
        for idx, data in enumerate(datalist):
            if idx in test_idx:  
                self.test_datalist.append(data)
                self.test_labellist.append(labellist[idx])
            else:
                self.train_datalist.append(data)
                self.train_labellist.append(labellist[idx])
        
        print("Total number of sample:{},train:{},test:{}".format(len(datalist),len(self.train_datalist),len(self.test_datalist)))

    def __len__(self):
        if self.phase == "train":
            return len(self.train_datalist)
        else:
            return len(self.test_datalist)

    def __getitem__(self,idx):
        if self.phase == 'train':
            datalist=self.train_datalist
            labellist=self.train_labellist
        else:
            datalist=self.test_datalist
            labellist=self.test_labellist
        #get data
        w_data=datalist[idx]
        w_data=w_data[:,self.channels]
        #get label
        a_class=labellist[idx]
        label=np.zeros(len(self.actions),dtype=np.float32)
        label[self.action_dict[a_class]]=1
        
        return torch.as_tensor(w_data.astype(np.float32)),torch.from_numpy(label)

    @staticmethod
    def collate_fn(batch):
        # return tuple(zip(*batch))
        return batch


if __name__ == "__main__":
    from model_config import build_args
    args=build_args("GCN")
    data=FMGdataset(args,phase="train")
    w_data,label=data.__getitem__(0)
    print(w_data.shape,label)
        
