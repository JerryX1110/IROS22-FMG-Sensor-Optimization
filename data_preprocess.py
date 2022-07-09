import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt
from model_config import build_args
plt.rc('font',family='Times New Roman') 
plt.rcParams['axes.unicode_minus'] = False
from pylab import mpl
mpl.rcParams['font.size'] = 12


def plot_data(df,filename,mode="separate",save=False):
    if mode == "holdon":
        fig,ax=plt.subplots()
        df.plot(ax=ax,legend=True)
        fig.set_size_inches((10,3))
        ax.set_xlim(0,df.shape[0])
        plt.xlabel('Time/ms')
        plt.title(filename)

    if mode=="separate":
        df_mean=df.mean(axis=1)
        fig, ax=plt.subplots(8,2,sharex=True)
        plt.suptitle(filename)
        ax=ax.reshape(-1)
        for i in range(len(ax)):
            df.iloc[:,i].plot(ax=ax[i],color="black")
            df_mean.plot(ax=ax[i],color="r")

            ax[i].set_ylim(0,1)
            ax[i].set_xlim(0,df.shape[0])
            ax[i].set_ylabel("CH{}".format(i))
            ax[i].set_xlabel('Time/ms')
            ax[i].set_yticks([0,1],[0,1])
            ax[i].tick_params(direction='in')
        fig.set_size_inches((12,7))
        if save:
            fig_ouputpath=os.path.join("./figure","Temporal_curve_new")
            utils.make_dir(fig_ouputpath)
            plt.savefig(os.path.join(fig_ouputpath,filename+".png"),bbox_inches="tight",dpi=300)
            plt.cla()
            plt.close("all")

def pre_process(file_path,savedir,filename,args):
    df=pd.read_csv(file_path,header=None).T #[T,C]
    #delete head's and tail's data
    df=df.iloc[50:-50].reset_index(drop=True)
    n_channel=df.shape[1]
    df.columns=['CH{}'.format(i+1) for i in range(n_channel)]
    #min_max Normalization
    df_norm= (df-df.min())/(df.max()-df.min())
    #window smooth
    df_roll=df_norm.rolling(window=10).mean()
    # plot_data(df_roll,filename,save=True)
    #rooling window
    # L_wins=[50,100,150,200,250]
    # strides=[1,3,5,10]
    # for L_win in L_wins:
    #     for stride in strides:
    L_win=args.L_win
    stride=args.stride
    window_data=get_window_data(df_roll,L_win,stride)
    path=os.path.join(savedir,"L{}_s{}".format(L_win,stride))
    utils.make_dir(path)
    np.save(os.path.join(path,filename+".npy"),window_data) 

def get_window_data(data,L_win,stride):#data:[T,C]
    data=data.values
    data[np.isnan(data)] = 0
    window_data=[]
    start=0
    while (start+L_win-1) < data.shape[0]:
        window_data.append(data[start:start+L_win,:])
        start += stride
    return window_data
    
def make_data():
    args=build_args()
    input_dir="./original_data_new"
    subjects=["dzh","hzh","xxh","zhx","zrc"]
    for i,subject in tqdm(enumerate(subjects),total=len(subjects)):
        sub_path=os.path.join(input_dir,subject)
        sub_files=os.listdir(sub_path)
        for j,act in tqdm(enumerate(args.class_name),total=len(args.class_name)):
            for r in range(args.repeat):
                pre_process(os.path.join(sub_path,sub_files[args.repeat*j+r]),args.data_root,"S{}_A{}_I{}".format(i+1,j+1,r+1),args)
                
if __name__ == "__main__":
    make_data()


    



    

        

