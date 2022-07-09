import numpy as np
import pandas as pd
import utils
from model_config import build_args
import os
import json
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman') 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
from pylab import mpl
mpl.rcParams['font.size'] = 18

def temporal_curve():
    data_root="./original_data_new"
    subjects=["dzh","hzh","xxh","zhx","zrc"]
    subject=subjects[2]
    path=os.path.join(data_root,subject)
    input_file=os.listdir(path)[30]
    input_dir=os.path.join(path,input_file)
    print(input_dir)

    df=pd.read_csv(input_dir,header=None).T
    df=df.iloc[360:600].reset_index(drop=True)
    n_channel=df.shape[1]
    df.columns=['CH{}'.format(i+1) for i in range(n_channel)]
    df=df.rolling(window=5).mean()


    fig,ax1=plt.subplots(figsize=(7,3))
    df.plot(ax=ax1,legend=False)
    ax1.set_xlim(0,df.shape[0])
    ax1.set_xlabel('Time/ms')
    ax1.set_ylabel('Voltage/mV')
    plt.savefig(os.path.join("./figure","signal_sample_curve.pdf"),bbox_inches="tight",dpi=300)
    
    #Ledamap
    # sensor_loc_list=np.array([3,11,2,12,10,9,15,16,8,5,7,1,4,6,13,14])-1
    # group_list=[sensor_loc_list[:6],sensor_loc_list[6:12],sensor_loc_list[12:]]
    # band=["I","II","III"]
    # fig2,axes=plt.subplots(1,3,figsize=(12,5))
    # grid = plt.GridSpec(1, 3, wspace=0.6,hspace=0.4)
    # for i in range(3):
    #     band_data=df.iloc[:,group_list[i]]
    #     # print(band_data.info())
    #     ax=plt.subplot(grid[0,i],polar=True)
    #     values=band_data.mean()
    #     angles=np.linspace(0,2*np.pi,len(values),endpoint=False)
    #     values=np.concatenate((values,[values[0]]))
    #     angles=np.concatenate((angles,[angles[0]]))
    #     ax.plot(angles,values,'o-',color='#ff0000',linewidth=2)
    #     ax.fill(angles,values,color='#ff0000',alpha=0.25)
    #     ax.set_thetagrids(angles*180/np.pi,band_data.columns)
    #     if i!=2:
    #         ax.set_xlabel("\nArmband {}".format(band[i]),labelpad=10)
    #     else:
    #         ax.set_xlabel("Armband {}".format(band[i]),labelpad=10)
    
    # # plt.suptitle(input_dir)
    # plt.savefig(os.path.join("./figure","signal_sample_leda.pdf"),bbox_inches="tight",dpi=300)
    
def model_performance():
    result=[]
    for model_name in ["LDA","LSTM","ANN","CNN","GCN"]:
        print(model_name)
        args=build_args(model_name)
        file_dir=os.path.join(args.output_root,"model_performance")

        cm_path=file_dir+"/{}_CM.npy".format(model_name)
        acc_path=file_dir+"/{}_ACC.npy".format(model_name)
        cm=np.load(cm_path)
        acc=np.load(acc_path)
        print(acc)
        result.append(acc)

        CM=np.rint(cm.mean(axis=0)) #rint取整
        print(CM)
        utils.v_confusion_matrix(CM,args.part_actions,save_path="./figure"+"/{}_CM.pdf".format(args.model_name))
        
    result=pd.DataFrame(result,index=["LDA","LSTM","ANN","CNN","GCN"],columns=["S1","S2","S3","S4","S5"])
    result["mean"]=result.mean(axis=1)
    result["std"]=result.std(axis=1)
    print(result)
    result.to_csv(file_dir+"/model_performance.csv")

def used_time():
    result=[]
    for model_name in ["LDA","LSTM","ANN","CNN","GCN"]:
        print(model_name)
        args=build_args(model_name)
        file_dir=os.path.join(args.output_root,"model_usedtime")

        time_path=file_dir+"/{}_Time.npy".format(model_name)
        usedtime=np.load(time_path)
        result.append(usedtime)

    result=pd.DataFrame(result,index=["LDA","LSTM","ANN","CNN","GCN"],columns=["S1","S2","S3","S4","S5"])
    result["mean"]=result.mean(axis=1)
    result["std"]=result.std(axis=1)
    print(result)
    result.to_csv(file_dir+"/model_usedtime.csv")

def point_optimalzation_analysis():
    sensor_loc_list=np.array([3,11,2,12,10,9,15,16,8,5,7,1,4,6,13,14])-1
    loc_list=["B1","B2","B3","B4","B5","B6","C1","C2","C3","C4","C5","C6","A1","A2","A3","A4"]
    loc_dict=dict(zip(sensor_loc_list,loc_list))
    print(loc_dict)
    args=build_args(dataset="dataset2")
    save_path=os.path.join(args.output_root,"channel_reduce")

    sub_index=0
    model_list=["LSTM","ANN","CNN","GCN"]
    optimal_result=pd.DataFrame(index=model_list,columns=list(range(1,17))[::-1])
    fig,ax=plt.subplots(figsize=(8,4))
    for index,model_name in enumerate(model_list):
        with open(os.path.join(save_path,'channel_reduction_result_{}_{}_s{}.json'.format(args.dataset,model_name,sub_index)),"r") as json_file:
            result_dict=json.load(json_file)
        ch_list=[]
        score_list=[]
        keys=list(result_dict.keys())
        for n,key in enumerate(keys):
            if n == len(keys)-1:
                _keys=list(result_dict[key].keys())
                if key == _keys[0]:
                    ch_list.append(_keys[0])
                    ch_list.append(_keys[1])
                    score_list.append(result_dict[key][_keys[0]])
                    score_list.append(result_dict[key][_keys[1]])
                else:
                    ch_list.append(_keys[1])
                    ch_list.append(_keys[0])
                    score_list.append(result_dict[key][_keys[1]])
                    score_list.append(result_dict[key][_keys[0]])
            else:
                ch_list.append(key)
                score_list.append(result_dict[key][key])
        # print(ch_list)
        opti_loc=[loc_dict[int(ch)] for ch in ch_list]
        optimal_result.iloc[index,:]=opti_loc
        
        # print(score_list)
        if model_name == "GCN":
            model_name="GAM-Net(Ours)"
        plt.plot(score_list[::-1],marker="o",markersize=5,label=model_name)

    print(optimal_result)
    optimal_result.to_csv(os.path.join(save_path,"point_optimization_result_{}.csv".format(args.dataset)))
    plt.legend(loc=4)

    plt.xticks(range(len(score_list)),list(range(1,17)))
    plt.xlabel("Sensor Number k")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join("./figure","point_optimization_result_{}.pdf".format(args.dataset)),dpi=400,bbox_inches='tight')

def ablation_study_optim():
    args=build_args(dataset="dataset2")
    save_path=os.path.join(args.output_root,"channel_reduce")
    with open(os.path.join(save_path,'channel_reduction_result_dataset2_GCN_s0.json'),"r") as json_file:
        result_dict=json.load(json_file)
    ch_list=[]
    score_list=[]
    keys=list(result_dict.keys())
    for n,key in enumerate(keys):
        if n == len(keys)-1:
            _keys=list(result_dict[key].keys())
            if key == _keys[0]:
                ch_list.append(_keys[0])
                ch_list.append(_keys[1])
                score_list.append(result_dict[key][_keys[0]])
                score_list.append(result_dict[key][_keys[1]])
            else:
                ch_list.append(_keys[1])
                ch_list.append(_keys[0])
                score_list.append(result_dict[key][_keys[1]])
                score_list.append(result_dict[key][_keys[0]])
        else:
            ch_list.append(key)
            score_list.append(result_dict[key][key])
    print(ch_list)
    print(score_list)

    with open(os.path.join(save_path,'channel_reduction_result_random_select.json'),"r") as json_file:
        result_dict=json.load(json_file)
    random_score_list=[]
    for key in list(result_dict.keys()):
        random_score_list.append(np.mean(list(result_dict[key].values())))
    random_score_list.append(0.984)

    fig,ax=plt.subplots(figsize=(8,4))
    plt.plot(score_list[::-1],marker="o",color="#CC0000",markersize=5,label="Ours")
    plt.plot(random_score_list,linestyle="--",color="gray",marker="o",markersize=5,label="Random")
    plt.legend(loc=4)
    plt.xticks(range(len(score_list)),list(range(1,17)))
    plt.xlabel("Sensor Number k")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join("./figure","ablation_study_random_select.pdf"),dpi=400,bbox_inches='tight')

if __name__=="__main__":
    temporal_curve()
    model_performance()
    used_time()

    point_optimalzation_analysis()
    ablation_study_optim()
    # plt.show()
