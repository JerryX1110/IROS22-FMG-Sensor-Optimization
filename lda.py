import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from model_config import build_args
import utils
import time
plt.rc('font',family='Times New Roman') 
plt.rcParams['axes.unicode_minus'] = False
from pylab import mpl
mpl.rcParams['font.size'] = 12



#single subject
def load_data(args):
    action_dict={action_cls:idx for idx, action_cls in enumerate(args.part_actions)}
    if not isinstance(args.subindex,list):
        args.subindex=[args.subindex]
    data_paths=[os.path.join(args.data_root,
                "L{}_s{}".format(args.L_win,args.stride),
                "S{}_A{}_I{}".format(sub+1,args.class_dict[act]+1,i+1))+".npy" 
                for sub in args.subindex for act in args.part_actions for i in range(args.repeat)]
    
    alldata=[]
    alllabel=[]
    for data_path in data_paths:
        a_id=int(data_path.split("_")[-2][1:])-1
        a_class=args.class_name[a_id]
        a_id=action_dict[a_class]

        w_data=np.load(data_path)
        w_data=w_data.reshape((w_data.shape[0],-1))
        label=a_id*np.ones((w_data.shape[0],1),dtype=np.int)

        alldata.append(w_data)
        alllabel.append(label)
    alldata = np.vstack(alldata)
    alllabel = np.vstack(alllabel)

    return alldata,alllabel


def LDAmodel(args):
    alldata,alllabel = load_data(args)
    print("Shape of data:{}".format(alldata.shape))
    alllabel=np.squeeze(alllabel)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(alldata,alllabel,test_size=args.test_ratio)

    model=LDA()
    model.fit(Xtrain,Ytrain)
    score = model.score(Xtest,Ytest)
    print("score:{}".format(score))

    t_start=time.perf_counter()
    Y_pred=model.predict(Xtest)
    t_end=time.perf_counter()
    t_mean=(t_end-t_start)/len(Y_pred)

    cm=confusion_matrix(Ytest, Y_pred, labels=None, sample_weight=None)
    cm=np.rint(100*cm/cm.sum(axis=1)[:, np.newaxis])

    return cm,score,t_mean



def window_stride_analysis(args):
    strides=[1,3,5,10]
    L_wins=[50,100,150,200,250]
    allresult=[]
    for stride in strides:
        args.stride=stride
        result=[]
        for L_win in L_wins:
            args.L_win=L_win
            scores=[]
            for subject in range(5):
                args.subindex=subject
                _,score,_=LDAmodel(args)
                scores.append(score)
            score=sum(scores)/len(scores)
            # score=LDAmodel(Actions,subjects,L_win,stride)
            result.append(score)
        allresult.append(result)

    fig=plt.figure()
    for i,result in enumerate(allresult):
        plt.plot(L_wins,result,"-.p",label=strides[i])
    fig.set_size_inches((6,4))
    plt.legend(title="stride",loc='lower right',fontsize=8,labelspacing=0.05)
    plt.title("LDA")
    plt.xlabel("window_size")
    plt.ylabel("Accuracy")
    plt.savefig("./figure/window_stride_analysis_AVG_new.png",bbox_inches="tight",dpi=300)

if __name__  == "__main__":
    args=build_args("LDA")
    cm,score,t_mean=LDAmodel(args)

    save_path="figure"
    utils.make_dir(save_path)
    title="Confusion matrix of "+args.model_name
    utils.v_confusion_matrix(cm,args.part_actions,title=title,save_path="./figure/{}_CM.pdf".format(args.model_name))
    plt.show()

    # window_stride_analysis(args)