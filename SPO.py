import os
from model_config import build_args
import utils
from lda import LDAmodel
import json
import random


#Channel Split Analysis
def Search_all():
    from main import main
    args=build_args("GCN")
    args.subindex=0
    utils.make_dir("./output_new")
    for n in [1,2,3]:
        result={}
        channel_list=utils.select_channel(n,list(range(16)))
        print(channel_list)
        for channel in channel_list:
            name="_".join(str(c) for c in channel)
            print(name)

            args.channels=channel
            args.mode="train"
            ACC=main(args)
            print(ACC)
            result[name]=ACC

            save_path=os.path.join(args.output_root,"channel_reduce")
            utils.make_dir(save_path)
            json_str = json.dumps(result, indent=4)
            with open(os.path.join(save_path,'channel_result_searchall_{}.json'.format(n)), 'w') as json_file:
                json_file.write(json_str)

def SPO(ch_list,mode="max"):#Sensor placement optimization
    from main import main
    utils.set_seed(0)
    for model_name in ["LSTM","ANN","CNN","GCN"]:
        for sub_index in [0]:
            args=build_args(model_name,dataset="dataset2")
            args.subindex=sub_index #index of subjects

            current_chlist=ch_list.copy()
            result_dict={}
            while len(current_chlist) > 1:
                print("#--------------------------------------------#")
                print("current_chlist:{}".format(current_chlist))
                score_dict={}
                for c in current_chlist:
                    temp_chlist=current_chlist.copy()
                    temp_chlist.remove(c)
                    args.channels=temp_chlist
                    print(args.channels)

                    if model_name == "LDA":
                        _,acc,_=LDAmodel(args)
                    else:
                        args.mode = "train"
                        acc=main(args)
                           
                    print("Remove CH{},ACC:{}".format(c,acc))
                    score_dict[c]=acc
                if mode == "max":#remove most unimportant point
                    best_ch=max(score_dict,key=lambda k:score_dict[k])
                elif mode == "min":#remove most important point
                    best_ch=min(score_dict,key=lambda k:score_dict[k])
                current_chlist.remove(best_ch)
                result_dict[best_ch]=score_dict

                save_path=os.path.join(args.output_root,"channel_reduce")
                utils.make_dir(save_path)
                json_path=os.path.join(save_path,'channel_reduction_result_{}_{}_s{}.json'.format
                                        (args.dataset,model_name,sub_index))
                json_str = json.dumps(result_dict, indent=4)
                with open(json_path, 'w') as json_file:
                    json_file.write(json_str)

def Random_Select():
    from main import main
    utils.set_seed(0)
    model_name="GCN"
    sub_index=0
    args=build_args(model_name,dataset="dataset2")
    args.subindex=sub_index

    ch_list=list(range(16))
    result_dict={}
    for n in range(15):
        channel_list=utils.select_channel(n+1,ch_list)

        channel_list=random.sample(channel_list,5)#random select 5 combination
        channel_result={}
        for channel in channel_list:
            name="_".join(str(c) for c in channel)
            print(name)
            args.channels=channel
            if model_name == "LDA":
                _,acc,_=LDAmodel(args)
            else:
                args.mode = "train"
                acc=main(args)
            channel_result[name]=acc
        result_dict[n+1]=channel_result

        save_path=os.path.join(args.output_root,"channel_reduce")
        utils.make_dir(save_path)
        json_path=os.path.join(save_path,'channel_reduction_result_random_select.json')
        json_str = json.dumps(result_dict, indent=4)
        with open(json_path, 'w') as json_file:
            json_file.write(json_str)
            

if __name__ == "__main__":
    SPO(ch_list=list(range(16)))
    # Random_Select()
    

