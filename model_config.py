import argparse

_CLASS_NAME = {"dataset1":["Keep_Still",
                            "Force1_Still","Static_Grip","Force2_Still","Force2_Motion",
                            "Force3_Still","Force3_Motion","Force4_Still","Force4_Motion",
                            "five_Finger_Open","Make_a_Fist","Scissor_Hand","five_fingers_Opening_and_Closing",
                            "Wrist_Swing","Wrist_Rotation","Wrist_Rotation_with_Gripping"],

                "dataset2":["Keep Still","Screw With Screwdriver","Lift A Bottle",
                            "Screw the Bottle Cap","Use Glue Gun","Lift Up Object","Cut The Rope",
                            "Tie A Rope","Rub a Surface with Sandpaper","WriTing","Wip The Table"]
}

_CLASS_NAME_SHORT = {"dataset1":["KS",
                                "F1S","SG","F2S","F2M",
                                "F3S","F3M","F4S","F4M",
                                "FO","MF","SH","OC",
                                "WS","WR","WRG"],

                    "dataset2":["KS","SWS","LAB",
                                "SBC","UGG","LUO","CTR",
                                "TAR","RSS","WT","WTT"]
}

_MODEL_HYPER_PARAMS = {
    "dataset1":{
            "LSTM":{
                "epochs":10,
                "batch_size":32,
                "lr":0.005,
                "hidden_dim":32,
                "n_layer":2   
            },
            "ANN":{
                "epochs":5,
                "batch_size":32,
                "lr":0.001,
                "hidden_dim":16,
                "n_layer":2   
            },
            "CNN":{
                "epochs":3,
                "batch_size":32,
                "lr":0.001,
                "hidden_dim":32,
                "n_layer":2   
            },
            "GCN":{
                "epochs":10,
                "batch_size":64,
                "lr":0.01,
                "hidden_dim":32,
                "n_layer":2   
            }
                },

    "dataset2":{
            "LSTM":{
                "epochs":4,
                "batch_size":32,
                "lr":0.005,
                "hidden_dim":32,
                "n_layer":2   
            },
            "ANN":{
                "epochs":6,
                "batch_size":32,
                "lr":0.001,
                "hidden_dim":16,
                "n_layer":2   
            },
            "CNN":{
                "epochs":3,
                "batch_size":32,
                "lr":0.001,
                "hidden_dim":32,
                "n_layer":2   
            },
            "GCN":{
                "epochs":10,
                "batch_size":64,
                "lr":0.01,
                "hidden_dim":32,
                "n_layer":2   
            }
                }
}

def build_args(model_name=None,dataset="dataset2"):
    parser = argparse.ArgumentParser("This script is used for the FMG-based Action Classification.")

    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--model_name", default="GCN", type=str)
    parser.add_argument("--model_file", default=None, type=str)
    args = parser.parse_args()

    args.gpu="0"
    args.dataset=dataset
    args.class_name=_CLASS_NAME_SHORT[args.dataset]
    args.class_name_long=_CLASS_NAME[args.dataset]
    args.class_dict={action_cls:idx for idx, action_cls in enumerate(args.class_name)}
    if args.dataset=="dataset1":
        args.data_root="./window_data"
        args.output_root="./output"
        args.repeat=2
        args.part_actions=args.class_name[0:1]+args.class_name[2:3]+args.class_name[9:] #choose part of actions
    elif args.dataset=="dataset2":
        args.data_root="./window_data_new"
        args.output_root="./output_new"
        args.repeat=3
        # args.part_actions=args.class_name[0:4]+args.class_name[5:7]+args.class_name[8:]
        args.part_actions=args.class_name[:]
    args.n_class=len(args.part_actions)
    args.action_dict={action_cls:idx for idx, action_cls in enumerate(args.part_actions)}
    print("number of class:{}\naction list:{}".format(args.n_class,args.part_actions))
    
    #prepare data
    args.L_win=150
    args.stride=1
    args.channels=list(range(16))   #choose part of channels
    print("length of window:{}\nstride of window:{}".format(args.L_win,args.stride))
    
    #train_test_subject
    args.subindex=0
    args.test_ratio=0.1

    if model_name is not None:
        args.model_name = model_name

    if model_name !="LDA":
        args.epochs=_MODEL_HYPER_PARAMS[args.dataset][args.model_name]["epochs"]
        args.batch_size=_MODEL_HYPER_PARAMS[args.dataset][args.model_name]["batch_size"]
        args.lr=_MODEL_HYPER_PARAMS[args.dataset][args.model_name]["lr"]
        args.hidden_dim=_MODEL_HYPER_PARAMS[args.dataset][args.model_name]["hidden_dim"]
        args.n_layer=_MODEL_HYPER_PARAMS[args.dataset][args.model_name]["n_layer"]

    return args

if __name__ == "__main__":
    build_args("GCN")

