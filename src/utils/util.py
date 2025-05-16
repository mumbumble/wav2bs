
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import os

json_format={
    "exportFps": 30,
    "trackPath": "C:\\Users\\dtrca\\git\\FA-backend\\log\\462bc059-a9b8-4057-851d-8a433a8aebf3\\audio.wav",
    "numPoses": 52,
    "numFrames": 0,
    "facsNames": [
        "eyeBlinkLeft",
        "eyeLookDownLeft",
        "eyeLookInLeft",
        "eyeLookOutLeft",
        "eyeLookUpLeft",
        "eyeSquintLeft",
        "eyeWideLeft",
        "eyeBlinkRight",
        "eyeLookDownRight",
        "eyeLookInRight",
        "eyeLookOutRight",
        "eyeLookUpRight",
        "eyeSquintRight",
        "eyeWideRight",
        "jawForward",
        "jawLeft",
        "jawRight",
        "jawOpen",
        "mouthClose",
        "mouthFunnel",
        "mouthPucker",
        "mouthLeft",
        "mouthRight",
        "mouthSmileLeft",
        "mouthSmileRight",
        "mouthFrownLeft",
        "mouthFrownRight",
        "mouthDimpleLeft",
        "mouthDimpleRight",
        "mouthStretchLeft",
        "mouthStretchRight",
        "mouthRollLower",
        "mouthRollUpper",
        "mouthShrugLower",
        "mouthShrugUpper",
        "mouthPressLeft",
        "mouthPressRight",
        "mouthLowerDownLeft",
        "mouthLowerDownRight",
        "mouthUpperUpLeft",
        "mouthUpperUpRight",
        "browDownLeft",
        "browDownRight",
        "browInnerUp",
        "browOuterUpLeft",
        "browOuterUpRight",
        "cheekPuff",
        "cheekSquintLeft",
        "cheekSquintRight",
        "noseSneerLeft",
        "noseSneerRight",
        "tongueOut"
    ],
    "weightMat": []
}

facsNames= [
        "jawForward",
        "jawLeft",
        "jawRight",
        "jawOpen",
        "mouthClose",
        "mouthFunnel",
        "mouthPucker",
        "mouthLeft",
        "mouthRight",
        "mouthSmileLeft",
        "mouthSmileRight",
        "mouthFrownLeft",
        "mouthFrownRight",
        "mouthDimpleLeft",
        "mouthDimpleRight",
        "mouthStretchLeft",
        "mouthStretchRight",
        "mouthRollLower",
        "mouthRollUpper",
        "mouthShrugLower",
        "mouthShrugUpper",
        "mouthPressLeft",
        "mouthPressRight",
        "mouthLowerDownLeft",
        "mouthLowerDownRight",
        "mouthUpperUpLeft",
        "mouthUpperUpRight",
        "cheekPuff",
        "cheekSquintLeft",
        "cheekSquintRight",
        "noseSneerLeft",
        "noseSneerRight"
    ]


import torch

def Pre_Parsing(data):
    faceNameMat=data["facsNames"]   
    weightMat=torch.tensor(data["weightMat"], dtype=torch.float).transpose(0,1)
    weight_ar=torch.zeros(len(facsNames),weightMat.shape[1])
    for i in range(len(facsNames)):
        facsName=facsNames[i]
        if facsName in faceNameMat:
            weight_ar[i]=weightMat[faceNameMat.index(facsName)]

    return torch.transpose(weight_ar,0,1).tolist()

def Parsing(json_data,fps=30):
    returndata=json_format
    returndata["exportFps"]=fps
    returndata["numFrames"]=len(json_data)
    weightMat=torch.transpose(json_data,0,1)

    weight_ar=torch.zeros((len(json_format["facsNames"]),returndata["numFrames"]))
    for i in range(len(json_format["facsNames"])):
        facsName=json_format["facsNames"][i]
        if facsName in facsNames:
            weight_ar[i]=weightMat[facsNames.index(facsName)]
    
    returndata["weightMat"]=torch.transpose(weight_ar,0,1).tolist()
    return returndata



def plot_losses(train_losses,name):
    standard=[0.00005 for i in range(len(train_losses))]
    plt.plot(train_losses, label="Training loss")
    plt.plot(standard,label="Standard")
    plt.legend()
    plt.title("Losses"+name)
    plt.savefig("./output/loss/losses"+name+".png")
    plt.close()

def plot_bs(feature_dim,GT_vertice,log,epoch,log_save_path):
    transposed_log = [list(i) for i in zip(*log)]
    list_ = [sum(i)/len(i) for i in transposed_log]
    
    min_value=min(min([min(i) for i in log]),min(GT_vertice[100]))
    max_value=max(max([max(i) for i in log]),max(GT_vertice[100]))
    for i in range(1,feature_dim+1):
        name=facsNames[i-1]
        GT_data=[GT_vertice[100][i-1].item() for k in range(0,epoch)]
        log_value = log[i-1]
        plt.plot(log_value, label=f"{name}" )
        plt.plot(GT_data, label=f"GT_{name}")
        plt.yticks(np.arange(min_value,max_value,(max_value-min_value)/10))
        plt.legend()
        plt.title(f"{name}")
        plt.savefig(f"{log_save_path}/{name}.png")
        plt.close()

    GT_data=[sum(GT_vertice[100].tolist())/feature_dim for k in range(len(list_))]   
    plt.plot(list_, label="total")
    plt.plot(GT_data, label="GT_total")
    plt.yticks(np.arange(min(list_),max(list_),(max(list_)-min(list_))/10))
    plt.legend()
    plt.title("total")
    plt.savefig(f"{log_save_path}/total.png")
    plt.close()
    