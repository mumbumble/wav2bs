import librosa, soundfile
import torch
import torch.nn as nn
import numpy as np
from src.utils.util import Parsing,plot_losses,Pre_Parsing,plot_bs
from model.difftrans_timit import DIFFUSION_BIAS
from src.data_loader_timit import get_dataloaders
import json, os, shutil
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import time
from glob import glob

if __name__ == "__main__":

    feature_dim=52
    id_dim=2

    #data load
    #audio_path="//grai/GRAI_FacialAnimator/timit/data"
    # audio_path="./data/a2f"
    audio_path = "Y:\\voice_exp_dataset\\game_voice_wav_original"
    # curve_path = "Y:\\voice_exp_dataset\\game_voice_a2f_curve_RVC_Charlotte_full_0.1_resampled"
    model_path = "./output/model/save_difftrans_noid_52"
    template_path = 'template.json'

    curve_norm = np.load(os.path.join(".", 'curve_norm_etc_52.npy'),allow_pickle=True).tolist()

    #train
    model=DIFFUSION_BIAS(feature_dim,id_dim)
    model = model.to(torch.device("cuda"))

    model.load_state_dict(torch.load(os.path.join(model_path,'10000_model.pth')))
    model.eval()


    # wav_filelist = glob('\\\\grai\\GRAI_FacialAnimator\\Dataset\\EVALUATION_SET\\wav\\*.wav')
    wav_filelist = glob('C:/Users/parkhw5485/project/a2b/datasets/test/*.wav')
    for wav_filename in tqdm(wav_filelist):
        output_name = os.path.splitext(os.path.basename(wav_filename))[0] 
        
        print(wav_filename)
        wav, _ = librosa.load(wav_filename,sr=16000)

        iteration = 50
        fps = 60

        with torch.no_grad():
            predictdata = model.predict(torch.tensor(wav).to(device="cuda"), iteration, fps)
            res = predictdata.detach().cpu().squeeze()


        eps = 1e-7
        c_min, c_max = curve_norm['min'], curve_norm['max']
        result_curve = (res - (-1)) / (1 - (-1) + eps) * (c_max - c_min + eps) + c_min
        result_curve = np.clip(result_curve, 0, 1)

        output=json.load(open(template_path,'rb'))

        output['numFrames']=result_curve.shape[0]
        output['weightMat']=result_curve.tolist()

        json.dump(output, open(os.path.join('./out/noid',output_name+".json"),'wt'))


        print('wav dur:', wav.shape[0]/16000)
        print('curve dur:', result_curve.shape[0]/fps)
        
        # from IPython import embed;embed(header='test');exit()
