import librosa, soundfile
import torch
import torch.nn as nn
import numpy as np
from src.utils.util import Parsing,plot_losses,Pre_Parsing,plot_bs
from model.difftrans_timit import DIFFUSION_BIAS
from src.data_loader import get_dataloaders
import json, os, shutil
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import time

from transformers import Wav2Vec2Processor

if __name__ == "__main__":

    feature_dim=32
    id_dim=2
    start_time = time.time()
    data_path="./datasets/eval/traindata/YMY/wav"

    iteration = 50
    fps = 60
    processor = Wav2Vec2Processor.from_pretrained("./model_load/wav2vec2/")
    curve_norm = np.load(os.path.join(".", 'curve_norm_etc_32.npy'),allow_pickle=True).tolist()
    template_path = 'template.json'
    model_path = "./output/model/save_difftrans_test"

    model=DIFFUSION_BIAS(feature_dim,id_dim)
    model = model.to(torch.device("cuda"))
    model.load_state_dict(torch.load(os.path.join(model_path,'10000_model.pth')))
    model.eval()

    for r, ds, fs in os.walk(os.path.join(data_path)):
        for f in fs:
            if f.endswith('wav'):
                wav_path = os.path.join(r, f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)


                for id,speaker_id_name in [
                    [torch.tensor([1,0]).to(device="cuda"),"KTG"],
                    [torch.tensor([0,1]).to(device="cuda"),"YMY"],
                    ]:

                    with torch.no_grad():
                        predictdata = model.predict(torch.tensor(input_values).to(device="cuda"), iteration, fps,id= id)
                        res = predictdata.detach().cpu().squeeze()

                    eps = 1e-7
                    c_min, c_max = curve_norm['min'], curve_norm['max']
                    result_curve = (res - (-1)) / (1 - (-1) + eps) * (c_max - c_min + eps) + c_min
                    result_curve = np.clip(result_curve, 0, 1)

                    output=json.load(open(template_path,'rb'))
                    output=Parsing(result_curve,fps=60)

                    json.dump(output, open(os.path.join("./datasets/eval/traindata/YMY/Pred",speaker_id_name,f.split(".")[0]+".json"),'wt'))


                print('wav dur:', input_values.shape[0]/16000)
                print('curve dur:', result_curve.shape[0]/fps)