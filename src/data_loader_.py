import os
import torch
from collections import defaultdict
from torch.utils import data
from src.utils.util import Pre_Parsing
import torch
import numpy as np
import pickle
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa ,json
from glob import glob
from tqdm import tqdm

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.data_type = data_type

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        id=self.data[index]["id"]
        fps=self.data[index]["fps"]
 
        return torch.FloatTensor(audio), vertice, file_name,id ,fps

    def __len__(self):
        return self.len
    

def read_data(data_path):
    print("Loading data...")
    processor = Wav2Vec2Processor.from_pretrained("./model_load/wav2vec2/") # HuBERT uses the processor of Wav2Vec 2.0
    
    duration_limit_secs = 15 # 30 secs

    subjects_dict = {
            'train': [
                'KTG_wav-A2FBS',
                'YMY_wav-A2FBS'
            ],
            'test': [
                'KTG_wav-A2FBS',
                'YMY_wav-A2FBS'
            ]
        }
    one_hot_labels = np.eye(len(subjects_dict["train"]))

    audio_list = []
    for r, ds, fs in os.walk(os.path.join(data_path)):
        for f in fs:
            if f.endswith('wav'):
                audio_list.append((f, r))

    data_splits = {
        'train':[],
        'test':[],
    }
    data = []
    for file, root_dir in tqdm(audio_list, desc="Loading data"):
        if file.endswith('wav'):
            wav_path = os.path.join(root_dir, file)
            speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
            if len(speech_array) / 16000 > duration_limit_secs:
                print(f'{wav_path} is may too long ({len(speech_array)/16000} secs > {duration_limit_secs}): SKIP')
                continue
            input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
            subject_id = root_dir.split("\\")[-1]
            result = {}
            result["audio"] = input_values
            result["name"] = subject_id+"_"+file.replace(".wav", "")
            result["path"] = os.path.abspath(wav_path)
            result["id"] = torch.FloatTensor(one_hot_labels[subjects_dict["train"].index(subject_id)])
            if subject_id == "KTG_wav-A2FBS":
                vertice_file_name="YMY_KTG-rvc.json_bsweight_"+file
            else:
                vertice_file_name=file.split("_")[0]+"_bsweight_sample_"+file.split("_")[1]
            vertice_path = os.path.join(root_dir, vertice_file_name.replace("wav", "json"))
            if not os.path.exists(vertice_path):
                result= None
            else:
                with open(vertice_path, 'r', encoding='utf-8') as file:
                    vertice_data = json.load(file)
                result["vertice"] = torch.tensor(Pre_Parsing(vertice_data)) # torch.tensor(vertice_data['weightMat'])
                result["fps"]=vertice_data["exportFps"] 

        if result is not None:
            data.append(result)
        else:
            print("Warning: data not found")
    
    curve_value_min = np.array(data[0]["vertice"][0])
    curve_value_max = np.array(data[0]["vertice"][0])
    eps = 1e-10

    for result in data:
        curve_set = np.vstack([curve_value_min, np.min(result["vertice"].numpy(), axis=0)])
        curve_value_min = np.min(curve_set, axis=0)

        curve_set = np.vstack([curve_value_max, np.max(result["vertice"].numpy(), axis=0)])
        curve_value_max = np.max(curve_set, axis=0)
    
    for i in range(len(data)):
        result = data[i]
        c_norm = (result["vertice"] - curve_value_min + eps) / (curve_value_max - curve_value_min + eps) * (1 - (-1)) + (-1)
        data[i]["vertice"] =  c_norm

    
    splits = {
            'train':range(1,450),
            'test':range(1,450)
            }
    
    for result in data:
        subject_id ="_".join(result["name"].split("_")[:-2])
        sentence_id = int(result["name"][-4:])
        for sub in ['train', 'test']:
            if subject_id in subjects_dict[sub] and sentence_id in splits[sub]:
                data_splits[sub].append(result)


    return data_splits, {'min':curve_value_min, 'max':curve_value_max, 'eps':eps}
    
from torch.nn.utils.rnn import pad_sequence


def pad_collate(batch):
    audios, vertices, file_names, id, fps = zip(*batch)

    max_audio_length = max(audio.shape[0] for audio in audios)
    max_vertice_length = max(vertice.shape[0] for vertice in vertices)
    batch_size = len(audios)
    vertex_size = vertices[0].shape[1]

    padded_audios = torch.zeros(size=(batch_size, max_audio_length))
    audio_masks = torch.zeros(size=(batch_size, max_audio_length))
    padded_vertices = torch.zeros(size=(batch_size, max_vertice_length, vertex_size))
    vertice_masks = torch.zeros(size=(batch_size, max_vertice_length))
    ids=torch.zeros(size=(batch_size, len(id[0])))

    for i in range(batch_size):
        c_audio_size = audios[i].shape[0]
        padded_audios[i,:c_audio_size] = audios[i]
        audio_masks[i,:c_audio_size] = 1
        c_vertex_size = vertices[i].shape[0]
        padded_vertices[i,:c_vertex_size,:] = vertices[i]
        vertice_masks[i,:c_vertex_size] = 1
        ids[i]=id[i]

    
    return padded_audios, audio_masks, padded_vertices, vertice_masks, file_names, ids, fps # vertice_size


def get_dataloaders(audio_path):
    dataset = {}
    data_out, etc = read_data(audio_path)
    print("#########Data loaded!")
    train_data = Dataset(data_out["train"],"train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=2, shuffle=True,collate_fn=pad_collate)
    test_data = Dataset(data_out["test"],"test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return dataset, etc


if __name__ == "__main__":
    audio_path="./datasets/a2b_data"

    data = get_dataloaders(audio_path)
    

    # import IPython; IPython.embed()
    Test_Data=next(iter(data["train"]))
    print(Test_Data[0].shape, Test_Data[1][0].shape, Test_Data[2], Test_Data[3])
    Test_Data=next(next(iter(data["test"])))
    print(Test_Data[0].shape, Test_Data[1][0].shape, Test_Data[2], Test_Data[3])
    sum=0
    print(next(iter(data["test"])))
    for i in data["train"]:
        """print(i[0].shape[2])
        print(i[1].shape[1])
        print(i[3])
        print(i[2])"""
        if i[2][0] in ["0bHueQDd6TI.mp3","0EKAnNq874o.mp3","0GQt6r4eXUQ.mp3","0h-TRyWggeI.mp3","0qPSWm1XK9k.mp3","0NzU34sSadc.mp3","03_AL1jJ6eE.mp3","06Qy8Z798KU.mp3"]:
            print(i[0].shape[2]/16000)
            print(i[1].shape[1]/25)
            print(i[2])
        sum+=i[0].shape[2]/16000

