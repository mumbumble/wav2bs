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
    duration_limit_secs = 30 # secs

    subjects_dict = [
                'KTG-wav-A2FBS',
                'YMY-wav-A2FBS'
            ]
    
    one_hot_labels = np.eye(len(subjects_dict))

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
        wav_path = os.path.join(root_dir, file)
        speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
        if len(speech_array) / 16000 > duration_limit_secs:
            print(f'{wav_path} is may too long ({len(speech_array)/16000} secs > {duration_limit_secs}): SKIP')
            continue
        input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
        root_name=root_dir.split("\\")[-1] 
        subject_id = 'KTG-wav-A2FBS' if root_name in ["YMY-wav-RVC-with-KTG","KTG-wav-A2FBS"] else 'YMY-wav-A2FBS'
        result = {}
        result["audio"] = input_values
        result["name"] = root_name+"_"+file.replace(".wav", "")
        result["path"] = os.path.abspath(wav_path)
        result["id"] = torch.FloatTensor(one_hot_labels[subjects_dict.index(subject_id)])
        if root_name == "YMY-wav-RVC-with-KTG":
            vertice_file_name="YMY_KTG-rvc.json_bsweight_"+file
        elif root_name =="KTG-wav-A2FBS":
            vertice_file_name=file
        else:
            vertice_file_name=file.split("_")[0]+"_bsweight_sample_"+file.split("_")[1]
        vertice_path = os.path.join(root_dir, vertice_file_name.replace(".wav", ".json"))
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


    ktg_numbers = [
            138, 514, 873, 771, 64, 747, 598, 315, 31, 79, 463, 583, 276, 121, 267, 733, 45, 433, 594, 779, 690, 193, 78, 324, 
            458, 591, 558, 211, 851, 631, 738, 596, 507, 108, 652, 32, 110, 319, 740, 683, 87, 214, 209, 811, 928, 695, 335, 
            531, 437, 765, 586, 310, 445, 399, 804, 515, 260, 533, 911, 24, 755, 362, 837, 881, 890, 678, 327, 465, 519, 478, 
            581, 434, 50, 320, 71, 137, 549, 896, 619, 343, 711, 602, 599, 540, 77, 6, 667, 199, 552, 236, 885, 724, 67, 545, 
            175, 240, 735, 875, 262, 73, 723, 364, 839, 718, 329, 823, 371, 306, 159, 245, 83, 759, 757, 358, 441, 97, 142, 
            421, 605, 351, 297, 333, 891, 406, 261, 366, 530, 865, 686, 169, 814, 61, 491, 454, 266, 345, 808, 870, 68, 232, 
            328, 248, 216, 431, 140, 883, 778, 559, 347, 378, 919, 912, 632, 111, 908, 801, 210, 299, 372, 929, 363, 920, 368, 
            847, 3, 413, 448, 166, 66, 466, 606, 673, 705, 40, 255, 877, 295, 696, 756, 544, 760, 932, 831, 34, 219, 909, 251
        ]
    ymy_numbers = [
            179, 358, 139, 353, 390, 149, 405, 397, 384, 374, 162, 401, 150, 170, 71, 65, 56, 167, 159, 176, 360, 62, 348, 409, 
            395, 63, 163, 330, 375, 425, 421, 359, 392, 154, 381, 347, 77, 349, 351, 66, 324, 32, 40, 112, 304, 26, 298, 128, 
            22, 293, 122, 130, 280, 279, 256, 133, 303, 281, 33, 117, 325, 113, 41, 316, 259
        ]

    splits = {
            'train': [[x for x in range(1,950) if x not in ktg_numbers],
                    [x for x in range(1,950) if x not in ymy_numbers]],
            
            'test':[ktg_numbers, ymy_numbers]}
    
    for result in data:
        subject_id =result["id"]
        sentence_id = int(result["name"][-4:])
        for sub in ['train', 'test']:
            if torch.equal(subject_id,torch.tensor([1.,0.])) and sentence_id in splits[sub][0]:
                data_splits[sub].append(result)
            elif torch.equal(subject_id,torch.tensor([0.,1.])) and sentence_id in splits[sub][1]:
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

