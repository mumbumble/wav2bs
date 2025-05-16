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
        fps=self.data[index]["fps"]
 
        return torch.FloatTensor(audio), vertice, file_name ,fps

    def __len__(self):
        return self.len


def read_data(audio_path, curve_path):
    print("Loading data...")
    # train_data = []
    # valid_data = []
    # test_data = []

    # processor = Wav2Vec2Processor.from_pretrained("./model_load/wav2vec2/") # HuBERT uses the processor of Wav2Vec 2.0

    """template_file = os.path.join(args.dataset, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')"""
    
    duration_limit_secs = 20 # 30 secs
    
    audio_list = glob(os.path.join(audio_path,'*.wav'))
    curve_list = glob(os.path.join(curve_path,'*.json'))
    pair_list = []
    for audio_filename in audio_list:
        compare_name = os.path.splitext(os.path.basename(audio_filename))[0]
        chunk = (audio_filename, None)
        for curve_filename in curve_list:
            if compare_name in curve_filename:
                chunk = (chunk[0], curve_filename)
                break
        
        if chunk[1] is None:
            raise Exception(f'{chunk} has None')

        audio, _ = librosa.load(audio_filename, sr=16000)
        if len(audio) / 16000 > duration_limit_secs:
            print(f'{audio_filename} is may too long ({len(audio)/16000} secs > {duration_limit_secs}): SKIP')
            continue

        pair_list.append(chunk)

    data_pair_list = []
    for audio_filename, curve_filename in tqdm(pair_list):
        audio, _ = librosa.load(audio_filename, sr=16000)
        curve_data = json.load(open(curve_filename,'rb'))
        fps = curve_data['exportFps']
        curve = np.array(curve_data['weightMat'])

        data_pair_list.append((audio, curve))

        # folder_name=r.split('\\')[-1]
        # data = defaultdict(dict)
        # for f in tqdm(fs):
        #     if f.split(".")[-1] in ["wav","mp3"]:
        #         mov_path = os.path.join(r,f)
        #         speech_array, sampling_rate = librosa.load(mov_path, sr=16000)
        #         input_values = processor(speech_array, return_tensors="pt", padding="longest", sampling_rate=sampling_rate).input_values
        #         key = f.replace("wav", "npy")
        #         data[key]["audio"] = input_values[0]
        #         subject_id = "_".join(key.split("_")[:-1])
        #         data[key]["name"] = folder_name+'_'+f
        #         vertice_path_wav=  os.path.join(r,(f.split("_")[0]+"_bsweight_sample_"+ f.split("_")[1]).replace("wav", "json"))
        #         if os.path.exists(vertice_path_wav):
        #             with open(vertice_path_wav, 'r', encoding='utf-8') as file:
        #                 vertice_data = json.load(file)
        #         else:
        #             del data[key]
        #             break
        #         data[key]["vertice"] = torch.tensor(Pre_Parsing(vertice_data))
        #         data[key]["fps"]=vertice_data["exportFps"] 

        # length=len(train_data)
        # for k, v in data.items():
        #     if len(train_data)>=400:#length+len(data)-10:
        #         test_data.append(v)

        #     else:
        #         train_data.append(v)
    
    if len(data_pair_list) == 0:
        raise Exception('no data pair')

    curve_value_min = np.array(data_pair_list[0][1][0])
    curve_value_max = np.array(data_pair_list[0][1][0])
    eps = 1e-10

    for audio, curve in data_pair_list:
        curve_set = np.vstack([curve_value_min, np.min(curve, axis=0)])
        curve_value_min = np.min(curve_set, axis=0)

        curve_set = np.vstack([curve_value_max, np.max(curve, axis=0)])
        curve_value_max = np.max(curve_set, axis=0)
    
    for i in range(len(data_pair_list)):
        audio, curve = data_pair_list[i]
        c_norm = (curve - curve_value_min + eps) / (curve_value_max - curve_value_min + eps) * (1 - (-1)) + (-1)
        data_pair_list[i] = (audio, c_norm)

    actual_data_list = []
    for i in range(len(data_pair_list)):
        actual_data_list.append(
            {
                'name': os.path.splitext(os.path.basename(pair_list[i][0]))[0],
                # 'audio':torch.unsqueeze(torch.tensor(data_pair_list[i][0],dtype=torch.float32), 0),
                # 'vertice':torch.unsqueeze(torch.tensor(data_pair_list[i][1],dtype=torch.float32), 0),
                'audio':torch.tensor(data_pair_list[i][0],dtype=torch.float32),
                'vertice':torch.tensor(data_pair_list[i][1],dtype=torch.float32),
                'fps':np.float32(fps)
            }
        )
    
    # simple
    ratio = 0.9
    actual_size = int(len(data_pair_list) * ratio)

    train_data = actual_data_list[:actual_size]
    test_data = actual_data_list[actual_size:]
    
    # from IPython import embed;embed(header='asdf');exit()


    return train_data, test_data, {'min':curve_value_min, 'max':curve_value_max, 'eps':eps, 'fps':fps}
    
from torch.nn.utils.rnn import pad_sequence


def pad_collate(batch):
    audios, vertices, file_names, fps = zip(*batch)

    max_audio_length = max(audio.shape[0] for audio in audios)
    max_vertice_length = max(vertice.shape[0] for vertice in vertices)
    batch_size = len(audios)
    vertex_size = vertices[0].shape[1]

    padded_audios = torch.zeros(size=(batch_size, max_audio_length))
    audio_masks = torch.zeros(size=(batch_size, max_audio_length))
    padded_vertices = torch.zeros(size=(batch_size, max_vertice_length, vertex_size))
    vertice_masks = torch.zeros(size=(batch_size, max_vertice_length))

    for i in range(batch_size):
        c_audio_size = audios[i].shape[0]
        padded_audios[i,:c_audio_size] = audios[i]
        audio_masks[i,:c_audio_size] = 1
        c_vertex_size = vertices[i].shape[0]
        padded_vertices[i,:c_vertex_size,:] = vertices[i]
        vertice_masks[i,:c_vertex_size] = 1

    # padded_audios = []
    # audio_masks = []
    # for audio in audios:
    #     pad_size = max_audio_length - audio.shape[0]
    #     padded_audio = torch.nn.functional.pad(audio, (0, pad_size), 'constant', 0)
    #     padded_audios.append(padded_audio)
    #     audio_masks.append(torch.cat([torch.ones(audio.shape[0]), torch.zeros(pad_size)]))

    # padded_vertices = []
    # vertice_masks = []
    # vertice_size=0
    # for vertice in vertices:
    #     vertice_size+=vertice.shape[0]
    #     pad_size = max_vertice_length - vertice.shape[0]
    #     padded_vertice = torch.nn.functional.pad(vertice, (0, 0, 0, pad_size), 'constant', 0)
    #     padded_vertices.append(padded_vertice)
    #     vertice_masks.append(torch.cat([torch.ones(vertice.shape[0]), torch.zeros(pad_size)]))
    
    # padded_audios = torch.stack(padded_audios)
    # audio_masks = torch.stack(audio_masks)
    # padded_vertices = torch.stack(padded_vertices)
    # vertice_masks = torch.stack(vertice_masks)

    # print(padded_audios.shape, audio_masks.shape, padded_vertices.shape, vertice_masks.shape)
    
    return padded_audios, audio_masks, padded_vertices, vertice_masks, file_names, fps, None # vertice_size


def get_dataloaders(audio_path, curve_path):
    dataset = {}
    train_data, test_data, etc = read_data(audio_path, curve_path)
    print("#########Data loaded!")
    train_data = Dataset(train_data)
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True,collate_fn=pad_collate)
    # valid_data = Dataset(valid_data,"val")
    # dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_data = Dataset(test_data,"test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return dataset, etc


if __name__ == "__main__":
    audio_path="//grai/GRAI_FacialAnimator/timit/data/TRAIN"

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

