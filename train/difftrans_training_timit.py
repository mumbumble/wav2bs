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


def trainer(save_path, train_loader, model, optimizer, criterion,GT_vertice,feature_dim,graph,start_epoch=0, epoch=600):
    now_epoch=start_epoch
    rec_losses = []
    vel_losses = []
    log_losses=[]

    rec_l = []
    vel_l = []
    log_l=[]    
    BS_log=[]
    for i in range(1,feature_dim+1):
        BS_log.append([])

    save_path = os.path.join(save_path)
    if not os.path.exists(save_path):
        #shutil.rmtree(save_path)
        os.makedirs(save_path)

    #train_subjects_list = [i for i in args.train_subjects.split(" ")]
    for e in range(start_epoch+1,start_epoch+epoch+1):
        iteration = 0
        rec_loss_log = []
        vel_loss_log = []
        loss_log=[]
        epoch_BS_log=[]
        for i in range(1,feature_dim+1):
            epoch_BS_log.append([])
        model.train()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        optimizer.zero_grad()

        for i, val in enumerate(train_loader):
            iteration += 1
            audio, audio_mask, vertice, vertice_mask , file_names,fps,vertice_size= val
            audio, audio_mask, vertice, vertice_mask   = audio.to(device="cuda"), audio_mask.to(device="cuda"),vertice.to(device="cuda"),vertice_mask.to(device="cuda")
            loss,vel_loss,rec_loss,vertice_out = model(audio, audio_mask, vertice, vertice_mask,criterion,fps)
            if graph in file_names :
                for i in range(1,feature_dim+1):
                    epoch_BS_log[i-1].append(vertice_out[0][100][i-1].item())
            loss.backward()
            loss_log.append(loss.item())
            rec_loss_log.append(rec_loss.item())
            vel_loss_log.append(vel_loss.item())
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.8f}".format((e), iteration ,np.mean(loss_log)))

        log_losses.append(np.mean(loss_log))
        rec_losses.append(np.mean(rec_loss_log))
        vel_losses.append(np.mean(vel_loss_log))
        for i in range(len(epoch_BS_log)):
            BS_log[i].append(np.mean(epoch_BS_log[i]))

        if e>500 and log_losses[-1]>0.001:
            return model, log_l,rec_l,vel_l,now_epoch

        if (e > 0 and e % 100 == 0) or e == epoch:
            now_epoch=e
            torch.save(model.state_dict(), os.path.join(save_path,'{}_model.pth'.format(e)))
            log_l.append(np.mean(log_losses[-100:]))
            rec_l.append(np.mean(rec_losses[-100:]))
            vel_l.append(np.mean(vel_losses[-100:]))
            log_save_path = os.path.join("./output/graph/"+graph+"_difftrans_"+str(e)+"/")
            if os.path.exists(log_save_path):
                shutil.rmtree(log_save_path)
            os.makedirs(log_save_path)

            plot_bs(feature_dim,GT_vertice,BS_log,epoch,log_save_path)

            name="difftrans_rec_loss_"+str(e)
            plot_losses(rec_losses,name)
            name="difftrans_velloss_"+str(e)
            plot_losses(vel_losses,name)

            li=[log_losses,rec_losses,vel_losses]
            arr = np.array(li)
            np.savetxt('ver1loss.csv', arr, delimiter=",")  
    
    

    return model, log_l,rec_l,vel_l,now_epoch

if __name__ == "__main__":

    feature_dim=32
    id_dim=2
    start_time = time.time()

    #data load
    #audio_path="//grai/GRAI_FacialAnimator/timit/data"
    audio_path="./data/a2f"

    dataset = get_dataloaders(audio_path)
    torch.cuda.empty_cache()
    #train
    model=DIFFUSION_BIAS(feature_dim,id_dim)
    model = model.to(torch.device("cuda"))

    Test_Data=next(iter(dataset["test"]))
    Test_name=Test_Data[2][0].split(".")[0]
    vertice_name=next(iter(dataset["train"]))[4][0]

    start_epoch=2200
    epoch=7800
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=0.00001)
    save_path = "./output/model/save_difftrans_3000/"
    
    trainstart_time = time.time()
    now_epoch=0

    while (start_epoch<start_epoch+epoch):
        if start_epoch!=0:
            model.load_state_dict(torch.load("./output/model/save_difftrans_3000/"+str(start_epoch)+"_model.pth"))
        print("load model   ./output/model/save_difftrans_3000/"+str(start_epoch)+"_model.pth")
        model,log_losses,rec_losses,vel_losses,start_epoch = trainer(save_path, dataset["train"], model, optimizer, loss_fn,Test_Data[1][0],feature_dim,vertice_name,start_epoch,epoch=epoch)

    #test
    teststart_time = time.time()
    print(f"Model-data loading time : {trainstart_time-start_time}")
    print(f"Training time : {teststart_time-trainstart_time}")
    model.load_state_dict(torch.load("./output/model/save_difftrans_3000/"+str(start_epoch)+"_model.pth"))


    model.eval()
    predictdata = model.predict(Test_Data[0][0].to(device="cuda"),50)

    savetime = time.time()
    predictdata=Parsing(predictdata)
    output_path="./output/bs_output/"
    output = Path(output_path+Test_name+"_difftrans_"+str(start_epoch)+".json") 
    with open(output, "w", encoding="utf-8") as f:
        json.dump(predictdata, f)

    print(f"output.py saved in {output}")

    endtime = time.time()

    print(f"Total time : {endtime-start_time}")
    print(f"Model-data loading time : {trainstart_time-start_time}")
    print(f"Training time : {teststart_time-trainstart_time}")
    print(f"Test time : {savetime-teststart_time}")
    print(f"Save time : {endtime-savetime}")