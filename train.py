import torch
#from mrs_acc.pipelines import BasePipeline
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import numpy as np
import h5py
import math
import sys
from omegaconf import OmegaConf
from mrs_acc.config import Config
from mrs_acc.utils import get_dataset,get_pipeline,get_loss_function,get_transform
import datetime as dt
from pathlib import Path
from tqdm import tqdm
from mrs_acc.reporting_json import create_training_json
import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:780"

def train(config_filename: str):
    config_dict = OmegaConf.to_object(OmegaConf.load(config_filename))
    config = Config(**config_dict)

    print(config)
    print("-------")  

    pipe = get_pipeline(config)

    train_transform = get_transform(config.train)
    val_transform = get_transform(config.validation)
    
    
    train_dataset = get_dataset(config.train,train_transform)
    val_dataset = get_dataset(config.validation,val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    
    # loads model from checkpoint if given starting file
    if config.starting_checkpoint != None:
        pipe.load_model_weight(config.starting_checkpoint)

    # variables for reporting and results saving
    training_date_str = dt.datetime.strftime(dt.datetime.now(),"%Y%m%d")
    run_name = config.name
    full_name = training_date_str+"_"+run_name

    #creating saving/reporting dirs for 
    Path(config.train.weight_output_folder).mkdir(exist_ok=True)
    #Path(config.train.weight_output_folder+f"/{full_name}").mkdir(exist_ok=True)
    Path(config.train.json_output_folder).mkdir(exist_ok=True)

    loss_fn = get_loss_function(config.train.loss_fn).to(config.device)

    val_loss_fn = get_loss_function(config.validation.loss_fn).to(config.device)

    weight_filename = f"{config.train.weight_output_folder}/{full_name}.pth"

    optim = torch.optim.Adam(pipe.model.parameters(), lr=config.train.learning_rate)

    last_loss = 1000000

    if config.train.scheduler_freq!=None:  
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optim,lambda lr: 0.5)
    
    # auxiliary lists/variables for logging results
    train_losses = []
    val_losses=[]
    epoch_count = config.train.epochs
    early_stopping_tresh = config.train.early_stop
    keep_best_tresh = config.train.keep_best_treshold

    early_stopping_counter=0
    keep_best_counter=0
    for epoch in range(epoch_count):
        epoch_train_losses = []
        for step, data in enumerate(tqdm(train_loader, position=0, leave=True)):
            
            loss = pipe.train_step(data,loss_fn,optim)
            epoch_train_losses.append(loss)
            

        epoch_val_losses = []
        for data in val_loader:
            val_loss = pipe.val_step(data,val_loss_fn)
            epoch_val_losses.append(val_loss)
            
           
        
        if config.train.scheduler_freq!=None:
            if (epoch+1)%config.train.scheduler_freq==0:
                lr_scheduler.step()


        train_loss = torch.Tensor(epoch_train_losses).mean().item()
        val_loss = torch.Tensor(epoch_val_losses).mean().item()
        print(f"epoch {epoch+1}/{epoch_count} - loss: {train_loss} - validation loss: {val_loss}")


        

        if val_loss>last_loss:
            keep_best_counter+=1
            early_stopping_counter+=1
            if keep_best_counter>keep_best_tresh:
                pipe.load_model_weight(weight_filename)
                keep_best_counter = 0
            if early_stopping_tresh!=None:
                if early_stopping_counter>early_stopping_tresh:
                    break
        else:
            keep_best_counter=0
            early_stopping_counter=0
            pipe.save_model_weight(weight_filename)
            last_loss=val_loss

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        #break


    #final save
    pipe.save_model_weight(weight_filename)

    out_json = create_training_json(config,training_date_str,full_name,train_losses,val_losses)

    with open(f"{config.train.json_output_folder}/{full_name}.json","w+") as f:
        json.dump(out_json,f,indent=4)
        f.close()


if __name__=="__main__":
    train(sys.argv[1])