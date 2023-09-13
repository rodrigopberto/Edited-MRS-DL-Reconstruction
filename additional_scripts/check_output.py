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
from mrs_acc.utils import get_dataset,get_pipeline,get_transform,get_metric_function_dict
from mrs_acc.reporting_json import create_testing_json
import datetime as dt
from pathlib import Path
from tqdm import tqdm
import json

def test_model(config_filename: str, weight_filename:str):

    config_dict = OmegaConf.to_object(OmegaConf.load(config_filename))
    if weight_filename!=None:
        config_dict["starting_checkpoint"]=weight_filename
    config = Config(**config_dict)

    print(config)
    print("-------")  

    pipe = get_pipeline(config)

    test_transform = get_transform(config.test)
    
    test_dataset = get_dataset(config.test,test_transform)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # loads model from checkpoint if given starting file
    if config.starting_checkpoint != None:
        pipe.load_model_weight(config.starting_checkpoint)
    else:
        raise Exception("Testing but did not provide weights")

    fig,ax = plt.subplots(4,2,figsize=(8,8))
   
    for step,data in enumerate(tqdm(test_loader)):
        if step>=8:
            break
        
        out_sample,out_ppm = pipe.predict(data)
        ax[step//2,step%2].plot(out_ppm,out_sample[0])
        ax[step//2,step%2].plot(data.ppm[0],data.target[0])
        ax[step//2,step%2].set_xlim(2.5,4)
        ax[step//2,step%2].invert_xaxis()
        ax[step//2,step%2].set_ylim(-0.01,0.05)        

        

    plt.show()

if __name__=="__main__":
    test_model(sys.argv[1],sys.argv[2])            
