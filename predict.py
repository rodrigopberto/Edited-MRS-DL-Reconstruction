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


def predict(config_filename: str, weight_filename=None,in_save_folder = None,
         overwrite_input_filename=None,overwrite_output_filename=None):
    config_dict = OmegaConf.to_object(OmegaConf.load(config_filename))
    if weight_filename!=None:
        config_dict["starting_checkpoint"]=weight_filename
    if overwrite_input_filename!=None and overwrite_output_filename==None:
        raise Exception("overwriting input but not output")
    elif overwrite_input_filename!=None and overwrite_output_filename!=None:
        config_dict["inference"]["data_file"]=overwrite_input_filename

    config = Config(**config_dict)



    if in_save_folder==None:
        save_folder="outputs"
    else:
        save_folder=in_save_folder
        #raise Exception("Please provide save folder")

    print(config)
    print("-------")  

    pipe = get_pipeline(config)

    predict_transform = get_transform(config.inference)
    
    predict_dataset = get_dataset(config.inference,predict_transform)

    predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False)
    
    # loads model from checkpoint if given starting file
    if config.starting_checkpoint != None:
        pipe.load_model_weight(config.starting_checkpoint)
    else:
        raise Exception("Predicting but did not provide weights")

    full_name = str(config.starting_checkpoint).split("/")[-1].split(".")[0]

    outputs,ppms = [],[]

    with h5py.File(config.inference.data_file) as hf:
        filename_list = hf["filenames"][()]


    for data in tqdm(predict_loader):
        output,ppm = pipe.predict(data)
        outputs.append(output)
        #ppms.append(ppm)

    output_filename = f"{save_folder}/{full_name}.h5"
    if overwrite_output_filename!=None:
        output_filename = overwrite_output_filename

    with h5py.File(output_filename,"w") as hf:
        hf.create_dataset("reconstruction",data=torch.cat(outputs,axis=0))
        hf.create_dataset("ppm",data=ppm)
        hf.create_dataset("filenames",data=filename_list)


if __name__=="__main__":
    weight_filename=None
    json_folder = None
    ow_input_file=None
    ow_output_file=None
    if len(sys.argv)>2:
        weight_filename=sys.argv[2]
    if len(sys.argv)>3:
        json_folder=sys.argv[3]
    if len(sys.argv)>4:
        ow_input_file=sys.argv[4]
        ow_output_file=sys.argv[5]
    predict(sys.argv[1],weight_filename,json_folder,ow_input_file,ow_output_file)