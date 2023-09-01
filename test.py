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


def test(config_filename: str, weight_filename=None, json_output_folder=None):
    config_dict = OmegaConf.to_object(OmegaConf.load(config_filename))
    if weight_filename!=None:
        config_dict["starting_checkpoint"]=weight_filename
    if json_output_folder!=None:
        config_dict["test"]["json_output_folder"]=json_output_folder
    config = Config(**config_dict)

    print(config)
    print("-------")  

    pipe = get_pipeline(config)

    test_transform = get_transform(config.test)
    
    test_dataset = get_dataset(config.test,test_transform)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # loads model from checkpoint if given starting file
    if config.starting_checkpoint != None:
        pipe.load_model_weight(config.starting_checkpoint)
    else:
        raise Exception("Testing but did not provide weights")

    full_name = str(config.starting_checkpoint).split("/")[-1].split(".")[0]

    Path(config.test.json_output_folder).mkdir(exist_ok=True)

    metric_fn_dict = get_metric_function_dict(config.test.metrics)

    test_metrics={key:[] for key in metric_fn_dict}

    for data in tqdm(test_loader):
        val_loss_dict = pipe.test_step(data,metric_fn_dict)
        for key in test_metrics:
            test_metrics[key].append(val_loss_dict[key])

    testing_json = create_testing_json(config,test_metrics)
    
    with open(f"{config.test.json_output_folder}/{full_name}.json","w") as f:
        json.dump(testing_json,f,indent=2)

    print("---- METRIC SUMMARIES ----")
    for key in test_metrics:
        print(f"{key}: {np.array(test_metrics[key]).mean():.4f} +- {np.array(test_metrics[key]).std():.4f}")




if __name__=="__main__":
    weight_filename=None
    json_folder = None
    if len(sys.argv)>2:
        weight_filename=sys.argv[2]
    if len(sys.argv)>3:
        json_folder=sys.argv[3]
    test(sys.argv[1],weight_filename,json_folder)