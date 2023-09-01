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
import pandas as pd


def test_pred(input_filename,ref_filename,out_filename):


    loss_function_names = ["mse","gaba_mse","gaba_snr","gaba_linewidth","shape_score"]
    metric_fn_dict = get_metric_function_dict(loss_function_names)

    with h5py.File(input_filename) as hf:
        recons = torch.from_numpy(hf["reconstruction"][()])
        ppm = torch.from_numpy(hf["ppm"][()])
        filenames =hf["filenames"][()]
    
    with h5py.File(ref_filename) as hf:
        target = torch.from_numpy(hf["target_spectra"][()])

    metrics_dict = {"filename":[]}
    metrics_dict.update({key:[] for key in loss_function_names})



    for i in tqdm(range(recons.shape[0])):
        i_recon = recons[i:i+1]
        i_target = target[i:i+1]
        i_filename = str(filenames[i]).split("'")[1]

        metrics_dict["filename"].append(i_filename)
        for metric in metric_fn_dict:
            metrics_dict[metric].append(metric_fn_dict[metric](i_recon,ppm,i_target).item())
    
    df = pd.DataFrame(metrics_dict)
    print(df)
    df.to_csv(out_filename,index=False)

    print("---- METRIC SUMMARIES ----")
    for key in loss_function_names:
        print(f"{key}: {np.array(metrics_dict[key]).mean()}")




if __name__=="__main__":
    input_filename = sys.argv[1]
    ref_filename = sys.argv[2]
    out_filename = sys.argv[3]
    test_pred(input_filename,ref_filename,out_filename)