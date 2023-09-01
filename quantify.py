import torch
from omegaconf import OmegaConf
from mrs_acc.config import Config
from mrs_acc.utils import get_dataset,get_pipeline,get_transform
from torch.utils.data import DataLoader
import datetime as dt
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import pandas as pd
from data.size_transforms import ref_spec_to_hr_spec_interpolate
import matlab.engine
import h5py
import numpy as np
import scipy.stats as st

def extract_fit_info(mrs_fit):
    gaba = mrs_fit["GABA"]

    gaba_linewidth = float(gaba["FWHM"])
    gaba_snr = float(gaba["SNR"])
    gaba_fit_error = float(gaba["FitError"])
    gaba_conc_water = float(gaba["ConcIU"])
    gaba_conc_cr = float(gaba["ConcCr"])

    glx = mrs_fit["Glx"]

    glx_linewidth = float(glx["FWHM"])
    glx_snr = float(glx["SNR"])
    glx_fit_error = float(glx["FitError"])
    glx_conc_water = float(glx["ConcIU"])
    glx_conc_cr = float(glx["ConcCr"])

    return {
        "gaba_snr":gaba_snr,
        "gaba_linewidth":gaba_linewidth,
        "gaba_fit_error":gaba_fit_error,
        "gaba_conc_water":gaba_conc_water,
        "gaba_conc_cr":gaba_conc_cr,
        "glx_snr":glx_snr,
        "glx_linewidth":glx_linewidth,
        "glx_fit_error":glx_fit_error,
        "glx_conc_water":glx_conc_water,
        "glx_conc_cr":glx_conc_cr,
    }


def quantify_model(config_filename: str, weight_filename:str, save_folder):

    eng = matlab.engine.start_matlab()
    s = eng.genpath("C:\\Users\\rodrigo\\Documents\\MRS\\Gannet")
    eng.addpath(s, nargout=0)

    config_dict = OmegaConf.to_object(OmegaConf.load(config_filename))
    if weight_filename!=None:
        config_dict["starting_checkpoint"]=weight_filename
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

    with h5py.File(config.test.data_file) as hf:
        filename_list = hf["filenames"][()]

    pre_df = []

    for step,data in enumerate(tqdm(test_loader)):
        output,_ = pipe.predict(data)
        output = output.cpu()[0]
        filename =str(filename_list[step]).split("'")[1]

        mrs_struct = eng.load(f"data/gannet_acc_load/{filename}.mat")["mrs_struct"]

        base_spec = torch.real(torch.from_numpy(np.array(mrs_struct["spec"]["vox1"]["GABAGlx"]["diff"]).flatten()))
        base_freq = torch.from_numpy(np.array(mrs_struct["spec"]["freq"]).flatten())
        
        new_spec = ref_spec_to_hr_spec_interpolate(output,base_spec,base_freq).numpy()

        mrs_struct["spec"]["vox1"]["GABAGlx"]["diff"]=new_spec

        mrs_fit = eng.GannetFitNoGraph(mrs_struct)

        pre_row = {"filename":filename}
        pre_row.update(extract_fit_info(mrs_fit["out"]["vox1"]))
        pre_df.append(pre_row)

    df = pd.DataFrame(pre_df)

    save_filename = weight_filename.split("/")[-1]

    df.to_csv(f"{save_folder}/{save_filename}.csv",index=False)

    full_df = pd.read_csv("pre_results/full_ref_df.csv")
    acc_df = pd.read_csv("pre_results/acc_ref_df.csv")

    jdf = pd.merge(full_df,acc_df,on="filename",suffixes=("","_acc"))
    
    df = pd.merge(df,jdf,on="filename",suffixes=("_model","_full"))

    model_error = abs(df["gaba_conc_water_model"]-df["gaba_conc_water_full"])
    acc_error = abs(df["gaba_conc_water_acc"]-df["gaba_conc_water_full"])

    model_cr_error = abs(df["gaba_conc_cr_model"]-df["gaba_conc_cr_full"])
    acc_cr_error = abs(df["gaba_conc_cr_acc"]-df["gaba_conc_cr_full"])

    #glx_model_error = abs(df["glx_conc_water_model"]-df["glx_conc_water_full"])
    #glx_acc_error = abs(df["glx_conc_water_acc"]-df["glx_conc_water_full"])

    print(f"GABA Conc Errors: {model_error.mean():.3f} +- {model_error.std():.3f}  -   {acc_error.mean():.3f} +- {acc_error.std():.3f}  - {st.wilcoxon(model_error.values,acc_error.values).pvalue:.4f} ")
    print(f"GABA/Cr Conc Errors: {model_cr_error.mean():.3f} +- {model_cr_error.std():.3f}  -   {acc_cr_error.mean():.3f} +- {acc_cr_error.std():.3f}  - {st.wilcoxon(model_cr_error.values,acc_cr_error.values).pvalue:.4f} ")
    print(f"Fit Errors Model/Acc: {df['gaba_fit_error_model'].mean():.3f} +- {df['gaba_fit_error_model'].std():.3f}  -   {df['gaba_fit_error_acc'].mean():.3f} +- {df['gaba_fit_error_acc'].std():.3f}  - {st.wilcoxon(df['gaba_fit_error_model'].values,df['gaba_fit_error_acc'].values).pvalue:.4f} ")
    print(f"Fit Errors Model/Full: {df['gaba_fit_error_model'].mean():.3f} +- {df['gaba_fit_error_model'].std():.3f}  -   {df['gaba_fit_error_full'].mean():.3f} +- {df['gaba_fit_error_full'].std():.3f}  - {st.wilcoxon(df['gaba_fit_error_model'].values,df['gaba_fit_error_full'].values).pvalue:.4f} ")
    #print(f"GLX Conc Errors: {glx_model_error.mean():.3f} +- {glx_model_error.std():.3f}  -   {glx_acc_error.mean():.3f} +- {glx_acc_error.std():.3f}  - {st.wilcoxon(glx_model_error.values,glx_acc_error.values).pvalue:.4f} ")
    print(f"SNR's - Model: {df['gaba_snr_model'].mean():.1f} - Full: {df['gaba_snr_full'].mean():.1f} - Acc: {df['gaba_snr_acc'].mean():.1f} - Model-Full:{st.wilcoxon(df['gaba_snr_model'].values,df['gaba_snr_full'].values).pvalue:.4f}")



if __name__=="__main__":
    save_folder="model_quantification"
    if len(sys.argv)>3:
        save_folder=sys.argv[3]
    quantify_model(sys.argv[1],sys.argv[2],save_folder)            
