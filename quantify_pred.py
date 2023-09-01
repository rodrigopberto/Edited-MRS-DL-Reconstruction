import torch
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import pandas as pd
from data.size_transforms import ref_spec_to_hr_spec_interpolate
import matlab.engine
import h5py
import numpy as np
import scipy.stats as st
import os

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


def quantify_pred(input_filename,output_filename):   

    print("Right Here!!!!")
    with h5py.File(input_filename) as hf:
        recons = hf["reconstruction"][()]
        filenames = hf["filenames"][()]
        input_ppm = hf["ppm"][:2048]

    eng = matlab.engine.start_matlab()
    s = eng.genpath("C:\\Users\\rodrigo\\Documents\\MRS\\Gannet")
    eng.addpath(s, nargout=0)

    pre_df = []

    for i in tqdm(range(recons.shape[0])):
        filename =str(filenames[i]).split("'")[1]

        mrs_struct = eng.load(f"data/gannet_acc_load/{filename}.mat")["mrs_struct"]

        base_spec = torch.real(torch.from_numpy(np.array(mrs_struct["spec"]["vox1"]["GABAGlx"]["diff"]).flatten()))
        base_freq = torch.from_numpy(np.array(mrs_struct["spec"]["freq"]).flatten())
        
        new_spec = ref_spec_to_hr_spec_interpolate(recons[i],base_spec,base_freq).numpy()

        mrs_struct["spec"]["vox1"]["GABAGlx"]["diff"]=new_spec

        mrs_fit = eng.GannetFitNoGraph(mrs_struct)

        pre_row = {"filename":filename}
        try:
            pre_row.update(extract_fit_info(mrs_fit["out"]["vox1"]))
        except:
            continue
        pre_df.append(pre_row)

    df = pd.DataFrame(pre_df)

    df.to_csv(f"{output_filename}.csv",index=False)

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
    quantify_pred(sys.argv[1],sys.argv[2])            
