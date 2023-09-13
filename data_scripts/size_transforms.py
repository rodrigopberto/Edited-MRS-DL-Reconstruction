import numpy as np
import torch
import math
import matplotlib.pyplot as plt


#### constants for resizing data

ref_sw = 2000
ref_npoints = 2048
ref_larmor_freq = 127.758139
ref_hr_npoints = 32768
ref_freq_range = ref_sw/ref_larmor_freq
ref_freq_short = 4.68 + (ref_npoints + 1 -torch.arange(1,ref_npoints+1))/ref_npoints*ref_freq_range -ref_freq_range/2
ref_freq_long = 4.68 + (ref_hr_npoints + 1 -torch.arange(1,ref_hr_npoints+1))/ref_hr_npoints*ref_freq_range -ref_freq_range/2

####


def input_fids_to_std_lr_spec(input_lr_fid,input_sw,input_armor_freq,input_lb):
    
    time_array = torch.arange(1,input_lr_fid.shape[0]+1)/input_sw
    fid = input_lr_fid*torch.exp(-time_array.reshape(-1,1)*input_lb*math.pi)

    
    input_zero_fill = int(32768*input_sw/2000)

    input_freq_range = input_sw/input_armor_freq
    freq_array = 4.68 + (input_zero_fill + 1 -torch.arange(1,input_zero_fill+1))/input_zero_fill*input_freq_range -input_freq_range/2
    hr_input_spec = torch.fft.fftshift(torch.fft.fft(fid,input_zero_fill,axis=0),axis=0)

    resampled_hr_spec = spec_resampling(hr_input_spec,freq_array,ref_freq_long)

    resampled_hr_fid = torch.fft.ifft(torch.fft.ifftshift(resampled_hr_spec,axis=0),axis=0)
    shortened_fid = resampled_hr_fid[:ref_npoints]
    
    out_spec = torch.fft.fftshift(torch.fft.fft(shortened_fid,axis=0),axis=0)


    #out_spec = hr_spec_to_ref_spec(hr_input_spec,freq_array)

    return out_spec

## keeping it separaded but somewhat copied because of size...
def hr_spec_to_ref_spec_1d(in_spec,in_freq):
    resampled_hr_spec = spec_resampling_1d(in_spec,in_freq,ref_freq_long)

    resampled_hr_fid = torch.fft.ifft(torch.fft.ifftshift(resampled_hr_spec,axis=0),axis=0)
    shortened_fid = resampled_hr_fid[:ref_npoints]
    
    out_spec = torch.fft.fftshift(torch.fft.fft(shortened_fid))

    return out_spec

def ref_spec_to_hr_spec(in_spec,base_spec,base_freq):

    ref_in_fid = torch.fft.ifft(torch.fft.ifftshift(in_spec))
    hr_in_spec = torch.fft.fftshift(torch.fft.fft(ref_in_fid,ref_hr_npoints))

    out_array = overwriting_spec_points(hr_in_spec,ref_freq_long,base_spec,base_freq)
    return out_array


def spec_resampling_1d(in_spec,in_freq,out_freq):
    out = torch.zeros(size=(out_freq.shape[0],),dtype=in_spec.dtype)
    for i in range(out_freq.shape[0]):
        out[i] = in_spec[torch.argmin(torch.abs(in_freq-out_freq[i]))]
    return out

def spec_resampling(in_spec,in_freq,out_freq):
    out = torch.zeros(size=(out_freq.shape[0],in_spec.shape[1]),dtype=in_spec.dtype)
    for i in range(out_freq.shape[0]):
        out[i,:] = in_spec[torch.argmin(torch.abs(in_freq-out_freq[i])),:]
    return out


def overwriting_spec_points(in_spec,in_freq,base_spec,base_freq):
    rewrite_array = [torch.argmin(torch.abs(base_freq-in_freq[i])) for i in range(in_freq.shape[0])]
    out_array = base_spec.clone()
    for i in range(len(rewrite_array)):
        out_array[rewrite_array[i]]=in_spec[i]
    return out_array

def ref_spec_to_hr_spec_interpolate(in_spec,base_spec,base_freq):
    rewrite_array = [torch.argmin(torch.abs(ref_freq_short[i]-base_freq)) for i in range(ref_freq_short.shape[0])]
    out_array = base_spec.clone()
    for i in range(len(rewrite_array)-1):
        init_i = rewrite_array[i]
        stop_i = rewrite_array[i+1]-1
        stretch = in_spec[i] +torch.arange(0,stop_i-init_i+1)*(in_spec[i+1]-in_spec[i])/(stop_i-init_i)
        out_array[init_i:stop_i+1]=stretch
    
    return out_array


