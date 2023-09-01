from torch.utils.data import Dataset
import h5py
import torch
import numpy as np
from typing import NamedTuple
import matplotlib.pyplot as plt



class ChallengeTrack1Dataset(Dataset):
    def __init__(self,filename=None,transform = None,transients=40):
        super(ChallengeTrack1Dataset,self).__init__()

        self.transients=transients
        if filename==None: 
            raise Exception("No filename provided!")
        with h5py.File(filename) as hf:
            gt_fids = hf["ground_truth_fids"][()]
            gt_fids_real = torch.from_numpy(np.real(gt_fids))
            gt_fids_imag = torch.from_numpy(np.imag(gt_fids))
            print(type(gt_fids_real),flush=True)
            print(gt_fids_real.shape,flush=True)
            self.gt_fids = (gt_fids_real+1j*gt_fids_imag)#.cfloat()
            self.t = torch.from_numpy(hf["t"][()]).float()
            self.ppm = torch.from_numpy(hf["ppm"][()]).float()
        

        gt_specs = torch.fft.fftshift(torch.fft.ifft(self.gt_fids,axis=1),axis=1)
        self.target = torch.real(gt_specs[:,:,1]-gt_specs[:,:,0])

        self.transform = transform
        
    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self,idx):

        i_target = self.target[idx]
        i_ppm = self.ppm[idx]
        i_gt_fids = self.gt_fids[idx].unsqueeze(-1).repeat((1,1,self.transients))
        i_t = self.t[idx]
        
        if self.transform!=None:
            i_transient_fids = self.transform(i_gt_fids,i_t,i_ppm)
        
        output = ChallengeMRSData(
            target=i_target,
            transient_fids=i_transient_fids,
            ppm = i_ppm,
            t = i_t
        )

        return output

class ChallengeTrack1TestDataset(Dataset):
    def __init__(self,filename=None,transform = None,transients=40):
        super(ChallengeTrack1TestDataset,self).__init__()

        assert transform==None

        self.transients=transients
        if filename==None: 
            raise Exception("No filename provided!")
        with h5py.File(filename) as hf:
            transients = hf["transients"][()]
            transients_real = torch.from_numpy(np.real(transients))
            transients_imag = torch.from_numpy(np.imag(transients))
            self.transient_fids = (transients_real+1j*transients_imag)#.cfloat()
            #self.transient_fids = torch.from_numpy(hf["transients"][()]).cfloat()
            self.t = torch.from_numpy(hf["t"][()]).float()
            self.ppm = torch.from_numpy(hf["ppm"][()]).float()
            self.target = torch.from_numpy(hf["target_spectra"][()]).float()
        
        self.transform = transform
        
    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self,idx):

        i_target = self.target[idx]
        i_ppm = self.ppm[idx]
        i_transient_fids = self.transient_fids[idx]
        i_t = self.t[idx]
       
        
        output = ChallengeMRSData(
            target=i_target,
            transient_fids=i_transient_fids,
            ppm = i_ppm,
            t = i_t
        )

        return output

class InVivoMRSDataset(Dataset):

    def __init__(self,filename=None,transform = None,transients=40):
        super(InVivoMRSDataset,self).__init__()



        self.transients=transients
        if filename==None: 
            raise Exception("No filename provided!")
        with h5py.File(filename) as hf:
            self.target = torch.from_numpy(hf["target_spectra"][()]).cfloat()
            self.target_on = torch.from_numpy(hf["target_spectra_on"][()]).cfloat()
            self.target_off = torch.from_numpy(hf["target_spectra_off"][()]).cfloat()
            self.transient_specs = torch.from_numpy(hf["transient_specs"][()]).cfloat()
            self.ppm = torch.from_numpy(hf["ppm"][()]).float()[0]
        
        self.transform = transform


    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self,idx):

        i_target = torch.real(self.target[idx])
        i_target_off = torch.real(self.target_off[idx])
        i_target_on = torch.real(self.target_on[idx])
        i_ppm = self.ppm
        i_transient_specs = self.transient_specs[idx]
        
        if self.transform!=None:
            i_transient_specs = self.transform(i_transient_specs,i_ppm)
        
        output = MRSData(
            target=i_target,
            target_off=i_target_off,
            target_on=i_target_on,
            transient_specs=i_transient_specs,
            ppm=i_ppm)

        return output

class InVivoCombineMRSDataset(Dataset):
    def __init__(self,filename=None,transform = None,transients=40,*,filename_on=None,filename_off=None):
        super(InVivoCombineMRSDataset,self).__init__()

        self.transients=transients
        if filename==None or filename_on==None or filename_off==None: 
            raise Exception("No filename provided!")
        with h5py.File(filename) as hf:
            self.target = torch.from_numpy(hf["target_spectra"][()]).cfloat()
            self.target_on = torch.from_numpy(hf["target_spectra_on"][()]).cfloat()
            self.target_off = torch.from_numpy(hf["target_spectra_off"][()]).cfloat()
            self.transient_specs = torch.from_numpy(hf["transient_specs"][()]).cfloat()
            self.ppm = torch.from_numpy(hf["ppm"][()]).float()[0]

        with h5py.File(filename_on) as hf:
            self.recon_on = torch.from_numpy(hf["reconstruction"][()]).float()
        
        with h5py.File(filename_off) as hf:
            self.recon_off = torch.from_numpy(hf["reconstruction"][()]).float()
        
        self.transform = transform
        
    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self,idx):

        i_target = torch.real(self.target[idx])
        i_target_off = torch.real(self.target_off[idx])
        i_target_on = torch.real(self.target_on[idx])
        i_ppm = self.ppm
        i_transient_specs = self.transient_specs[idx]
        i_recon_on = self.recon_on[idx]
        i_recon_off = self.recon_off[idx]
        
        if self.transform!=None:
            i_transient_specs = self.transform(i_transient_specs,i_ppm)
        
        output = MRSData(
            target=i_target,
            target_off=i_target_off,
            target_on=i_target_on,
            transient_specs=i_transient_specs,
            ppm=i_ppm,
            recon_off=i_recon_off,
            recon_on=i_recon_on)

        return output

class SimulatedDataset(Dataset):
    def __init__(self,filename=None,transform = None,transients=40):
        super(SimulatedDataset,self).__init__()

        self.transients=transients
        if filename==None: 
            raise Exception("No filename provided!")
        with h5py.File(filename) as hf:
            gt_fids = hf["ground_truth_fids"][()]
            gt_fids_real = torch.from_numpy(np.real(gt_fids))
            gt_fids_imag = torch.from_numpy(np.imag(gt_fids))
            self.gt_fids = (gt_fids_real+1j*gt_fids_imag).cfloat()
            self.t = torch.from_numpy(hf["t"][()]).float()
            self.ppm = torch.from_numpy(hf["ppm"][()]).float()
        

        self.gt_specs = torch.real(torch.fft.fftshift(torch.fft.ifft(self.gt_fids,axis=1),axis=1))
        self.target = self.gt_specs[:,:,1]-self.gt_specs[:,:,0]

        self.transform = transform
        
    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self,idx):

        i_target = self.target[idx]
        i_ppm = self.ppm[idx]
        i_gt_fids = self.gt_fids[idx].unsqueeze(-1).repeat((1,1,self.transients))
        i_t = self.t[idx]
        
        if self.transform!=None:
            i_transient_fids = self.transform(i_gt_fids,i_t,i_ppm)
        
        i_transient_specs = torch.fft.fftshift(torch.fft.ifft(i_transient_fids,axis=0),axis=0)
        
        resamp_array = get_resampling_array(i_ppm,ref_freq_short)

        i_target = spec_resampling(i_target,resamp_array)
        i_target_off = spec_resampling(self.gt_specs[idx][:,0],resamp_array)
        i_target_on = spec_resampling(self.gt_specs[idx][:,1],resamp_array)
        i_transient_specs = spec_resampling(i_transient_specs,resamp_array)

        output = MRSData(
            target=i_target,
            target_on=i_target_on,
            target_off=i_target_off,
            transient_specs=i_transient_specs,
            ppm = ref_freq_short,
        )

        return output


ref_sw = 2000
ref_npoints = 2048
ref_larmor_freq = 127.758139
ref_hr_npoints = 32768
ref_freq_range = ref_sw/ref_larmor_freq
ref_freq_short = 4.68 + (ref_npoints + 1 -torch.arange(1,ref_npoints+1))/ref_npoints*ref_freq_range -ref_freq_range/2

def spec_resampling_1d(in_spec,in_freq,out_freq):
    out = torch.zeros(size=(out_freq.shape[0],),dtype=in_spec.dtype)
    for i in range(out_freq.shape[0]):
        out[i] = in_spec[torch.argmin(torch.abs(in_freq-out_freq[i]))]
    return out

def get_resampling_array(in_freq,out_freq):
    rearrange_array = []
    for i in range(out_freq.shape[0]):
        rearrange_array.append(torch.argmin(torch.abs(in_freq-out_freq[i])))
    return rearrange_array

def spec_resampling(in_spec,resampling_array):
    out = torch.zeros(size=in_spec.shape,dtype=in_spec.dtype)
    for i in range(len(resampling_array)):
            out[i] = in_spec[resampling_array[i]]

    return out

class MRSData(NamedTuple):
    target: torch.Tensor
    target_on: torch.Tensor
    target_off: torch.Tensor
    transient_specs: torch.Tensor
    ppm: torch.Tensor
    sw: int = 0
    recon_on: torch.Tensor=torch.Tensor([0])
    recon_off: torch.Tensor=torch.Tensor([0])

class ChallengeMRSData(NamedTuple):
    target: torch.Tensor
    transient_fids: torch.Tensor
    ppm: torch.Tensor
    t: torch.Tensor