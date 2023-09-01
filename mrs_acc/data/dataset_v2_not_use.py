import torch
import h5py

from typing import NamedTuple


class MRSDataSample(NamedTuple):
    transient_specs: torch.Tensor
    ppm: torch.Tensor
    target: torch.Tensor
    target_off: torch.Tensor
    target_on: torch.Tensor

class InVivoDataset(torch.utils.data.Dataset):
    
    def __init__(self,filename=None,transform=None):
        super(InVivoDataset,self).__init__()

        with h5py.File(filename) as hf:
            self.transient_specs = torch.from_numpy(hf["transient_specs"][()]).to(torch.complex64)
            self.ppm = torch.from_numpy(hf["ppm"][()]).float()
            self.target = torch.from_numpy(hf["target_spectra"][()]).float()
            self.target_on = torch.from_numpy(hf["target_spectra_on"][()]).float()
            self.target_off = torch.from_numpy(hf["target_spectra_off"][()]).float()

        self.transform = transform

    def __len__(self):
        return self.transient_specs.shape[0]

    def __getitem__(self,idx):

        i_transient_specs = self.transient_specs[idx]
        i_ppm = self.ppm[idx]
        i_target = self.target[idx]
        i_target_on = self.target_on[idx]
        i_target_off = self.target_off[idx]

        if self.transform !=None:
            i_transient_specs = self.transform(i_transient_specs,i_ppm)
        
        return MRSDataSample(
            transient_specs=i_transient_specs,
            ppm = i_ppm,
            target=i_target,
            target_off=i_target_off,
            target_on=i_target_on
        )