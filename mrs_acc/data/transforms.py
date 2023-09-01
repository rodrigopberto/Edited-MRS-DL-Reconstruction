import torch
import math
from typing import NamedTuple

class NormalNoise:

    def __init__(self,
                 amp_base_noise = 10, amp_var_noise = 5,
                 freq_base_noise = 0.5, freq_var_noise = 0.4,
                 phase_base_noise = 2, phase_var_noise = 1
                 ):
        
        self.amp_base_noise = amp_base_noise
        self.amp_var_noise = amp_var_noise
        self.freq_base_noise  = freq_base_noise
        self.freq_var_noise = freq_var_noise
        self.phase_base_noise = phase_base_noise
        self.phase_var_noise = phase_var_noise

    
    def __call__(self,in_fids,t,ppm):
        fids = in_fids.clone()
        if len(fids.shape)!=3:
            raise Exception("fids dimension is not 3")
        
        amp_noise_level = 0.05*(self.amp_base_noise+(2*torch.rand(1)-1)*self.amp_var_noise)
        noise_real = torch.randn(size = fids.shape)*amp_noise_level
        noise_imag = 1j*torch.randn(size = fids.shape)*amp_noise_level

        fids = fids+noise_real+noise_imag

        freq_noise = torch.randn(size=(1,2,fids.shape[2]))*(self.freq_base_noise+(2*torch.rand(1)-1)*self.freq_var_noise)
        phase_noise = torch.randn(size=(1,2,fids.shape[2]))*(self.phase_base_noise+(2*torch.rand(1)-1)*self.phase_var_noise)

        fids = fids*torch.exp(1j*(t.unsqueeze(1).unsqueeze(2)*freq_noise*2*math.pi + torch.ones(fids.shape)*phase_noise*math.pi/180))

        return fids
    

class Accelerate:

    def __init__(self,acceleration_rate=4):
        
        self.acceleration_rate=acceleration_rate

    
    def __call__(self,in_specs,ppm):
        in_specs = in_specs.clone()
        if len(in_specs.shape)!=3:
            raise Exception("fids dimension is not 3")
        
        in_specs = in_specs[:,:,:int(in_specs.shape[2]/self.acceleration_rate)]

        return in_specs
    
class RandomAccelerate:

    def __init__(self,acceleration_rate=4):
        
        self.acceleration_rate=acceleration_rate
    
    
    
    def __call__(self,in_specs,ppm):
        in_specs = in_specs.clone()
        if len(in_specs.shape)!=3:
            raise Exception("fids dimension is not 3")
        
        p = torch.randperm(in_specs.shape[2])
        in_specs = in_specs[:,:,p]
        
        in_specs = in_specs[:,:,:int(in_specs.shape[2]/self.acceleration_rate)]

        return in_specs