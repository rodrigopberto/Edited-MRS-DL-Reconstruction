import torch
from mrs_acc.data.utils import max_min_gaba_normalization,max_min_naa_normalization,max_median_gaba_normalization
import numpy as np

class MSEMetric:

    def __init__(self,min_ppm=2.5,max_ppm=4):

        self.min_ppm = min_ppm
        self.max_ppm = max_ppm

    def __call__(self,x,ppm,y):

        min_ind = torch.amin(torch.argwhere(ppm<=4))
        max_ind = torch.amax(torch.argwhere(ppm>=2.5))

        x_norm,_,_ = max_min_gaba_normalization(x,ppm)
        y_norm,_,_ = max_min_gaba_normalization(y,ppm)

        x_crop_norm = x_norm[:,min_ind:max_ind]
        y_crop_norm = y_norm[:,min_ind:max_ind]

        return torch.square(y_crop_norm-x_crop_norm).mean(axis=1).mean(axis=0)

class GABAMSEMetric:

    def __init__(self,min_ppm=2.8,max_ppm=3.2):

        self.min_ppm = min_ppm
        self.max_ppm = max_ppm

    def __call__(self,x,ppm,y):

        min_ind = torch.amin(torch.argwhere(ppm<=self.max_ppm))
        max_ind = torch.amax(torch.argwhere(ppm>=self.min_ppm))

        #x_norm,_,_ = max_min_gaba_normalization(x,ppm)
        #y_norm,_,_ = max_min_gaba_normalization(y,ppm)

        x_crop_norm = x[:,min_ind:max_ind]
        y_crop_norm = y[:,min_ind:max_ind]

        return torch.square(y_crop_norm-x_crop_norm).mean(axis=1).mean(axis=0)


class MSENoNormMetric:

    def __init__(self,min_ppm=2.5,max_ppm=5):

        self.min_ppm = min_ppm
        self.max_ppm = max_ppm

    def __call__(self,x,ppm,y):

        min_ind = torch.amin(torch.argwhere(ppm<=4))
        max_ind = torch.amax(torch.argwhere(ppm>=2.5))

        x_norm,_,_ = max_min_gaba_normalization(x,ppm)
        y_norm,_,_ = max_min_gaba_normalization(y,ppm)
        
        #print(abs(x_norm-x).sum())
        #raise Exception("s")

        x_crop_norm = x[:,min_ind:max_ind]
        y_crop_norm = y[:,min_ind:max_ind]

        return torch.square(y_crop_norm-x_crop_norm).mean(axis=1).mean(axis=0)


class GABASNRMetric:

    def __init__(self,gaba_min_ppm=2.8,gaba_max_ppm=3.2,noise_min_ppm=9.8,noise_max_ppm=10.8):

        self.gaba_min_ppm = gaba_min_ppm
        self.gaba_max_ppm = gaba_max_ppm
        self.noise_min_ppm = noise_min_ppm
        self.noise_max_ppm = noise_max_ppm

    def __call__(self,x,ppm,y):
        
        x_norm,_,_ = max_median_gaba_normalization(x,ppm)

        gaba_max_ind, gaba_min_ind = torch.amax(torch.argwhere(ppm>=self.gaba_min_ppm)),torch.amin(torch.argwhere(ppm<=self.gaba_max_ppm))
        dt_max_ind, dt_min_ind = torch.amax(torch.argwhere(ppm>=self.noise_min_ppm)),torch.amin(torch.argwhere(ppm<=self.noise_max_ppm))

        max_peak = x_norm[:,gaba_min_ind:gaba_max_ind].max(axis=1).values
        std = x_norm[:,dt_min_ind:dt_max_ind].std(axis=1)

        return (max_peak/(2*std)).mean(axis=0)


class GABALinewidthMetric:
    def __init__(self,gaba_min_ppm=2.8,gaba_max_ppm=3.2):

        self.gaba_min_ppm = gaba_min_ppm
        self.gaba_max_ppm = gaba_max_ppm
    
    def __call__(self,x,ppm,y):
        min_ind = torch.amin(torch.argwhere(ppm<=self.gaba_max_ppm))
        max_ind = torch.amax(torch.argwhere(ppm>=self.gaba_min_ppm))


        linewidths=[]
        for i in range(x.shape[0]):
            spec = x[i,min_ind:max_ind]
            spec = (spec-spec.min())/(spec.max()-spec.min())

            max_peak = spec.max()

            ind_max_peak = torch.argmax(spec)
            left_side = spec[:ind_max_peak]
            if torch.where(left_side>max_peak/2)[0].shape[0]==0:
                left_ind = ind_max_peak
            else:
                left_ind = torch.amin(torch.where(left_side>max_peak/2)[0])+min_ind
            right_side = spec[ind_max_peak:]
            if torch.where(right_side>max_peak/2)[0].shape[0]==0:
                right_ind = ind_max_peak
            else:
                right_ind = torch.amax(torch.where(right_side>max_peak/2)[0])+min_ind+ind_max_peak
            left_ppm = ppm[left_ind]
            right_ppm = ppm[right_ind]



            linewidths.append(left_ppm-right_ppm)
        
   
        return torch.Tensor(linewidths).mean()

class ShapeScore:
    def __init__(self,gaba_min_ppm=2.8,gaba_max_ppm=3.2,glx_min_ppm=3.6,glx_max_ppm=3.9,gaba_weight=0.6,glx_weight=0.4):

        
        self.gaba_min_ppm = gaba_min_ppm
        self.gaba_max_ppm = gaba_max_ppm
        self.glx_min_ppm = glx_min_ppm
        self.glx_max_ppm = glx_max_ppm
        self.gaba_weight=gaba_weight
        self.glx_weight = glx_weight
    
    def __call__(self,x,ppm,y):
        
        gaba_min_ind = torch.amin(torch.argwhere(ppm<=self.gaba_max_ppm))
        gaba_max_ind = torch.amax(torch.argwhere(ppm>=self.gaba_min_ppm))

        glx_min_ind = torch.amin(torch.argwhere(ppm<=self.glx_max_ppm))
        glx_max_ind = torch.amax(torch.argwhere(ppm>=self.glx_min_ppm))

        shape_scores = []
        for i in range(x.shape[0]):
            gaba_spec_x = x[i,gaba_min_ind:gaba_max_ind].cpu().numpy()
            gaba_spec_x = (gaba_spec_x-gaba_spec_x.min())/(gaba_spec_x.max()-gaba_spec_x.min())

            gaba_spec_y = y[i,gaba_min_ind:gaba_max_ind].cpu().numpy()
            gaba_spec_y = (gaba_spec_y-gaba_spec_y.min())/(gaba_spec_y.max()-gaba_spec_y.min())

            gaba_score = np.corrcoef(gaba_spec_x,gaba_spec_y)[0,1]

            glx_spec_x = x[i,glx_min_ind:glx_max_ind].cpu().numpy()
            glx_spec_x = (glx_spec_x-glx_spec_x.min())/(glx_spec_x.max()-glx_spec_x.min())

            glx_spec_y = y[i,glx_min_ind:glx_max_ind].cpu().numpy()
            glx_spec_y = (glx_spec_y-glx_spec_y.min())/(glx_spec_y.max()-glx_spec_y.min())

            glx_score = np.corrcoef(glx_spec_x,glx_spec_y)[0,1]

            shape_scores.append(gaba_score*self.gaba_weight + glx_score*self.glx_weight)
        
        return sum(shape_scores)/len(shape_scores)

class CrShapeScore:
    def __init__(self,cr_min_ppm=2.8,cr_max_ppm=3.2):

        
        self.cr_min_ppm = cr_min_ppm
        self.cr_max_ppm = cr_max_ppm

    
    def __call__(self,x,ppm,y):
        
        cr_min_ind = torch.amin(torch.argwhere(ppm<=self.cr_max_ppm))
        cr_max_ind = torch.amax(torch.argwhere(ppm>=self.cr_min_ppm))

        shape_scores = []
        for i in range(x.shape[0]):
            cr_spec_x = x[i,cr_min_ind:cr_max_ind].cpu().numpy()
            cr_spec_x = (cr_spec_x-cr_spec_x.min())/(cr_spec_x.max()-cr_spec_x.min())

            cr_spec_y = y[i,cr_min_ind:cr_max_ind].cpu().numpy()
            cr_spec_y = (cr_spec_y-cr_spec_y.min())/(cr_spec_y.max()-cr_spec_y.min())

            cr_score = np.corrcoef(cr_spec_x,cr_spec_y)[0,1]

            shape_scores.append(cr_score)
        
        return sum(shape_scores)/len(shape_scores)

class SNRMetric:

    def __init__(self,region_min_ppm=2.8,region_max_ppm=3.2,noise_min_ppm=8,noise_max_ppm=10,diff_spec=False):

        self.region_min_ppm = region_min_ppm
        self.region_max_ppm = region_max_ppm
        self.noise_min_ppm = noise_min_ppm
        self.noise_max_ppm = noise_max_ppm
        self.diff_spec=diff_spec

    def __call__(self,x,ppm,y):
        
        if self.diff_spec:
            x_norm,_,_ = max_min_gaba_normalization(x,ppm)
        else:
            x_norm,_,_ = max_min_naa_normalization(x,ppm)

        region_max_ind, region_min_ind = torch.amax(torch.argwhere(ppm>=self.region_min_ppm)),torch.amin(torch.argwhere(ppm<=self.region_max_ppm))
        dt_max_ind, dt_min_ind = torch.amax(torch.argwhere(ppm>=self.noise_min_ppm)),torch.amin(torch.argwhere(ppm<=self.noise_max_ppm))

        max_peak = x_norm[:,region_min_ind:region_max_ind].max(axis=1).values
        std = x_norm[:,dt_min_ind:dt_max_ind].std(axis=1)

        return (max_peak/(2*std)).mean(axis=0)


class LinewidthMetric:
    def __init__(self,region_min_ppm=2.8,region_max_ppm=3.2):

        self.region_min_ppm = region_min_ppm
        self.region_max_ppm = region_max_ppm
    
    def __call__(self,x,ppm,y):
        min_ind = torch.amin(torch.argwhere(ppm<=self.region_max_ppm))
        max_ind = torch.amax(torch.argwhere(ppm>=self.region_min_ppm))

        linewidths=[]
        for i in range(x.shape[0]):
            spec = x[i,min_ind:max_ind]
            spec = (spec-spec.min())/(spec.max()-spec.min())

            max_peak = spec.max()

            ind_max_peak = torch.argmax(spec)
            left_side = spec[:ind_max_peak]
            if torch.where(left_side>max_peak/2)[0].shape[0]==0:
                left_ind = ind_max_peak
            else:
                left_ind = torch.amin(torch.where(left_side>max_peak/2)[0])+min_ind
            right_side = spec[ind_max_peak:]
            if torch.where(right_side>max_peak/2)[0].shape[0]==0:
                right_ind = ind_max_peak
            else:
                right_ind = torch.amax(torch.where(right_side>max_peak/2)[0])+min_ind+ind_max_peak
            left_ppm = ppm[left_ind]
            right_ppm = ppm[right_ind]

            linewidths.append(left_ppm-right_ppm)
        
        return torch.Tensor(linewidths).mean()
