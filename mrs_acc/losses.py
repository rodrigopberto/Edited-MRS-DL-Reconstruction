import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import curve_fit
import math


class RangeAndShapeLoss(nn.Module):
    def __init__(self):
        super(RangeAndShapeLoss,self).__init__()

        self.shape_loss_fn = ShapeScoreLoss()
        self.range_loss_fn = RangeMAELoss()
    
    def forward(self,x,ppm,y):
        shape_loss = self.shape_loss_fn(x,ppm,y)
        range_loss = self.range_loss_fn(x,ppm,y)

        return shape_loss+range_loss*0.5

class ShapeScoreLoss(nn.Module):
    def __init__(self,gaba_min_ppm=2.8,gaba_max_ppm=3.2,glx_min_ppm=3.6,glx_max_ppm=3.9,gaba_weight=0.6,glx_weight=0.4):
        super(ShapeScoreLoss,self).__init__()
        
        self.gaba_min_ppm = gaba_min_ppm
        self.gaba_max_ppm = gaba_max_ppm
        self.glx_min_ppm = glx_min_ppm
        self.glx_max_ppm = glx_max_ppm
        self.gaba_weight=gaba_weight
        self.glx_weight = glx_weight
    
    def forward(self,x,ppm,y):
        
        gaba_min_ind = torch.amin(torch.argwhere(ppm<=self.gaba_max_ppm))
        gaba_max_ind = torch.amax(torch.argwhere(ppm>=self.gaba_min_ppm))

        glx_min_ind = torch.amin(torch.argwhere(ppm<=self.glx_max_ppm))
        glx_max_ind = torch.amax(torch.argwhere(ppm>=self.glx_min_ppm))

        shape_scores = torch.zeros(size=(x.shape[0],))
        for i in range(x.shape[0]):
            gaba_spec_x = x[i,gaba_min_ind:gaba_max_ind]
            gaba_spec_x = (gaba_spec_x-gaba_spec_x.min())/(gaba_spec_x.max()-gaba_spec_x.min())

            gaba_spec_y = y[i,gaba_min_ind:gaba_max_ind]
            gaba_spec_y = (gaba_spec_y-gaba_spec_y.min())/(gaba_spec_y.max()-gaba_spec_y.min())

            gaba_score = torch.corrcoef(torch.cat([gaba_spec_x.unsqueeze(0),gaba_spec_y.unsqueeze(0)],axis=0))[0,1]

            glx_spec_x = x[i,glx_min_ind:glx_max_ind]
            glx_spec_x = (glx_spec_x-glx_spec_x.min())/(glx_spec_x.max()-glx_spec_x.min())

            glx_spec_y = y[i,glx_min_ind:glx_max_ind]
            glx_spec_y = (glx_spec_y-glx_spec_y.min())/(glx_spec_y.max()-glx_spec_y.min())

            glx_score = torch.corrcoef(torch.cat([glx_spec_x.unsqueeze(0),glx_spec_y.unsqueeze(0)],axis=0))[0,1]

            shape_scores[i] = gaba_score*self.gaba_weight + glx_score*self.glx_weight
        
        return 1 - shape_scores.mean()

class RangeMAELossPeakArea(nn.Module):

    def __init__(self):
        super(RangeMAELossPeakArea,self).__init__()

    def forward(self,x,ppm,y):
        
        gaba_max_ind = torch.amax(torch.where(ppm >= 2.8)[0])
        gaba_min_ind = torch.amin(torch.where(ppm <= 3.2)[0])

        gaba_slim_max_ind = torch.amax(torch.where(ppm >= 2.9)[0])
        gaba_slim_min_ind = torch.amin(torch.where(ppm <= 3.1)[0])

        glx_max_ind = torch.amax(torch.where(ppm >= 3.55)[0])
        glx_min_ind = torch.amin(torch.where(ppm <= 3.95)[0])

        gaba_x = x[:,gaba_min_ind:gaba_max_ind]
        gaba_y = y[:,gaba_min_ind:gaba_max_ind]

        gaba_slim_x = x[:,gaba_slim_min_ind:gaba_slim_max_ind]
        gaba_slim_y = y[:,gaba_slim_min_ind:gaba_slim_max_ind]

        #print(gaba_x.shape)

        gaba_peak_x,_ = torch.max(gaba_x,dim=1)
        gaba_peak_y,_ = torch.max(gaba_y,dim=1)

        gaba_area_x = torch.sum(gaba_slim_x,dim=1)
        gaba_area_y = torch.sum(gaba_slim_y,dim=1)

        glx_x = x[:,glx_min_ind:glx_max_ind]
        glx_y = y[:,glx_min_ind:glx_max_ind]

        gaba_mae = torch.abs(gaba_x-gaba_y).mean()
        glx_mae = torch.abs(glx_x-glx_y).mean()
        global_mae = torch.abs(x-y).mean()

        mae_loss = (gaba_mae*6 + glx_mae*2 + global_mae)/9
        peak_loss = torch.abs(gaba_peak_x-gaba_peak_y).mean()
        area_loss = torch.abs(gaba_area_x-gaba_area_y).mean()
        
        #print(f"gaba: {gaba_mae} - glx:{glx_mae} - global: {global_mae}")

        #print(f"mae_loss: {mae_loss} - peak loss:{peak_loss} - area_loss: {area_loss}")

        #raise Exception("stop")
        loss = (mae_loss + peak_loss/4 + area_loss/30)

        return loss
        #return gaba_mae + glx_mae

class RangeMAELoss(nn.Module):

    def __init__(self):
        super(RangeMAELoss,self).__init__()

    def forward(self,x,ppm,y):
        
        gaba_max_ind = torch.amax(torch.where(ppm >= 2.8)[0])
        gaba_min_ind = torch.amin(torch.where(ppm <= 3.2)[0])

        glx_max_ind = torch.amax(torch.where(ppm >= 3.55)[0])
        glx_min_ind = torch.amin(torch.where(ppm <= 3.95)[0])

        gaba_x = x[:,gaba_min_ind:gaba_max_ind]
        gaba_y = y[:,gaba_min_ind:gaba_max_ind]

        glx_x = x[:,glx_min_ind:glx_max_ind]
        glx_y = y[:,glx_min_ind:glx_max_ind]

        gaba_mae = torch.abs(gaba_x-gaba_y).mean()
        glx_mae = torch.abs(glx_x-glx_y).mean()
        global_mae = torch.abs(x-y).mean()

        #print(f"gaba: {gaba_mae} - glx:{glx_mae} - global: {global_mae}")

        return (gaba_mae*6 + glx_mae*3 + global_mae)/10 
        #return gaba_mae + glx_mae

class RangeMAELossDown(nn.Module):

    def __init__(self):
        super(RangeMAELossDown,self).__init__()

    def forward(self,x,ppm,y):
        
        gaba_max_ind = torch.amax(torch.where(ppm >= 2.8)[0])
        gaba_min_ind = torch.amin(torch.where(ppm <= 3.2)[0])

        glx_max_ind = torch.amax(torch.where(ppm >= 3.55)[0])
        glx_min_ind = torch.amin(torch.where(ppm <= 3.95)[0])

        gaba_x = x[:,gaba_min_ind:gaba_max_ind]
        gaba_y = y[:,gaba_min_ind:gaba_max_ind]

        glx_x = x[:,glx_min_ind:glx_max_ind]
        glx_y = y[:,glx_min_ind:glx_max_ind]

        gaba_mae = torch.abs(gaba_x-gaba_y).mean()
        glx_mae = torch.abs(glx_x-glx_y).mean()
        global_mae = torch.abs(x-y).mean()

        #print(f"gaba: {gaba_mae} - glx:{glx_mae} - global: {global_mae}")

        return (gaba_mae*6 + glx_mae*3 + global_mae)/1000000000
        #return gaba_mae + glx_mae

class CrMAELoss(nn.Module):
    def __init__(self):
        super(CrMAELoss,self).__init__()

    def forward(self,x,ppm,y):
        
        cr_max_ind = torch.amax(torch.where(ppm >= 2.8)[0])
        cr_min_ind = torch.amin(torch.where(ppm <= 3.2)[0])

        rest_max_ind = torch.amax(torch.where(ppm >= 2.5)[0])
        rest_min_ind = torch.amin(torch.where(ppm <= 4)[0])

        cr_x = x[:,cr_min_ind:cr_max_ind]
        cr_y = y[:,cr_min_ind:cr_max_ind]

        rest_x = x[:,rest_min_ind:rest_max_ind]
        rest_y = y[:,rest_min_ind:rest_max_ind]


        cr_mae = torch.abs(cr_x-cr_y).mean()
        rest_mae = torch.abs(rest_x-rest_y).mean()
        global_mae = torch.abs(x-y).mean()*0

        #print(f"Cr: {cr_mae} - global: {global_mae}")

        return (cr_mae*8 + rest_mae*2 + global_mae)/10
        #return gaba_mae + glx_mae

class RangeMAELoss3(nn.Module):

    def __init__(self):
        super(RangeMAELoss3,self).__init__()

    
    # for the forward pass, a 1d ppm array must be passed and it's assumed that
    # it's valid for all sets
    def forward(self,x,ppm,y):

        # defining indexes of boundaries
        gaba_min_ind = torch.amin(torch.argwhere(ppm<=3.2))
        gaba_max_ind = torch.amax(torch.argwhere(ppm>=2.8))

        glx_min_ind = torch.amin(torch.argwhere(ppm<=3.95))
        glx_max_ind = torch.amax(torch.argwhere(ppm>=3.55))

        # selecting part of arrays pertaining to region of interest
        loss_x_gaba = x[:,gaba_min_ind:gaba_max_ind]
        loss_y_gaba = y[:,gaba_min_ind:gaba_max_ind]

        loss_x_glx = x[:,glx_min_ind:glx_max_ind]
        loss_y_glx = y[:,glx_min_ind:glx_max_ind]

        #calculate absolute loss mean value
        loss_gaba = torch.abs(loss_x_gaba-loss_y_gaba).mean(dim=1).mean(axis=0)
        loss_glx = torch.abs(loss_x_glx-loss_y_glx).mean(dim=1).mean(axis=0)
        loss = torch.abs(x-y).mean(dim=1).mean(axis=0)

        #print(f"gaba: {loss_gaba} - glx:{loss_glx} - global: {loss}")

        return (loss_gaba*5+loss_glx*2+loss)/8

class GABAMAELoss(nn.Module):
    def __init__(self):
        super(GABAMAELoss,self).__init__()

    def forward(self,x,ppm,y):
        
        gaba_max_ind = torch.amax(torch.where(ppm >= 2.8)[0])
        gaba_min_ind = torch.amin(torch.where(ppm <= 3.2)[0])

        gaba_x = x[:,gaba_min_ind:gaba_max_ind]
        gaba_y = y[:,gaba_min_ind:gaba_max_ind]

        gaba_mae = torch.abs(gaba_x-gaba_y).mean()

        return gaba_mae

def gaba_glx_model(freq,x_0,x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9,x_10,x_11):
    return x_0*np.exp(x_1*np.square(freq-x_2)) + x_3*np.exp(x_4*np.square(freq-x_5)) + x_6*np.exp(x_7*np.square(freq-x_8)) + x_9*(freq-x_2)+ x_10*np.sin(math.pi*freq/(1.31*4)) + x_11*np.cos(math.pi*freq/(1.31*4))

def gaba_glx_model_weighted(freq,x_0,x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9,x_10,x_11):
    w = np.ones(shape=(freq.shape))
    cho_min_ind,cho_max_ind = np.amin(np.argwhere(freq<=3.285)),np.amax(np.argwhere(freq>=3.16))
    w[cho_min_ind:cho_max_ind]=0.001
    return np.sqrt(w)*gaba_glx_model(freq,x_0,x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9,x_10,x_11)

def fit_gaba(x,ppm):

    #print(x.shape)
    #print(ppm.shape)

    freqbound_min_ind,freqbound_max_ind = np.amin(np.argwhere(ppm<=4.1)),np.amax(np.argwhere(ppm>=2.79))
    gaba_min_ind,gaba_max_ind = np.amin(np.argwhere(ppm<=3.2)),np.amax(np.argwhere(ppm>=2.78))
    glx_min_ind,glx_max_ind = np.amin(np.argwhere(ppm<=4.1)),np.amax(np.argwhere(ppm>=3.4))

    #print(x[gaba_min_ind:gaba_max_ind].shape)

    maxin_gaba = x[gaba_min_ind:gaba_max_ind].max()
    maxin_glx = x[glx_min_ind:glx_max_ind].max()

    grad_points = (x[freqbound_max_ind]-x[freqbound_min_ind])/abs(freqbound_max_ind-freqbound_min_ind)
    linear_init = grad_points/abs(ppm[1]-ppm[2])

    gauss_model_init = [maxin_glx,-700,3.71,maxin_glx,-700,3.79,maxin_gaba,-90,3.02,-linear_init,0,0]
    #scaling conditions
    for i in [0,3,6,9]:
        gauss_model_init[i]=gauss_model_init[i]/maxin_glx

    lb = [-4000*maxin_glx, -1000, 3.71-0.02, -4000*maxin_glx, -1000, 3.79-0.02, -4000*maxin_gaba, -200,3.02-0.05,-40*maxin_gaba,-2000*maxin_gaba,-2000*maxin_gaba]
    ub = [4000*maxin_glx, -40, 3.71+0.02, 4000*maxin_glx, -40, 3.79+0.02, 4000*maxin_gaba, -40,3.02+0.05,40*maxin_gaba,1000*maxin_gaba,1000*maxin_gaba]
    for i in [0,3,6,9]:
        lb[i] = lb[i]/maxin_glx
        ub[i] = ub[i]/maxin_glx

    w = np.ones(shape=(x[freqbound_min_ind:freqbound_max_ind].shape))
    cho_min_ind,cho_max_ind = np.amin(np.argwhere(ppm<=3.285)),np.amax(np.argwhere(ppm>=3.16))
    w[cho_min_ind:cho_max_ind]=0.001


    #print(gauss_model_init)
    gauss_model_init,_ = curve_fit(gaba_glx_model,ppm[freqbound_min_ind:freqbound_max_ind],x[freqbound_min_ind:freqbound_max_ind]/maxin_glx,gauss_model_init,bounds=(lb,ub))
    #print(gauss_model_init)
    #print("------")
    gauss_model_param,_ = curve_fit(gaba_glx_model_weighted,ppm[freqbound_min_ind:freqbound_max_ind],np.sqrt(w)*x[freqbound_min_ind:freqbound_max_ind]/maxin_glx,gauss_model_init,bounds=(lb,ub))
    #print(gauss_model_init)
    #print(gauss_model_param)

    #rescale
    for i in [0,3,6,9]:
        gauss_model_param[i] = gauss_model_param[i]*maxin_glx

    area = gauss_model_param[6] / np.sqrt(-gauss_model_param[7]) * np.sqrt(math.pi)
    height = gauss_model_param[6]

    return area,height


class GABAFitAreaLoss(nn.Module):
    def __init__(self):
        super(GABAFitAreaLoss,self).__init__()

    def forward(self,x,ppm,y):
        
        losses = torch.zeros(size=(x.shape[0],))
        for i in range(x.shape[0]):
            try:
                area_x,height_x = fit_gaba(x[i].detach().cpu().numpy(),ppm.cpu().numpy())
                area_y,height_y = fit_gaba(y[i].cpu().numpy(),ppm.cpu().numpy())
                losses[i]=torch.abs(torch.Tensor([area_x-area_y]))*1000
            except:
                losses[i] = 3
        return losses.mean()


class RangeMAEONOFFLoss(nn.Module):

    def __init__(self):
        super(RangeMAEONOFFLoss,self).__init__()

    def forward(self,x_diff,x_off,x_on,ppm,y_diff,y_off,y_on):
        
        gaba_max_ind = torch.amax(torch.where(ppm >= 2.8)[0])
        gaba_min_ind = torch.amin(torch.where(ppm <= 3.2)[0])

        glx_max_ind = torch.amax(torch.where(ppm >= 3.55)[0])
        glx_min_ind = torch.amin(torch.where(ppm <= 3.95)[0])

        #print(ppm.shape)

        gaba_x = x_diff[:,gaba_min_ind:gaba_max_ind]
        gaba_y = y_diff[:,gaba_min_ind:gaba_max_ind]

        glx_x = x_diff[:,glx_min_ind:glx_max_ind]
        glx_y = y_diff[:,glx_min_ind:glx_max_ind]

        gaba_mae = torch.abs(gaba_x-gaba_y).mean()
        glx_mae = torch.abs(glx_x-glx_y).mean()
        global_diff_mae = torch.abs(x_diff-y_on).mean()

        off_cr_loss_mae = torch.abs(x_off[:,gaba_min_ind:gaba_max_ind]-y_off[:,gaba_min_ind:gaba_max_ind]).mean()
        on_cr_loss_mae = torch.abs(x_on[:,gaba_min_ind:gaba_max_ind]-y_on[:,gaba_min_ind:gaba_max_ind]).mean()

        #print(f"gaba: {gaba_mae} - glx:{glx_mae} - global: {global_mae}")

        return (gaba_mae*6 + glx_mae*3 + global_diff_mae + off_cr_loss_mae/2 + on_cr_loss_mae/2)/10 
        #return gaba_mae + glx_mae