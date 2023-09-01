

import torch
import numpy as np
import matplotlib.pyplot as plt
from mrs_acc.data.utils import max_naa_off_normalization



class ChallengeDiffPipeline:


    def __init__(self,model,device="cpu"):
        self.device=device

        self.model = model.to(self.device)
        
    def _pre_model_process_x(self,x_input,ppm):
        spec = torch.fft.fftshift(torch.fft.ifft(x_input,axis=1),axis=1)
        spec_norm,max_values = max_naa_off_normalization(spec,ppm)
        diff_spec_norm = spec_norm[:,:,1,:]-spec_norm[:,:,0,:]
        diff_spec_norm = diff_spec_norm.unsqueeze(3)
        diff_spec_norm = torch.cat([torch.real(diff_spec_norm),torch.imag(diff_spec_norm)],axis=3).float()
        
        diff_spec_norm = diff_spec_norm.permute(0,3,1,2)

        return diff_spec_norm,max_values
    
    def _post_model_process(self,model_output,max_value):
        output = model_output.detach().cpu()*max_value.unsqueeze(1)
        #return model_output.detach()
        return output

    def _post_model_process_reverse(self,data,max_value):
        post_model_y = data.target/max_value.unsqueeze(1)
        return post_model_y

    def train_step(self,data,loss_fn,optim):
        x_model,max_value = self._pre_model_process_x(data.transient_fids,data.ppm[0])
        x_model = x_model.to(self.device)

        y_model = self._post_model_process_reverse(data,max_value)
        y_model = y_model.to(self.device)

        pred = self.model(x_model,data.ppm[0])

        loss = loss_fn(pred,y_model,data.ppm[0])

        optim.zero_grad()
        loss.backward()
        optim.step()

        return loss.item()

    def val_step(self,data,loss_fn):

        with torch.no_grad():
            x_model,max_value = self._pre_model_process_x(data.transient_fids,data.ppm[0])
            x_model = x_model.to(self.device)

            y_model = self._post_model_process_reverse(data,max_value)
            y_model = y_model.to(self.device)

            pred = self.model(x_model,data.ppm[0])

            loss_result = loss_fn(pred,y_model,data.ppm[0]).item()

        return loss_result
    
    def test_step(self,data,metric_fn_dict):

        with torch.no_grad():
            x_model,max_value = self._pre_model_process_x(data.transient_fids,data.ppm[0])
            x_model = x_model.to(self.device)
            y_model = self._post_model_process_reverse(data,max_value)
            y_model = y_model.to(self.device)

            pred = self.model(x_model,data.ppm[0])

            metric_dict = {key:metric_fn_dict[key](pred,y_model,data.ppm[0]).item() for key in metric_fn_dict}

        return metric_dict
    
    def predict(self,data):
        with torch.no_grad():
            x_model,max_value = self._pre_model_process_x(data.transient_fids,data.ppm[0])
            x_model = x_model.to(self.device)
            y = self.model(x_model,data.ppm[0])

        return self._post_model_process(y,max_value),data.ppm[0]
    

    def save_model_weight(self,filename):
        torch.save(self.model.state_dict(),filename)

    def load_model_weight(self,filename):
        self.model.load_state_dict(torch.load(filename))
        


