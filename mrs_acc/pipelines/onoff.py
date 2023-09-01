

import torch
import numpy as np
import matplotlib.pyplot as plt
from mrs_acc.data.utils import max_cr_normalization



class OnOffPipeline:


    def __init__(self,model,device="cpu"):
        self.device=device

        self.model = model.to(self.device)
        
    def _pre_model_process_x(self,x_input,ppm):
        spec_norm,max_values = max_cr_normalization(x_input,ppm)
        spec_norm = torch.cat([torch.real(spec_norm),torch.imag(spec_norm)],axis=2)
        spec_norm = spec_norm.permute(0,2,1,3)

        #print(spec_norm.shape)

        return spec_norm,max_values
    
    def _post_model_process(self,model_output,max_value):
        output = model_output.detach().cpu()*max_value.unsqueeze(1)
        #return model_output.detach()
        return output

    def _post_model_process_reverse(self,target,max_value):
        #post_model_y = (data.target-min_value.unsqueeze(1))/(max_value.unsqueeze(1)-min_value.unsqueeze(1))
        post_model_y = (target/max_value.unsqueeze(1))
        return post_model_y

    def train_step(self,data,loss_fn,optim):
        x_model,max_value = self._pre_model_process_x(data.transient_specs,data.ppm[0])
        x_model = x_model.to(self.device)

        y_model = self._post_model_process_reverse(data.target,max_value)
        y_model = y_model.to(self.device)

        pred = self.model(x_model,data.ppm[0])

        loss = loss_fn(pred,data.ppm[0],y_model)

        """y_model = self._post_model_process_reverse(data,max_value)
        fig,ax = plt.subplots()
        ax.plot(data.ppm[0],x_model[0,0,:].cpu().mean(axis=1))
        ax.plot(data.ppm[0],y_model[0].cpu())
        ax.plot(data.ppm[0],pred[0].detach().cpu())
        ax.set_xlim(1,5)
        plt.show()
        raise Exception("img show - train")"""
        
        optim.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        #print(self.model.pre_conv[0].weight.grad)
        optim.step()

        #print(loss.item())

        return loss.item()

    def val_step(self,data,loss_fn):

        with torch.no_grad():
            x_model,max_value = self._pre_model_process_x(data.transient_specs,data.ppm[0])
            x_model = x_model.to(self.device)

            y_model = self._post_model_process_reverse(data.target,max_value)
            y_model = y_model.to(self.device)

            pred = self.model(x_model,data.ppm[0])

            loss_result = loss_fn(pred,data.ppm[0],y_model).item()

        return loss_result
    
    def test_step(self,data,metric_fn_dict):

        with torch.no_grad():
            x_model,max_value = self._pre_model_process_x(data.transient_specs,data.ppm[0])
            x_model = x_model.to(self.device)
            y_model = self._post_model_process_reverse(data.target,max_value)
            y_model = y_model.to(self.device)

            pred = self.model(x_model,data.ppm[0])

            metric_dict = {key:metric_fn_dict[key](pred,data.ppm[0],y_model).item() for key in metric_fn_dict}

        return metric_dict
    
    def predict(self,data):
        with torch.no_grad():
            x_model,max_value = self._pre_model_process_x(data.transient_specs,data.ppm[0])

            

            x_model = x_model.to(self.device)
            y = self.model(x_model,data.ppm[0])

            """y_model = self._post_model_process_reverse(data.target,max_value)
            fig,ax = plt.subplots(2,1,figsize=(8,8))
            ax[0].plot(data.ppm[0],x_model[0,0,:].cpu().mean(axis=1))
            ax[0].plot(data.ppm[0],y_model[0])
            ax[0].plot(data.ppm[0],y[0].cpu())
            ax[0].set_xlim(1,5)
            ax[0].set_ylim(-0.3,2)

            y_out = self._post_model_process(y.cpu(),max_value)[0]
            y_in = data.target[0]
            ax[1].plot(data.ppm[0],x_model[0,0,:].cpu().mean(axis=1)*max_value[0])
            ax[1].plot(data.ppm[0],y_in)
            ax[1].plot(data.ppm[0],y_out)
            ax[1].set_xlim(1,5)
            ax[1].set_ylim(-0.2,0.2)
            plt.show()
            raise Exception("img show")"""

        return self._post_model_process(y,max_value),data.ppm[0]
    

    def save_model_weight(self,filename):
        torch.save(self.model.state_dict(),filename)

    def load_model_weight(self,filename):
        self.model.load_state_dict(torch.load(filename))
        

