import torch

def max_min_gaba_normalization(x,ppm):
    
    min_ind = torch.amin(torch.argwhere(ppm<=3.2))
    max_ind = torch.amax(torch.argwhere(ppm>=2.8))

    norm_crop = x[:,min_ind:max_ind]
    if torch.is_complex(norm_crop):
        norm_crop = torch.real(norm_crop)

    if len(norm_crop.shape)>2:
        min_values = norm_crop.mean(axis=-1).reshape((norm_crop.shape[0],-1)).min(axis=1).values.unsqueeze(1)
        max_values = norm_crop.mean(axis=-1).reshape((norm_crop.shape[0],-1)).max(axis=1).values.unsqueeze(1)
    else:
        min_values = norm_crop.reshape((norm_crop.shape[0],-1)).min(axis=1).values.unsqueeze(1)
        max_values = norm_crop.reshape((norm_crop.shape[0],-1)).max(axis=1).values.unsqueeze(1)

    if len(x.shape)>2:
        min_values=min_values.unsqueeze(2)
        max_values=max_values.unsqueeze(2)
    if len(x.shape)>3:
        min_values=min_values.unsqueeze(3)
        max_values=max_values.unsqueeze(3)

    x_norm = (x-min_values)/(max_values-min_values)

    return x_norm,max_values.reshape(-1),min_values.reshape(-1)

def max_median_gaba_normalization(x,ppm):
    
    min_ind = torch.amin(torch.argwhere(ppm<=3.2))
    max_ind = torch.amax(torch.argwhere(ppm>=2.8))

    norm_crop = x[:,min_ind:max_ind]
    if torch.is_complex(norm_crop):
        norm_crop = torch.real(norm_crop)

    #min_values = norm_crop.reshape((norm_crop.shape[0],-1)).min(axis=1).values.view(-1,1)
    #max_values = norm_crop.reshape((norm_crop.shape[0],-1)).max(axis=1).values.view(-1,1)
    if len(norm_crop.shape)>2:
        min_values = torch.median(torch.real(x).mean(axis=-1).reshape((x.shape[0],-1)),axis=1).values.unsqueeze(1)
        max_values = norm_crop.mean(axis=-1).reshape((norm_crop.shape[0],-1)).max(axis=1).values.unsqueeze(1)
    else:
        min_values = torch.median(x.reshape((torch.real(x).shape[0],-1)),axis=1).values.unsqueeze(1)
        max_values = norm_crop.reshape((norm_crop.shape[0],-1)).max(axis=1).values.unsqueeze(1)

    if len(x.shape)>2:
        min_values=min_values.unsqueeze(2)
        max_values=max_values.unsqueeze(2)
    if len(x.shape)>3:
        min_values=min_values.unsqueeze(3)
        max_values=max_values.unsqueeze(3)

    x_norm = (x-min_values)/(max_values-min_values)

    return x_norm,max_values.reshape(-1),min_values.reshape(-1)

def max_gaba_normalization(x,ppm):

    min_ind = torch.amin(torch.argwhere(ppm<=3.2))
    max_ind = torch.amax(torch.argwhere(ppm>=2.8))

    norm_crop = x[:,min_ind:max_ind]
    if torch.is_complex(norm_crop):
        norm_crop = torch.real(norm_crop)

    #min_values = norm_crop.reshape((norm_crop.shape[0],-1)).min(axis=1).values.view(-1,1)
    #max_values = norm_crop.reshape((norm_crop.shape[0],-1)).max(axis=1).values.view(-1,1)
    if len(norm_crop.shape)>2:
        max_values = norm_crop.mean(axis=-1).reshape((norm_crop.shape[0],-1)).max(axis=1).values.unsqueeze(1)
    else:
        max_values = norm_crop.reshape((norm_crop.shape[0],-1)).max(axis=1).values.unsqueeze(1)

    if len(x.shape)>2:
        max_values=max_values.unsqueeze(2)
    if len(x.shape)>3:
        max_values=max_values.unsqueeze(3)

    x_norm = x/max_values

    return x_norm,max_values.reshape(-1)

def max_naa_off_normalization(x,ppm):
    min_ind = torch.amin(torch.argwhere(ppm<=2.15))
    max_ind = torch.amax(torch.argwhere(ppm>=1.85))

    norm_crop = x[:,min_ind:max_ind,:1]
    if torch.is_complex(norm_crop):
        norm_crop = torch.abs(norm_crop)

    if len(norm_crop.shape)>2:
        max_values = norm_crop.mean(axis=-1).reshape((norm_crop.shape[0],-1)).max(axis=1).values.unsqueeze(1)
    else:
        max_values = norm_crop.reshape((norm_crop.shape[0],-1)).max(axis=1).values.unsqueeze(1)

    if len(x.shape)>2:
        max_values=max_values.unsqueeze(2)
    if len(x.shape)>3:
        max_values=max_values.unsqueeze(3)

    x_norm = x/max_values

    return x_norm,max_values.reshape(-1)

def max_min_naa_normalization(x,ppm):
    
    min_ind = torch.amin(torch.argwhere(ppm<=4))
    max_ind = torch.amax(torch.argwhere(ppm>=1))

    norm_crop = x[:,min_ind:max_ind]
    if torch.is_complex(norm_crop):
        norm_crop =  torch.real(norm_crop)

    #min_values = norm_crop.reshape((norm_crop.shape[0],-1)).min(axis=1).values.view(-1,1)
    #max_values = norm_crop.reshape((norm_crop.shape[0],-1)).max(axis=1).values.view(-1,1)
    if len(norm_crop.shape)>2:
        min_values = norm_crop.mean(axis=-1).reshape((norm_crop.shape[0],-1)).min(axis=1).values.view(-1,1)
        max_values = norm_crop.mean(axis=-1).reshape((norm_crop.shape[0],-1)).max(axis=1).values.view(-1,1)
    else:
        min_values = norm_crop.reshape((norm_crop.shape[0],-1)).min(axis=1).values.view(-1,1)
        max_values = norm_crop.reshape((norm_crop.shape[0],-1)).max(axis=1).values.view(-1,1)

    if len(x.shape)>2:
        min_values=min_values.unsqueeze(2)
        max_values=max_values.unsqueeze(2)
    if len(x.shape)>3:
        min_values=min_values.unsqueeze(3)
        max_values=max_values.unsqueeze(3)
    
    x_norm = (x-min_values)/(max_values-min_values)

    return x_norm,max_values.reshape(-1),min_values.reshape(-1)

def max_cr_normalization(x,ppm):
    min_ind = torch.amin(torch.argwhere(ppm<=3.2))
    max_ind = torch.amax(torch.argwhere(ppm>=2.8))

    norm_crop = x[:,min_ind:max_ind,:1]
    if torch.is_complex(norm_crop):
        norm_crop = torch.abs(norm_crop)

    if len(norm_crop.shape)>2:
        max_values = norm_crop.mean(axis=-1).reshape((norm_crop.shape[0],-1)).max(axis=1).values.unsqueeze(1)
    else:
        max_values = norm_crop.reshape((norm_crop.shape[0],-1)).max(axis=1).values.unsqueeze(1)





    if len(x.shape)>2:
        max_values=max_values.unsqueeze(2)
    if len(x.shape)>3:
        max_values=max_values.unsqueeze(3)

    x_norm = x/max_values

    return x_norm,max_values.reshape(-1)

