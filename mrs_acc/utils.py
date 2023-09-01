from mrs_acc.config import Config
from mrs_acc.models import (Unet2D,UNET2DOLD,UNET2DMLP,DownUnet2DMLP,PreDownUnet2D,DownUnet2DMLPV2,Unet1D,DownUnet2DMLPV3,CombineCNN,
                            CombineCNNZoom,Recurrent1DUnet,RecurrentCNN,DualDownUnet2D,PreDownUnet2DTanh,DualPreDownUnet2D,DualPreDownUnet2DONOFF,
                            DownUnet2DMLPV4)
from mrs_acc.data.datasets import InVivoMRSDataset,ChallengeTrack1TestDataset,ChallengeTrack1Dataset,InVivoCombineMRSDataset,SimulatedDataset
from mrs_acc.pipelines import DiffPipeline,ChallengeDiffPipeline,ONPipeline,OFFPipeline,CombinePipeline,OnOffPipeline,OnOffLossPipeline
from mrs_acc.losses import (RangeMAELoss,RangeMAELoss3,ShapeScoreLoss,RangeAndShapeLoss,CrMAELoss,
                            RangeMAELossPeakArea,RangeMAELossDown,GABAFitAreaLoss,RangeMAEONOFFLoss,GABAMAELoss)
from mrs_acc.data.transforms import NormalNoise,Accelerate,RandomAccelerate
from mrs_acc.metrics import MSEMetric,GABALinewidthMetric,GABASNRMetric,ShapeScore,MSENoNormMetric,CrShapeScore,GABAMSEMetric
from mrs_acc.config.default import Config,Train,Test,Pipeline,Validation,Inference


def get_pipeline(config):
    pipe_name = config.pipeline.pipeline_name
    model_name = config.pipeline.model_name
    model_config = config.pipeline.model_config

    pipe_fn = _get_pipeline(pipe_name)

    model = _get_model(model_name,model_config)

    return pipe_fn(model,config.device)


def _get_pipeline(pipeline_name):
    if pipeline_name=="diff":
        return DiffPipeline
    elif pipeline_name=="challenge_diff":
        return ChallengeDiffPipeline
    elif pipeline_name=="on":
        return ONPipeline
    elif pipeline_name=="off":
        return OFFPipeline
    elif pipeline_name=="combine":
        return CombinePipeline
    elif pipeline_name=="onoff":
        return OnOffPipeline
    elif pipeline_name=="onoffloss":
        return OnOffLossPipeline
    else:
        raise Exception(f"pipeline not implemented - {pipeline_name}")

def _get_model(model_name, model_config):
    if model_name=="unet_2d":
        return Unet2D(**model_config)
    elif model_name=="unet_2d_old":
        return UNET2DOLD(**model_config)
    elif model_name=="unet_2d_mlp":
        return UNET2DMLP(**model_config)
    elif model_name=="pre_down_unet_2d":
        return PreDownUnet2D(**model_config)
    elif model_name=="pre_down_unet_2d_tanh":
        return PreDownUnet2DTanh(**model_config)
    elif model_name=="down_unet_2d_mlp":
        return DownUnet2DMLP(**model_config)
    elif model_name=="down_unet_2d_mlp_v2":
        return DownUnet2DMLPV2(**model_config)
    elif model_name=="down_unet_2d_mlp_v3":
        return DownUnet2DMLPV3(**model_config)
    elif model_name=="down_unet_2d_mlp_v4":
        return DownUnet2DMLPV4(**model_config)
    elif model_name=="unet_1d":
        return Unet1D(**model_config)
    elif model_name=="combine_cnn":
        return CombineCNN(**model_config)
    elif model_name=="combine_cnn_zoom":
        return CombineCNNZoom(**model_config)
    elif model_name=="recu_unet_1d":
        return Recurrent1DUnet(**model_config)
    elif model_name=="recu_cnn":
        return RecurrentCNN(**model_config)
    elif model_name=="dual_down_unet_2d":
        return DualDownUnet2D(**model_config)
    elif model_name=="dual_pre_down_unet_2d":
        return DualPreDownUnet2D(**model_config)
    elif model_name=="dual_pre_down_unet_2d_onoff":
        return DualPreDownUnet2DONOFF(**model_config)
    #if model_name=="unet_2d_mlp":
    #    return UNET2DMLP(**model_config)
    #elif model_name=="avgnet":
    #    return AVGNET(**model_config)
    #elif model_name=="region_net":
    #    return Region1D2DNet(**model_config)
    #elif model_name=="region_unet_2d":
    #    return Region2DUNET(**model_config)
    #elif model_name=="combine_net":
    #    return Combine1D2DNet(**model_config)
    #elif model_name=="resunet_2d":
    #    return RESUNET2D(**model_config)
    #elif model_name=="unet_onoff":
    #    return UNETONOFF(**model_config)
    #elif model_name=="resunet_2":
    #    return ResUnet_2(**model_config)
    #elif model_name=="drunet":
    #    return DRUNET(**model_config)
    #elif model_name=="unet_1d2d":
    #    return UNET1D2D(**model_config)
    elif model_name=="control":
        return None
    else:
        raise Exception(f"Model Not Supported - {model_name}" )
    

def get_dataset(config,transform):
    if config.dataset_type=="invivo":
        return InVivoMRSDataset(config.data_file,transform=transform)
    elif config.dataset_type=="challenge_track_1":
        return ChallengeTrack1Dataset(config.data_file,transform=transform)
    elif config.dataset_type=="challenge_track_1_test":
        return ChallengeTrack1TestDataset(config.data_file,transform=transform)
    elif config.dataset_type=="invivo_combine":
        return InVivoCombineMRSDataset(config.data_file,transform=transform,filename_off=config.data_file_off,filename_on=config.data_file_on)
    elif config.dataset_type=="simulated":
        return SimulatedDataset(config.data_file,transform=transform)
    else:
        raise Exception(f"dataset not implemented! - {config.dataset_type}")

def get_transform(config):
    if config.transform==None:
        return None
    elif config.transform=="normal_noise":
        return NormalNoise(**config.transform_config)
    elif config.transform=="accelerate":
        return Accelerate(**config.transform_config)
    elif config.transform=="random_accelerate":
        return RandomAccelerate(**config.transform_config)
    else:
        raise Exception(f"transform not implemented! - {config.transform}")


def get_loss_function(loss_name,loss_config_dict={}):
    if loss_name=="range_mae":
        return RangeMAELoss(**loss_config_dict)
    elif loss_name=="shape_score":
        return ShapeScore(**loss_config_dict)
    elif loss_name=="range_and_shape":
        return RangeAndShapeLoss(**loss_config_dict)
    elif loss_name=="range_mae_cr":
        return CrMAELoss(**loss_config_dict)
    if loss_name=="mae":
        return MAELoss(**loss_config_dict)
    elif loss_name=="range_mae_2":
        return RangeMAELoss2(**loss_config_dict)
    elif loss_name=="range_mae_3":
        return RangeMAELoss3(**loss_config_dict)
    elif loss_name=="gaba_area":
        return GabaAreaLoss(**loss_config_dict)
    elif loss_name=="range_mae_peak_area":
        return RangeMAELossPeakArea(**loss_config_dict)
    elif loss_name=="range_mae_down":
        return RangeMAELossDown(**loss_config_dict)
    elif loss_name=="gaba_fit_area_loss":
        return GABAFitAreaLoss(**loss_config_dict)
    elif loss_name=="range_mae_onoff_loss":
        return RangeMAEONOFFLoss(**loss_config_dict)
    elif loss_name=="gaba_mae_loss":
        return GABAMAELoss(**loss_config_dict)
    else:
        raise Exception(f"Loss not implemented - {loss_name}")
    
def get_loss_function_dict(loss_name_list,device):
    return {loss_name:get_loss_function(loss_name).to(device) for loss_name in loss_name_list}

def get_metric_function(metric_name,metric_config_dict={}):
    if metric_name=="mse":
        return MSEMetric(**metric_config_dict)
    if metric_name=="gaba_mse":
        return GABAMSEMetric(**metric_config_dict)
    elif metric_name=="mse_no_norm":
        return MSENoNormMetric(**metric_config_dict)
    elif metric_name=="gaba_linewidth":
        return GABALinewidthMetric(**metric_config_dict)
    elif metric_name=="gaba_snr":
        return GABASNRMetric(**metric_config_dict)
    elif metric_name=="shape_score":
        return ShapeScore(**metric_config_dict)
    elif metric_name=="shape_score_cr":
        return CrShapeScore(**metric_config_dict)
    else:
        raise Exception(f"Metric not implemented - {metric_name}")
    
def get_metric_function_dict(metric_name_list):
    return {metric_name:get_metric_function(metric_name) for metric_name in metric_name_list}



def config_to_dict(config):
    types_to_convert=[Config,Train,Test,Pipeline,Validation,Inference]
    config_dict = dict(config)
    for key in config_dict:
        if type(config_dict[key]) in (types_to_convert):
            config_dict[key]=dict(config_dict[key])
    return config_dict
