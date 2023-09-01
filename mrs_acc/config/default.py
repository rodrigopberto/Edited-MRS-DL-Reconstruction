from pydantic import BaseModel, root_validator, validator
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from omegaconf import DictConfig,ListConfig,OmegaConf


class Pipeline(BaseModel):
    pipeline_name: str
    model_name: str
    model_config: dict = {}

    @classmethod
    def __get_validators__(self):
        yield self.validate

    @classmethod
    def validate(self,x):
        if type(x)==dict:
            return self(**x)
        elif type(x)==DictConfig:
            return self(**dict(x))
        else:
            raise Exception("expected dict")

class Train(BaseModel):
    dataset_type: Optional[str] = None
    transform: Optional[str] = None
    transform_config: dict={}
    data_file: str = None
    data_file_on: str = None
    data_file_off: str = None

    weight_output_folder: str= None
    json_output_folder: str = None

    loss_fn: str="range_mse"
    learning_rate: float = 0.001

    epochs: int = 10
    batch_size: int = 10
    force_stop: Optional[int] = None

    scheduler_freq: Optional[int] = None
    early_stop: Optional[int] = None
    keep_best_treshold: Optional[int] = 0

    

    @classmethod
    def __get_validators__(self):
        yield self.validate

    @classmethod
    def validate(self,x):
        if type(x)==dict:
            return self(**x)
        elif type(x)==DictConfig:
            return self(**dict(x))
        else:
            return self()

class Validation(BaseModel):
    dataset_type: Optional[str] = None
    transform: Optional[str] = None
    transform_config: dict={}
    data_file: str = None
    data_file_on: str = None
    data_file_off: str = None

    loss_fn: str = "range_mse"
    


    @classmethod
    def __get_validators__(self):
        yield self.validate

    @classmethod
    def validate(self,x):
        if type(x)==dict:
            return self(**x)
        elif type(x)==DictConfig:
            return self(**dict(x))
        else:
            return self()

class Test(BaseModel):
    dataset_type: Optional[str] = None
    transform: Optional[str] = None
    transform_config: dict={}
    data_file: str = None
    data_file_on: str = None
    data_file_off: str = None
    
    metrics: List[str] = []
    json_output_folder: str = None

    @classmethod
    def __get_validators__(self):
        yield self.validate

    @classmethod
    def validate(self,x):
        if type(x)==dict:
            return self(**x)
        elif type(x)==DictConfig:
            return self(**dict(x))
        elif type(x)==ListConfig:
            return self(OmegaConf.to_obj(x))
        else:
            return self()
    

class Inference(BaseModel):
    dataset_type: Optional[str] = None
    transform: Optional[str] = None
    transform_config: dict={}
    data_file: str = None
    data_file_on: str = None
    data_file_off: str = None


    @classmethod
    def __get_validators__(self):
        yield self.validate

    @classmethod
    def validate(self,x):
        if type(x)==dict:
            return self(**x)
        elif type(x)==DictConfig:
            return self(**dict(x))
        elif type(x)==ListConfig:
            return self(OmegaConf.to_obj(x))
        else:
            return self()


class Config(BaseModel):
    starting_checkpoint: Optional[str]=None
    name: str
    device: str

    pipeline: Pipeline
    train: Train = Train()
    validation: Validation = Validation()
    test: Test = Test()
    inference: Inference = Inference()