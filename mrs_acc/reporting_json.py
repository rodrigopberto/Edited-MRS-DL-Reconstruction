from mrs_acc.utils import config_to_dict



def create_training_json(config,date_str,fullname,train_losses,val_losses):
    
    output_json = {
        "name":config.name,
        "training_date_str":date_str,
        "full_name":fullname,
        "config":config_to_dict(config),

        "train_losses":train_losses,
        "val_losses": val_losses

    }

    return output_json


def create_testing_json(config,metrics):


    output_json = {
        "name":config.name,
        "config":config_to_dict(config),

        "metrics":metrics,

    }

    return output_json