# Deep Learning GABA-edited MRS Reconstruction Pipelines

This is a library for Deep learning based GABA-edited MRS Reconstruction, which was developed as part of my master's thesis. The library is structured to work with configuration files and is easily extensable to support more preprocessing of the model inputs, different formats, models and etc.

It also contains functions to evaluate reconstructions and to quantify the concentrations using Gannet (Matlab-based MRS processing tool) with the python matlab engine.

#### Library structure

The library components are in the `mrs_acc` folder and its subfolders and the `utils.py` file acts a map between what is specified in the config files and the components. The current organization of the config file is as follows:

 * name: naming using for saving resulting weights and logs
 * device: cpu or cuda
 * starting_checkpoint: starting weights for training (also used for testing/predicting if other weights not specified)
 * **Pipeline**:
    * *pipeline_name*: type of pipeline (diff, on_off, ...)
    * *model_name*: type of model (TODO)
    * *model_config*: dictionary of model parameters, such as number of filters, steps, etc.
 * **Train**:
    * *dataset_type*: Type of data, defines dataset class used
    * *data_file*: path of train data (only supports single file as of now)
    * *transform*: transformation applied to dataset
    * *transform_config*: optional configuration for transform
    * *weight_output_folder*: folder to output weights
    * *json_output_folder*: folder to output reporting jsons
    * *loss_fn*: loss function name
    * *learning_rate*: initial learning rate used by optimizer
    * *epochs*: number of epochs to run training
    * *batch_size*: size of batches
    * *scheduler_freq*: how many epochs to run before halving the learning rate
    * *early_stop*: how many epochs to run before stopping training if no progress is validation error is made
    * *keep_best_threshold*: how many epochs to run before resetting the weights to the previous best validation error
    * *force_stop*: for testing purposes, stops function at given epoch
 * **Validation**
    * *dataset_type*
    * *data_file*
    * transform
    


