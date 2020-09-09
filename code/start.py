import numpy as np
import os
import torch
import yaml

from Plotting import Plot
from Run import Run
from Visualisation import Vis
import Models
import LossFunction
import Optimizer
import DirManagement
from DatasetLoader import OnehotDatasetSplit



# Path for specification file.
spesification_file = './Specification.yaml'

# Opens the file and loads it as a dict.
with open(spesification_file, "r") as file:
    main_config = yaml.load(file, Loader=yaml.FullLoader)

config = main_config['model_config']

# Adds the root path to the file names
DirManagement.make_root_paths(config)

# Makes paths/folders to save figures and models.
if config['plot']['save_fig']:
    DirManagement.make_path(config, 'plot', 'plot_folder_name')
if config['utils']['save_model']:
    DirManagement.make_path(config, 'utils', 'save_model_name')

# Changes some of the parameters in the config file if a pre trained model is used.
if config['pre_trained']['use_saved_model']:
    DirManagement.pre_train_config(config)


# Model type parameters
type_config = config['model_param']
model_type = type_config['model_type']
loss_type = type_config['loss_type']
optimizer_type = type_config['optimizer_type']

# Get the training and test data
train_data, test_data = OnehotDatasetSplit(config['paths']['dataset_path'],
                                           config['hyper_param']['test_split_pr'],
                                           config['hyper_param']['batch_size'],
                                           config['hyper_param']['max_data'])

# Create a model.
model = getattr(Models, model_type)(test_data.dataset.data.shape[2:])

# Define a optimizer
optimizer = getattr(Optimizer, optimizer_type)(config, model)

# Get a loss function
loss_fn = getattr(LossFunction, loss_type)

# Class creation
Plot = Plot(config, model, optimizer)
Run = Run(config, Plot)

# Loads information from pre trained model
if config['pre_trained']['use_saved_model']:
    # Load the stored information
    stored_model_info = torch.load(config['pre_trained']['path_saved_model'] + '/model.pth')

    model.load_state_dict(stored_model_info['model_state_dict'])
    optimizer.load_state_dict(stored_model_info['optimizer_state_dict'])

if config['utils']['train_model']:
    Run.train_model(model, optimizer, loss_fn,train_data, test_data)

#Run.test_model(model, optimizer, loss_fn, train_data, test_data)


# Run the specified visualisations.
if config['utils']['run_vis']:
    num = main_config['visual']['num']
    vis = Vis(config,Plot, train_data, test_data, num)
    del main_config['visual']['num']
    for key, value in main_config['visual'].items():
        vis_config = value
        vis_func = getattr(vis, key)

        vis_func(vis_config,model)








