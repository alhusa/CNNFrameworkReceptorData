import os
import yaml
def make_path(config,cat, subfolder):
    '''
    Checks if a directory of the same name exists. Creates a directory with the name given in the specifications if none
    exists. Creates adds 1,2,3... behind the name if the directory already exists.

    :param config: Dict containing program parameters.
    :param cat: The category of data that needs a folder (plots or model for now)
    :param subfolder: Key to where the name of the subfolder is found.
    :return:
    '''

    # Finds the paths in to the subfolder.
    file_path = config[cat][subfolder]

    # Checks if path to plots exists and creates a directory if it does not.
    if not os.path.isdir(file_path): os.makedirs(file_path)

    # Adds an integer behind the name if the folder already exist.
    else:
        c = 1
        file_path = file_path + str(c)
        while(os.path.isdir(file_path)):
            file_path = file_path[:-len(str(c-1))] + str(c)
            c += 1
        os.makedirs(file_path)

    # Stores the path in the config file.
    config[cat][subfolder] = file_path + '/'


def make_root_paths(config):
    '''
    Add the root path to all the paths in the specification file.

    :param config: Dict containing program parameters.
    :return:
    '''
    # Get the path to the master folder
    root_path = os.path.dirname(os.path.normpath(os.path.dirname(os.path.abspath(__file__))))

    # Creats the new paths
    config['paths']['dataset_path'] = root_path + config['paths']['dataset_path']
    config['plot']['plot_folder_name'] = root_path + '/plots/' + config['plot']['plot_folder_name']
    config['utils']['save_model_name'] = root_path + '/models/' + config['utils']['save_model_name']
    config['pre_trained']['path_saved_model'] = root_path + '/models/' + config['pre_trained']['path_saved_model']



def pre_train_config(config):
    '''
    Changes the config file to use settings from the pre trained model. Model, optimizer and loss function types are
    always taken from the trained model. The hyper parameters from the trained model is used if 'use_pre_trained_param'
    is set to True.

    :param config: Dict containing program parameters.
    :return:
    '''

    # Opens the specification file for the pre trained model and uses them to overwrite the config dict.
    with open(config['pre_trained']['path_saved_model'] + '/ModelSpesification.yaml', "r") as file:
        pre_config = yaml.load(file, Loader=yaml.FullLoader)
    config['model_param'] = pre_config['model_param']

    if config['pre_trained']['use_pre_trained_param']:
        config['hyper_param'] = pre_config['hyper_param']

def make_folder(path):
    '''
    Creates a subfolder in the directory.
    :param path: Path to the directory.
    :return:
    '''

    # Make the folder.
    os.mkdir(path)

    return

