import torch


def sgd(config, model):
    '''
    Creates a PyTorch SGD optimiser.

    :param config: Dict containing program parameters.
    :param model: The current ML model.
    :return: The optimizer class.
    '''
    return torch.optim.SGD(model.parameters(), lr=config['hyper_param']['learning_rate'])