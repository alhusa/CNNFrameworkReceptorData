import torch.nn.functional as F



def cross_entropy(pred, labels):
    '''
    Uses the pytorch cross entropy loss function.
    :param pred: Predicted labels.
    :param labels: True labels.
    :return: The cross entropy loss.
    '''
    return F.cross_entropy(pred, labels)