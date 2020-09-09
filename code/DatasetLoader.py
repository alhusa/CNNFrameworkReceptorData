import torch.utils.data as data
import pickle
import numpy as np
import torch

class OnehotDataset(data.Dataset):
    """
    The pytorch dataloader class. Loads the one hot encoded dataset:
    :param:
    """
    def __init__(self, data,labels):

        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        """
        :return: The number of samples
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        Get an item from the dataset using the given index:
        :param index:
        :return:
        """
        return self.data[index], self.labels[index]


def OnehotDatasetSplit(dataset_path, split_percent, batch_size, max_data = 100000):
    '''
    Splits the one hot encoded dataset into training and test data. Uses the PyTorch dataloader.
    :param datasetPath: Path to the one hot encoded dataset.
    :param split_percent: The percentage of the data that should be used for testing.
    :param batch_size: The size of the batches.
    :return: A test and a train dataset.
    '''


    # Opens the file specified
    file = open(dataset_path, 'rb')

    # Loads the data from file.
    data = pickle.load(file)
    labels = pickle.load(file)


    # Creates and shuffles indices.
    indices = list(range(len(labels)))
    np.random.shuffle(indices)

    data = data[indices, :, :, :]
    labels = labels[indices]

    data = data[:max_data, :, :, :]
    labels = labels[:max_data]

    indices = list(range(len(labels)))
    np.random.shuffle(indices)

    print(split_percent)
    if split_percent != 0:
        # Creates a index which splits the dataset according to the split percentage.
        split = int(split_percent * len(labels))

        # Splits the indices into training and testing indices.
        train_ind = indices[split:]
        test_ind = indices[:split]

        # Loads a training and testing set using the one hot dataloader.
        load_train = torch.utils.data.DataLoader(OnehotDataset(data[train_ind,:,:,:],labels[train_ind]), batch_size=batch_size, shuffle=True)
        load_test = torch.utils.data.DataLoader(OnehotDataset(data[test_ind,:,:,:],labels[test_ind]), batch_size=batch_size, shuffle=True)
    else:
        load_test = torch.utils.data.DataLoader(OnehotDataset(data,labels), batch_size=batch_size, shuffle=False)
        load_train = None


    return load_train,load_test


