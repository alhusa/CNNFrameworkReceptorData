import torch.nn as nn
import torch.nn.functional as F





class CNN_relu(nn.Module):
    def __init__(self, input_size, padding=0):
        '''
        A simple CNN with relu.
        '''
        super(CNN_relu, self).__init__()
        kernel_size = [5, 5, 5]
        self.dense_input = calc_output(input_size, 0, kernel_size)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=5)
        self.fc1 = nn.Linear(in_features=(self.dense_input), out_features=2)


    def forward(self, X,batch_size):

        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))

        X = X.reshape((batch_size,self.dense_input))
        X = self.fc1(X)

        return X



