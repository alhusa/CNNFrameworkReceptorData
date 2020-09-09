import torch
import numpy as np
import yaml

import Plotting as plot
from DirManagement import make_path
class Run():
    '''
    Class that trains or evaluates the models.
    '''
    def __init__(self, config, Plot):
        '''
        Stores information from the config in the class.

        :param config: Dict containing program parameters.
        :param Plot: An object of the Plot class.
        '''
        self.n_epoch = config['hyper_param']['n_epoch']
        self.verbose = config['utils']['verbose']
        self.save_model = config['utils']['save_model']
        self.save_model_path = config['utils']['save_model_name']
        self.Plot = Plot
        self.config = config

    def run_epoch(self, model, optimizer, loss_function, n_epoch, data_loader, is_training, verbose=True):
        '''
        Runs the model for one epoch using batches.

        :param model: The model that is used.
        :param n_epoch: The number of the epoch.
        :param data_loader: The input data.
        :param optimizer: A torch optimizer.
        :param is_training: A boolean. If the model is training or not.
        :param verbose: A boolean. Whether to print information while running.
        :return: Average loss and accuracy for the epoch.
        '''

        # Sets the model for training or evaluation mode
        # Not needed for models without layers that work differently during training and testing
        if is_training: model.train()
        else: model.eval()

        total_correct = 0
        total_loss = 0
        for batch_ind, batch_data in enumerate(data_loader):
            data = batch_data[0]
            labels = batch_data[1]
            batch_size = len(labels)

            if is_training:
                # Runs the model forwards.
                pred = model.forward(data,batch_size)

                # Claculates the loss and adds it to the total loss.
                loss = loss_function(pred, labels)
                total_loss += loss.item()#.detach().numpy()

                # Set the gradients to zero.
                optimizer.zero_grad()

                # Performs the backward propagation.
                loss.backward()

                # Updates the weights
                optimizer.step()
            else:
                # Deactivates the autograd engine.
                with torch.no_grad():
                    # Runs the model forwards.
                    pred = model.forward(data,batch_size)

                    # Calculates the loss and adds it to the total loss.
                    loss = loss_function(pred, labels)
                    total_loss += loss.item()#.detach().numpy()


            # Finds the number of correct labels.
            pred_labels = pred.max(1, keepdim=False)[1]
            total_correct += pred_labels.eq(labels).sum().numpy()


        # Finds the average loss and accuracy for the epoch.
        avg_loss = total_loss / len(data_loader)
        accuracy = total_correct / len(data_loader.dataset)

        # Prints information
        if verbose:
            if is_training:
                print('Training')
            else:
                print('Testing')
            print(f'Average loss for epoch {n_epoch} is {avg_loss}')
            print(f'Accuracy for epoch {n_epoch} is {accuracy}\n')


        return avg_loss, accuracy


    def train_model(self, model, optimizer, loss_fn, train_data, test_data):
        '''
        Trains the model for the given number of epochs. Saves the model if chosen in specs. Makes some simple plots
        if chosen in the specs.

        :param config: Dict containing program parameters.
        :param model: The current model
        :param optimizer: The current optimizer.
        :param loss_fn: The loss function used.
        :param train_data: The data the model should be trained on.
        :param test_data: The data the model should be tested on.
        :return:
        '''




        #Create array to store data
        train_loss = np.empty(self.n_epoch)
        train_acc = np.empty(self.n_epoch)
        test_loss = np.empty(self.n_epoch)
        test_acc = np.empty(self.n_epoch)

        for epoch in range(self.n_epoch):
            train_loss[epoch], train_acc[epoch] = self.run_epoch(model, optimizer, loss_fn,
                                                                 epoch, train_data, is_training=True,
                                                                 verbose=self.verbose)
            test_loss[epoch], test_acc[epoch] = self.run_epoch(model, optimizer, loss_fn,
                                                               epoch, test_data, is_training=False,
                                                               verbose=self.verbose)




        if self.save_model:
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, self.save_model_path  + 'model.pth')

            with open(self.save_model_path  + 'ModelSpesification.yaml', 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)




        self.Plot.loss_plot(train_loss, test_loss)
        self.Plot.acc_plot(train_acc, test_acc)


    def test_model(self, model, optimizer, loss_fn, test_data):
        '''
            Test the model for the given number of epochs.

            :param config: Dict containing program parameters.
            :param model: The current model
            :param optimizer: The current optimizer.
            :param loss_fn: The loss function used.
            :param test_data: The data that should be tested.
            :return:
            '''

        # Create array to store data
        test_loss = np.empty(self.n_epoch)
        test_acc = np.empty(self.n_epoch)

        for epoch in range(self.n_epoch):
             test_loss[epoch], test_acc[epoch] = self.run_epoch(model, optimizer, loss_fn,
                                                               epoch, test_data, is_training=False,
                                                               verbose=self.verbose)



