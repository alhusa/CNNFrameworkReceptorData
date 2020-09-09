import numpy as np
import matplotlib.pyplot as plt
import torch

from DatasetLoader import OnehotDatasetSplit

class Vis():
    '''
    Class that contains methods for visualising data.
    '''
    def __init__(self, config, Plot, train_data, test_data, num):
        '''
            Stores information from the config in the class.

            :param config: Dict containing information about how the program should be run.
            :param Plot: An object of the Plot class.
        '''
        self.Plot = Plot
        self.verbose = config['utils']['verbose']
        # self.train_data = train_data
        # self.test_data = test_data
        self.num_to_amino = np.array(['A', 'R', 'N', 'D', 'B', 'C', 'E',
                                 'Q', 'Z', 'G', 'H', 'I', 'L',
                                 'K', 'M', 'F', 'P', 'S', 'T',
                                 'W', 'Y', 'V', '*'])

        self.data, self.label = next(iter(test_data))
        self.num = num
        self.data = self.data[:num, :, :, :]
        self.labels = self.label[:num]


    def saliency_map(self, config, model):
        '''
        Creates saliency maps for the model.
        :param config: Dict containing program parameters.
        :param model: The model that should be visualised.
        :return:
        '''

        def zero_neg_grad(gradient):
            '''
            Takes in the gradients and sets the negative ones to zero.
            :param gradient:
            :return:
            '''
            grad_data = gradient.data
            grad_data[grad_data < 0] = 0
            gradient.data = grad_data
            return gradient
        guided = config['guied']


        # Make the input data require a gradient
        data = self.data.clone().requires_grad_()
        labels = self.labels.clone()

        # Set the model in evaluation mode.
        model.eval()
        if guided: runs = 2
        else: runs = 1

        for i in range(runs):
            # Run the model forward
            pred = model.forward(data,self.num)

            # Takes the values for each class from the prediction. Gathers the values in each column given by the labels.
            pred = pred.gather(1, labels.view(-1, 1)).squeeze()

            # Zeroes out negative gradients if true.
            if guided: hook = data.register_hook(zero_neg_grad)

            # Run the model backwards for each of the inputs.
            pred.backward(torch.ones(self.num, dtype=torch.float32))

            # Get the gradients for the inputs.
            saliency = data.grad.data #abs(data.grad.data)

            # Removes the hook.
            if guided: hook.remove()

            amino_index = torch.max(data.data, dim=2)

            sal_input = []
            for i in range(self.num):
                sal_input.append(data.grad.data[i,0,amino_index[1][i,0,:],np.arange(data.data.shape[3])])

            # Zero out the gradietns.
            model.zero_grad()

            # Plots the maps
            self.Plot.saliency_map_plot(saliency[:,0,:,:],self.num,labels,self.num_to_amino, guided,self.onehot_to_amino(data),sal_input)

            guided = False

    def genereate_max_input(self, config, model, target_label=1):
        '''
        Iteratively generates an input to the model that maximizes the prediction of a given class. This is done by
        updating the input bases on the gradients for the score of the given class.

        :param config: Dict containing program parameters.
        :param model: The model that should be visualised.
        :param target_label: The class to be maximized
        :return:
        '''

        # Set the model in evaluation mode.
        model.eval()

        # Generate a random input.
        gen_seq = torch.randn(1, 1, 23, 20).requires_grad_()

        # Store the information from the config file.
        lamb = config['lamb']
        iterations = config['iterations']
        fig_each_iter = config['fig_each_iter']
        learning_rate = config['learning_rate']

        # Loops for a given number of iterations
        for i in range(iterations):

            # Make a prediction based on the input.
            pred = model.forward(gen_seq,1)

            # Get the score for the given label.
            score = pred[0][target_label]

            # Generate the gradients of the input.
            score.backward()

            # Get the gradients of the input.
            gradient = gen_seq.grad.data

            # Regulate the gradient
            gradient -= lamb * gen_seq

            # Normalize the gradients and update the input.
            gen_seq.data += learning_rate * (gradient / gradient.norm())

            # Zero out the gradients.
            model.zero_grad()

            # Print and plots information
            if i%fig_each_iter==0:
                if self.verbose:
                    print(f'Current input after {i} iterations:')
                    print(''.join(self.onehot_to_amino(gen_seq)[0]))
                self.Plot.plot_max_input(gen_seq.data.clone()[0, 0, :, :],i,self.num_to_amino,self.onehot_to_amino(gen_seq)[0])

    def SNP(self, config, model):
        '''
        Changes a single amino acid and stores the difference in prediction probability.

        :param config: Dict containing program parameters.
        :param model: The model that should be visualised.
        :return:
        '''
          # Make the input data require a gradient.
        seq = self.data.clone()
        labels = self.labels.clone()

        # Array to store the difference.
        pred_diff = np.zeros((self.num,seq.data.shape[2],seq.data.shape[2]))

        for k in range(seq.data.shape[0]):

            # Runs the model forwards.
            pred = model.forward(seq[k:k+1,:,:,:], 1)

            # Loops over each position in the input.
            for i in range(seq.data.shape[3]):
                for j in range(seq.data.shape[2]):

                    # Creates a new column
                    snp = torch.zeros(seq.data.shape[2], dtype=torch.int32)
                    snp[j] = 1

                    # Changes one column of the input at the time.
                    mod_seq = seq[k:k+1,:,:,:].clone()
                    mod_seq.data[:,:,:,i] = snp

                    # Run the model forward
                    snp_pred = model.forward(mod_seq, 1)

                    # Stores the difference in prediction
                    pred_diff[k][j][i] = snp_pred.data[0][1] - pred.data[0][1]


        self.Plot.snp_plot(pred_diff, self.num, labels, self.num_to_amino, self.onehot_to_amino(seq))

    def visualize_kernel(self, config, model):
        num_layer = 1
        for name, param in model.named_parameters():
            if name.split('.')[1] == 'weight':# and name[:4]== 'conv':
                data = param.data.clone()
                if name[:2]== 'fc':
                    data = data.reshape((2,1,1,model.dense_input))
                    print_text = False
                else: print_text = True

                self.Plot.visualize_kernel(data, num_layer,print_text)
                num_layer += 1

    def visualize_layer(self, config, model):
        def get_feature_map(name):
            def hook(model, input, output):
                feature_map[name] = output.detach()

            return hook
        data = self.data.clone()
        hooks = {}
        for name, module in model.named_modules():
            hooks[name] = module.register_forward_hook(get_feature_map(name))

        for i in range(data.shape[0]):
            feature_map = {}
            model.forward(data[i:i+1,:,:,:],1)
            self.Plot.seq_heatmap(data[i,0,:,:],self.num_to_amino,self.onehot_to_amino(data[i:i+1,:,:,:])[0],i)
            for module, map in feature_map.items():
                if module[:4]== 'conv':
                    self.Plot.visualize_layer(map.data, int(module[-1]),i)


        for name, hook in hooks.items():
            hook.remove()


    def onehot_to_amino(self, input):
        amino_seq = []
        for i in range(input.data.shape[0]):
            amino_seq.append(self.num_to_amino[np.argmax(input.data[i, 0, :, :], axis=0)])

        return amino_seq
