import matplotlib.pyplot as plt
import seaborn
import numpy as np
from DirManagement import make_folder
import matplotlib.gridspec as gridspec
import torch


class Plot():
    '''
    Class for plotting information from the different methods.
    '''
    def __init__(self, config, model, optimizer):
        '''
        Stores information from the config in the class.

        :param config: Dict containing program parameters.
        '''

        self.save_fig = config['plot']['save_fig']
        self.path = config['plot']['plot_folder_name']
        self.verbose = config['utils']['verbose']

        if self.save_fig: self.store_model_info(model,optimizer)

    def store_model_info(self, model,optimizer):
        '''
        Store information about the model used in the plot folder.
        :param model: The model used.
        :param optimizer: The optimizer used.
        :return:
        '''

        # Open the file and writes the model information in the file.
        with open(self.path + 'model_info.txt', "w+") as file:
            file.write(str(model)+'\n')
            file.write(str(optimizer))
    def save_info_acc_loss(self, train_data, test_data, type_data):
        '''
        Stores aaccuracy and loss information to a file
        :param train_data: Training data to be stored.
        :param test_data: Testing data to be stored.
        :param type_data: What type of data is stored.
        :return:
        '''

        with open(self.path + 'run_info.txt', "a+") as file:
            file.write(f'{type_data} for each epoch:\n')
            file.write(f'There are {train_data.shape[0]} epochs.\n')
            file.write('Train data:\n')
            np.savetxt(file, train_data, delimiter=',')
            file.write(f'Last value: {train_data[len(train_data)-1]}. Max value: {np.max(train_data)}. \n'
                       f'Min value: {np.min(train_data)}. Mean value: {np.mean(train_data)}. \n'
                       f'Standard deviation: {np.std(train_data)}\n\n')
            file.write('Test data:\n')
            np.savetxt(file, test_data, delimiter=',')
            file.write(f'Last value: {test_data[len(train_data) - 1]}. Max value: {np.max(test_data)}. \n'
                       f'Min value: {np.min(test_data)}. Mean value: {np.mean(test_data)}. \n'
                       f'Standard deviation: {np.std(test_data)}\n\n\n')

    def loss_plot(self, train_loss, test_loss):
        '''
        Plots the average loss for each epoch for the test and train data. Saves the figure if chosen in specs and shows
        the figure if verbose is true.

        :param config: Dict containing information about how the program is run.
        :param train_loss: Average loss of the training data.
        :param test_loss: Average loss of the test data.
        :return:
        '''

        # Creates a array with the length of the number of epochs.
        x = np.arange(len(train_loss))

        # Plots a simple line plot
        plt.figure()
        plt.title('Average loss for each epoch')
        plt.plot(x, train_loss, 'g', label='Train')
        plt.plot(x, test_loss, 'r', label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Average loss')

        # Gets the path of the folder to and saves the figure in the folder if chosen in the specs.
        if self.save_fig:
            self.save_info_acc_loss(train_loss, test_loss, 'Loss')
            plt.savefig(self.path + 'avg_loss_plot')

        # Shows the plot if verbose is true.
        if self.verbose:
            plt.show()
        plt.close()

    def acc_plot(self, train_acc, test_acc):
        '''
            Plots the average accuracy for each epoch for the test and train data. Saves the figure if chosen in specs and shows
            the figure if verbose is true.

            :param train_loss: Average loss of the training data.
            :param test_loss: Average loss of the test data.
            :return:
        '''

        # Creates a array with the length of the number of epochs.
        x = np.arange(len(train_acc))

        # Plots a simple line plot
        plt.figure()
        plt.title('Average accuracy for each epoch')
        plt.plot(x, train_acc, 'g', label='Train')
        plt.plot(x, test_acc, 'r', label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Average accuracy')

        # Gets the path of the folder to and saves the figure in the folder if chosen in the specs.
        if self.save_fig:
            self.save_info_acc_loss(train_acc, test_acc, 'Accuracy')
            plt.savefig(self.path + 'avg_acc_plot')

        # Shows the plot if verbose is true.
        if self.verbose:
            plt.show()
        plt.close()

    def plot_max_input(self, input, iter, annotation, gen_seq):
        '''
        Plot a map showing the maximum input.
        :param input: Data to be plotted.
        :param label: The label for the input.
        :param annotation: Annotation for the plots.
        :param iter: How many iterations used.
        :return:
        '''
        plt.figure()
        fig, ax = plt.subplots(1,1)
        ax.imshow(input)
        plt.title(f'Heatmap of the maximum input')
        ax.set_yticks(np.arange(len(annotation)))
        ax.set_yticklabels(annotation)
        ax.set_xticks(np.arange(len(gen_seq)))
        ax.set_xticklabels(gen_seq)
        plt.xlabel('Input seq')
        plt.ylabel('Amino acid')

        # Gets the path of the folder to and saves the figure in the folder if chosen in the specs.
        if self.save_fig:
            plot_dir = self.path + 'max_input/'
            if iter == 0:
                make_folder(plot_dir)
            plt.savefig(plot_dir + f'max_input_{iter}')

        # Shows the plot if verbose is true.
        if self.verbose:
            plt.show()
        plt.close()
    def saliency_map_plot(self, input, num_sal, label, annotation, guided, input_seq, input_sal):
        '''
        Plot saliency maps.
        :param input: Data to be plotted.
        :param num_sal: Numper of maps.
        :param label: The label for the input.
        :param annotation: Annotation for the plots.
        :param guided: If guided backprop is used or not.
        :param input_seq: The input sequence that was used.
        :return:
        '''

        for i in range(num_sal):

            fig = plt.figure()#constrained_layout=True)# figsize=(10*ratio,10))
            widths = [20]
            heights = [22,1]
            spec = fig.add_gridspec(ncols=1, nrows=2, width_ratios=widths,
                                      height_ratios=heights)
            ax1 = fig.add_subplot(spec[0, 0])
            ax2 = fig.add_subplot(spec[1, 0])


            vmax = torch.max(input[i])
            vmin = torch.min(input[i])
            im1 = ax1.imshow(input[i], cmap=plt.cm.hot, vmin=vmin, vmax=vmax)
            im2 = ax2.imshow(input_sal[i][None,:], cmap=plt.cm.hot, vmin=vmin, vmax=vmax)

            extra_title = ''
            if guided: extra_title = ' with zero grad'
            if label[i]==1:
                ax1.set_title(f'Saliency map for input with implanted motif' + extra_title)
            else:
                ax1.set_title(f'Saliency map for input without implanted motif' + extra_title)
            ax1.set_yticks(np.arange(len(annotation)))
            ax1.set_yticklabels(annotation)
            ax1.xaxis.set_visible(False)
            ax2.set_xticks(np.arange(len(input_seq[i])))
            ax2.set_xticklabels(input_seq[i])
            ax2.yaxis.set_visible(False)
            ax2.set_xlabel('Input sequence')
            ax1.set_ylabel('Amino acid')

            p0 = ax1.get_position().get_points().flatten()
            # p1 = ax2.get_position().get_points().flatten()

            ax_cbar = fig.add_axes([p0[3]-0.08,p0[1],0.03,p0[2]-0.025])
            plt.colorbar(im1,cax=ax_cbar, orientation='vertical')
            #fig.colorbar(im1,cax=cax)

            # Gets the path of the folder to and saves the figure in the folder if chosen in the specs.
            if self.save_fig:
                if guided: plot_dir = self.path + 'guided_saliency_map/'
                else: plot_dir = self.path + 'saliency_map/'
                if i == 0:
                    make_folder(plot_dir)
                plt.savefig(plot_dir + f'saliency_implanted_{label[i]==1}_{i}')

            # Shows the plot if verbose is true.
            if self.verbose:
                plt.show()
            plt.close()

    def snp_plot(self, input, num_snp, label, annotation, input_seq):
        '''
        Plot changes to the score based on snp.
        :param input: Data to be plotted.
        :param num_sal: Numper of maps.
        :param label: The label for the input.
        :param annotation: Annotation for the plots.
        :param input_seq: The input sequence that was used.
        :return:
        '''

        for i in range(num_snp):
            plt.figure()
            fig, ax = plt.subplots(1,1)
            im = ax.imshow(input[i], cmap=plt.cm.winter)
            if label[i]==1:
                plt.title(f'SNP map for input with implanted motif')
            else:
                plt.title(f'SNP map for input without implanted motif')
            ax.set_yticks(np.arange(len(annotation)))
            ax.set_yticklabels(annotation)
            ax.set_xticks(np.arange(len(input_seq[i])))
            ax.set_xticklabels(input_seq[i])
            plt.xlabel('Input seq')
            plt.ylabel('Amino acid')
            fig.colorbar(im)

            # Gets the path of the folder to and saves the figure in the folder if chosen in the specs.
            if self.save_fig:
                plot_dir = self.path + 'snp_map/'
                if i == 0:
                    make_folder(plot_dir)
                plt.savefig(plot_dir + f'SNP_implanted_{label[i]==1}_{i}')

            # Shows the plot if verbose is true.
            if self.verbose:
                plt.show()
            plt.close()

    def visualize_kernel(self, input, layer, print_text):
        '''
        Plot heatmaps of the kernels.
        :param input: Data to be plotted.
        :param layer: Which layer of the model the input belongs to
        :return:
        '''

        vmax = torch.max(input)
        vmin = torch.min(input)
        fig = plt.figure(constrained_layout=False,figsize=(16,12))
        if not print_text: fig.suptitle(f'Dense layer weights', size=28)
        else: fig.suptitle(f'Kernels layer {layer}', size=28)
        widths = np.ones(input.shape[1]+1)*5
        widths[-1] = 1
        heights = np.ones(input.shape[0])
        #for i in range()
        spec = gridspec.GridSpec(ncols=input.shape[1]+1, nrows=input.shape[0], figure=fig, width_ratios=widths,
                                      height_ratios=heights)
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                ax = fig.add_subplot(spec[i, j])
                if i == 0:
                    ax.xaxis.set_label_position('top')
                    ax.set_xticklabels([])
                    ax.set_xlabel(f'Filter channel {j+1}', fontsize=16)
                else: ax.xaxis.set_visible(False)
                if j == 0:
                    ax.set_ylabel(f'Filter {i+1}', fontsize=16)
                    ax.set_yticklabels([])
                else: ax.yaxis.set_visible(False)

                if print_text:
                    for y in range(input.shape[2]):
                        for x in range(input.shape[3]):
                            ax.text(x + 0, y + 0, f'{input[i,j,y,x]:.2f}',
                                     horizontalalignment='center',
                                     verticalalignment='center', fontsize=16)

                im = ax.imshow(input[i,j,:,:], cmap=plt.cm.summer, vmin=vmin, vmax=vmax)


        ax_cbar = fig.add_axes([0.95, 0.2, 0.02, 0.7])
        cbar = plt.colorbar(im, cax=ax_cbar, orientation='vertical')
        cbar.ax.tick_params(labelsize=16)
        spec.tight_layout(fig, rect=[0, 0, 1, 0.95])

        # Gets the path of the folder to and saves the figure in the folder if chosen in the specs.
        if self.save_fig:
            plot_dir = self.path + 'kernel_map/'
            if layer == 1:
                make_folder(plot_dir)
            plt.savefig(plot_dir + f'kernels_layer{layer}')

        # Shows the plot if verbose is true.
        if self.verbose:
            plt.show()
        plt.close()

    def visualize_layer(self, input, layer, num):
        '''
        Plot heatmaps of the feature maps.
        :param input: Data to be plotted.
        :param layer: Which layer of the model the input belongs to
        :return:
        '''

        vmax = torch.max(input)
        vmin = torch.min(input)

        fig = plt.figure(constrained_layout=False, figsize=(16, 12))

        fig.suptitle(f'Feature maps for layer {layer}', size=28)
        widths = np.ones(input.shape[1] + 1) * 5
        widths[-1] = 1
        heights = np.ones(input.shape[0])
        spec = gridspec.GridSpec(ncols=input.shape[1] + 1, nrows=input.shape[0], figure=fig, width_ratios=widths,
                                 height_ratios=heights)
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                ax = fig.add_subplot(spec[i, j])
                if i == 0:
                    ax.xaxis.set_label_position('top')
                    ax.set_xticklabels([])
                    ax.set_xlabel(f'Feature map channel {j + 1}', fontsize=16)
                else:
                    ax.xaxis.set_visible(False)
                if j == 0:
                    ax.set_ylabel(f'Feature map {i + 1}', fontsize=16)
                    ax.set_yticklabels([])
                else:
                    ax.yaxis.set_visible(False)

                im = ax.imshow(input[i, j, :, :], cmap=plt.cm.summer, vmin=vmin, vmax=vmax)

        ax_cbar = fig.add_axes([0.95, 0.2, 0.02, 0.7])
        cbar = plt.colorbar(im, cax=ax_cbar, orientation='vertical')
        cbar.ax.tick_params(labelsize=16)
        spec.tight_layout(fig, rect=[0, 0, 1, 0.95])

        # Gets the path of the folder to and saves the figure in the folder if chosen in the specs.
        if self.save_fig:
            plot_dir = self.path + f'feature_maps/feature_maps{num}/'
            plt.savefig(plot_dir + f'feature_map_layer{layer}')

        # Shows the plot if verbose is true.
        if self.verbose:
            plt.show()
        plt.close()



    def seq_heatmap(self, input, annotation, seq, num):
        plt.figure()
        fig, ax = plt.subplots(1, 1)
        ax.imshow(input,cmap=plt.cm.summer)
        plt.title(f'Heatmap of the input')
        ax.set_yticks(np.arange(len(annotation)))
        ax.set_yticklabels(annotation)
        ax.set_xticks(np.arange(len(seq)))
        ax.set_xticklabels(seq)
        plt.xlabel('Input seq')
        plt.ylabel('Amino acid')

        # Gets the path of the folder to and saves the figure in the folder if chosen in the specs.
        if self.save_fig:
            plot_dir = self.path + f'feature_maps/'
            if num == 0:
                make_folder(plot_dir)

            plot_dir = plot_dir + f'feature_maps{num}/'
            make_folder(plot_dir)
            plt.savefig(plot_dir + f'input_sequence')

        # Shows the plot if verbose is true.
        if self.verbose:
            plt.show()
        plt.close()