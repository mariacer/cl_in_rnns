#!/usr/bin/env python3
# Copyright 2019 Benjamin Ehret, Maria Cervera
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title          :sequential/plotting_sequential.py
# @author         :be, mc
# @contact        :mariacer@ethz.ch
# @created        :24/03/2020
# @version        :1.0
# @python_version :3.6.8
"""
Plotting functions for this subpackage are contained in this file.
"""

import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_test_results(out_dir, classification=False):
    """Plot the test results from an existing stored run.

    Depending on whether it is a classification task or not, the loss or the
    accuracies will be plotted.

    Args:
        out_dir (str): The directory to analyse.
    """
    if classification:
        var_short_name = 'acc'
    else:
        var_short_name = 'loss'

    results_path = os.path.join(out_dir, 'test_%s.pickle'%var_short_name)
    save_path = os.path.join(out_dir, 'test_%s.pdf'%var_short_name)

    try:
        with open(results_path, "rb") as f:
            test_results = pickle.load(f)
    except:
        raise ValueError('The requested directory does not have a stored '+
                         'set of results to be plotted.') 

    plt.figure()
    im = plt.imshow(test_results.transpose(), cmap=plt.get_cmap('viridis'), \
        vmin=0, vmax=100)
    plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.xlabel("Trained Task")
    plt.ylabel("Tested Task")
    plt.grid(False)
    plt.savefig(save_path)

def plot_train_loss(out_dir):
    """Plot the loss during training from an existing stored run.

    Args:
        out_dir (str): The directory to analyse.

    """
    loss_path = os.path.join(out_dir, 'train_loss.pickle')
    save_path = os.path.join(out_dir, 'train_loss.pdf')

    try:
        with open(loss_path, "rb") as f:
            loss = pickle.load(f)
    except:
        raise ValueError('The requested directory does not have a stored '+
                         'set of losses to be plotted.')

    plt.figure()
    for t in range(len(loss)):
        iters = np.arange(0, len(loss[t]['task']))*50
        plt.plot(iters, loss[t]['task'])
    plt.grid(False)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(save_path) 

def plot_outputs(outputs, targets, config, mode='test', curr_iter='',
                 sample=20, task_id=None, to_binary=False):
    """Plot the outputs of the network and the corresponding targets
    during testing.

    Args:
        outputs (list): The list of outputs for all tested tasks.
        targets (list): The list of targets.
        config: Command-line arguments.
        mode (str, optional): Indicates whether we are training or testing.
        curr_iter (str, optional): The current training iteration (only 
            relevant on `train` mode).
        sample (int, optional): The sample to be plotted.
        task_id (int, optional): The current task (only relevant on `train` 
            mode).
        to_binary (boolean, optional): Whether to convert the output into
            a binary pattern.

    """
    if config.show_plots:

        figure_dir = os.path.join(config.out_dir, 'figures')
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)

        num_tasks = len(outputs)
        assert(len(outputs) == len(targets))

        # from grid_strategy import strategies
        # grid = strategies.RectangularStrategy.get_grid_arrangement(num_tasks)

        nrows = np.min([5, num_tasks])*2
        ncols = int(np.ceil(num_tasks/5))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

        for t in range(num_tasks):
            output = outputs[t].detach()
            output = output[:, sample, :].t()
            if to_binary:
                output[output<0.5] = 0
                output[output>=0.5] = 1
            target = targets[t][:, sample, :].t()
            axes[t*2].set_ylabel('Task %i target'%t)
            im1 = axes[t*2].imshow(target.cpu())
            fig.colorbar(im1, ax=axes[t*2])
            axes[t*2+1].set_xlabel('time')
            axes[t*2+1].set_ylabel('output')
            im2 = axes[t*2+1].imshow(output.cpu())
            fig.colorbar(im2, ax=axes[t*2+1])

        name = mode
        if mode == 'train':
            name += '_task%i_iter%i'%(task_id, curr_iter)

        plt.savefig(os.path.join(figure_dir, 'outputs_%s.pdf'%name))


def visualise_task(dhandler, i, config, num_samples=20):
    """Visualize data for a given task.

    Args: 
        dhandler: The data handler for the current task.
        i (int): The current task.
        config: Command-line arguments.
        num_samples (optional, int): The number of samples to be plotted.
    """
    if config.show_plots:

        figure_dir = os.path.join(config.out_dir, 'figures')
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)

        # get all test data for this task
        X_real = dhandler.input_to_torch_tensor( \
                dhandler.get_test_inputs(), 'cpu', mode='inference')
        T_real = dhandler.output_to_torch_tensor( \
                dhandler.get_test_outputs(), 'cpu', mode='inference')

        input_unit = 0
        output_unit = 0
        sample = 0

        plt.figure()

        # Plot many samples for a given input and output unit.
        plt.subplot(2,2,1)
        plt.title('Input samples for input unit %i'%input_unit)
        plt.imshow(X_real[:,:num_samples,input_unit].t())
        plt.ylabel('samples')
        plt.xlabel('t')
        plt.grid(False)
        plt.subplot(2,2,3)
        plt.title('Target samples for output unit %i'%output_unit)
        plt.imshow(T_real[:,:num_samples,output_unit].t())
        plt.ylabel('samples')
        plt.xlabel('t')
        plt.grid(False)

        # Plot many input and output units for a given sample.
        plt.subplot(2,2,2)
        plt.title('Input units for sample%i'%sample)
        plt.imshow(X_real[:,sample,:].t())
        plt.ylabel('input units')
        plt.xlabel('t')
        plt.grid(False)
        plt.subplot(2,2,4)
        plt.title('Output units for sample%i'%sample)
        plt.imshow(T_real[:,sample,:].t())
        plt.ylabel('output units')
        plt.xlabel('t')
        plt.grid(False)

        plt.savefig(os.path.join(figure_dir, 'task%i.pdf'%i))


def visualise_data(dhandlers, config, device):
    """Visualize inputs and targets of tasks.

    Args: 
        dhandlers (list): The data handlers for all tasks.
        config: Command-line arguments.
        device: The torch.device to use.
    """
    if config.show_plots:

        figure_dir = os.path.join(config.out_dir, 'figures')
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)

        n_tasks = len(dhandlers)
        sample = 0

        # Plot one sample for all tasks.
        plt.figure()
        for i in range(n_tasks):
            dhandler = dhandlers[i]

            # get all test data for this task
            X_real = dhandler.input_to_torch_tensor( \
                    dhandler.get_test_inputs(), 'cpu',  mode='inference')
            T_real = dhandler.output_to_torch_tensor( \
                    dhandler.get_test_outputs(), 'cpu', mode='inference')

            # plot all data for this task
            plt.subplot(2, n_tasks, i + 1)
            plt.title('task %i'%i)
            plt.imshow(X_real[:,sample,:].t())
            plt.xlabel('t')
            plt.grid(False)
            plt.subplot(2, n_tasks, i + 1 + n_tasks)
            plt.imshow(T_real[:,sample,:].t())
            plt.xlabel('t')
            plt.grid(False)

        plt.savefig(os.path.join(figure_dir, 'data.pdf'))


def configure_matplotlib_params(fig_size = [6.4, 4.8], two_axes=True,
                                font_size=10, usetex=False):
    """Helper function to configure default matplotlib parameters.

    Args:
        fig_size: Figure size (width, height) in inches.
        usetex (bool): Whether ``text.usetex`` should be set (leads to an
            error on systems that don't have latex installed).
        font (str): The font.
    """
    params = {
        'axes.labelsize': font_size,
        'font.size': font_size,
        'font.sans-serif': ['Times New Roman'],
        'text.usetex': usetex,
        'text.latex.preamble': [r'\usepackage[scaled]{helvet}',
                                r'\usepackage{sfmath}'],
        'font.family': 'sans-serif',
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'axes.titlesize': font_size,
        'axes.spines.right' : not two_axes,
        'axes.spines.top' : not two_axes,
        'figure.figsize': fig_size,
        'legend.handlelength': 0.5,
        'xtick.major.pad':0.3,
        'ytick.major.pad':0.3,
        'axes.labelpad':0.1,
        'figure.subplot.bottom':0.27,
        'figure.subplot.left':0.27,
        'figure.subplot.right':0.98,
        'figure.subplot.top':0.95,
        'legend.frameon':False
    }

    matplotlib.rcParams.update(params)
    del matplotlib.font_manager.weight_dict['roman']
    matplotlib.font_manager._rebuild()