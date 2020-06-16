#!/usr/bin/env python3
# Copyright 2020 Maria Cervera

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
# @title           :sequential/copy/state_space_analysis.py
# @author          :mc
# @contact         :mariacer@ethz.ch
# @created         :16/04/2020
# @version         :1.0
# @python_version  :3.6.8
"""
Study dimensionality of hidden states for different continual learning settings
-------------------------------------------------------------------------------

In this script, we perform an analysis of the dimensionality of the hidden
states of a recurrent network, for different continual learning settings.
Most interestingly, this framework can be used to study the relationship
between the dimensionality of the hidden space, and the memory requirements
of the task.

Specifically, this script analyses the hidden dimensionality of checkpointed
models in a certain run, for which the path needs to be provided.
One can equally provide a path to a folder containing the results of
several individual runs (similar to the output of an hpsearch), where individual
runs differ in some complexity measure. For example, this can be the length of
the presented patterns for the copy task, or the number of classes per task in
classification tasks. Only when multiple runs are presented, the correlation
between task memory requirements and dimensionality of the hidden space will be
performed.

Run as follows:

.. code-block:: 

    python3 state_space_analysis.py path/to/results/folder/

For running these analyses, one needs to have run before the following:

.. code-block::

    python3 hpsearch.py --grid_module=ewc_study_config

Making sure that the different runs have different complexities, and that they
all have the following arguments activated for all runs:
``--store_final_models --store_during_models``.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import argparse
import os
import pickle
import shutil
from subprocess import call
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import glob
from scipy.optimize import curve_fit
import numpy as np
from collections import OrderedDict
import logging
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D

import sequential.plotting_sequential as plc
import sequential.train_utils_sequential as stu
from sequential.train_sequential import test
import utils.sim_utils as sutils
from utils import logger_config
import utils.ewc_regularizer as ewc
import torch
import utils.misc as misc
from sequential import train_args_sequential

def load_models(out_dir, device, logger, mnet, hnet, task_id=None):
    """Small wrapper to load checkpointed models for mnet and hnet.

    Args:
        out_dir (str): The path to the output directory.
        device: The device.
        logger: The logger.
        mnet: The main network.
        hnet: The hypernetwork.
        task_id (int, optional): The id of the last task to be trained for the
            checkpointed model to be loaded.

    Returns:
        (tuple): Tuple containing:

        - **mnet**: The checkpointed main network.
        - **hnet**: The checkpointed hypernetwork.

    """
    # Since the main network doesn't have fisher value attributes, but these
    # have been checkpointed, we manually add them before loading the model.
    # Note that when using online EWC, task_id has no effect on the buffer 
    # naming, since the Fisher is accumulated over all tasks (whereas the 
    # original EWC had a Fisher per task).s
    online = True
    for i, p in enumerate(mnet.internal_params):
        buff_w_name, buff_f_name = ewc._ewc_buffer_names(task_id, i, online)
        mnet.register_buffer(buff_w_name, torch.empty_like(p))
        mnet.register_buffer(buff_f_name, torch.empty_like(p))

    # Load the models.
    _, mnet, hnet, _ = stu.load_models(out_dir, device, logger, \
        mnet, hnet=hnet, return_models=True, task_id=task_id)

    return mnet, hnet


def get_hh_fisher_estimates(mnet):
    """Extract the diagonal fisher estimates for the hidden-to-hidden weights.

    Args:
        mnet: The main network.

    Returns:
        (list): The diagonal fisher estimates of all hidden-to-hidden params.

    """
    # Get the list of meta information about regularized weights.
    n_cm = 0 if mnet._context_mod_no_weights else mnet._num_context_mod_shapes()
    regged_tnet_meta = mnet._param_shapes_meta[n_cm:]

    diag_hh_fisher = []
    for i, p in enumerate(regged_tnet_meta):
        _, buff_f_name = ewc._ewc_buffer_names(None, i, True)
        if 'info' in p.keys() and \
                p['info'] == 'hh':
            diag_hh_fisher_i = getattr(mnet, buff_f_name)

            # We flatten the matrices and gather all params in a single list.
            diag_hh_fisher.extend(torch.flatten(diag_hh_fisher_i))

    return diag_hh_fisher


def get_int_activations(out_dir):
    """Get the internal hidden activations for all tested tasks.

    Args:
        out_dir (str): The directory to analyse.

    Return:
        (torch.Tensor): The hidden activations of all tasks. It has dimensions
            ``[seq_length, batch_size*num_tasks, num_hidden]``.
    """

    with open(out_dir + "/int_activations.pickle", "rb") as f:
        activations = pickle.load(f)

    # Concatenate the activations across all tasks.
    all_activations = torch.tensor(())
    for act in activations:
        all_activations = torch.cat((all_activations, act), dim=1)

    return all_activations


def compute_pca(x):
    """Compute PCA components and explained variance ratios.

    Args:
        x (torch.Tensor): The hidden activations. It has dimensions
            ``[num_samples, num_features]``.

    Returns:
        (np.array): The explained variance ratios for the PCs.
    """
    # Normalize the features
    x = StandardScaler().fit_transform(x)

    # Compute principal components
    pca = PCA()
    pca.fit_transform(x)

    return pca.explained_variance_ratio_


def compute_kpca(x, max_num_samples=2000):
    """Compute Kernel PCA components and get eigenvalues.

    Args:
        x (torch.Tensor): The hidden activations. It has dimensions
            ``[num_samples, num_features]``.

    Returns:
        (np.array): The normalized eigenvalues (such that they sum up to one).
    """
    # Normalize the features
    x = StandardScaler().fit_transform(x)

    # Compute kernel principal components using radial basis functions.
    pca = KernelPCA(kernel ='rbf', gamma = 15)

    # Doing Kernel PCA on the entire dataset is too computationally expensive,
    # so we randomly select some samples.
    indices = np.random.permutation(x.shape[0])[:max_num_samples]
    pca.fit_transform(x[indices, :])

    # We only look at as many eigenvalues as number of features (hidden neurons)
    eigenvalues = pca.lambdas_[:x.shape[1]]

    # Normalize their values.
    eigenvalues /= np.sum(eigenvalues)

    return eigenvalues


def get_num_dimensions(x, p_var=0.5, n_pcs=10, do_kernel_pca=False):
    """Get the dimensionality of the hidden activations on the entire sequence.

    Given a set of hidden activations, this function computes the principal 
    components, where different sequences and different timesteps are all 
    treated as different samples, and the dimensions space is determined by 
    the number of hidden neurons. We define the number of dimensions of the
    hidden space as the number of principal components needed to explain a
    certain variance :math:`p_{var}` of the data.

    Note: 
        For LSTMs, for the moment we perform the analysis on the external hidden 
        states (`h_t` and not `c_t`). For vanilla RNNs, we perform the analysis 
        on the internal hidden states.

    Args:
        x (torch.Tensor): The test hidden activations on all tasks. It has
            dimensions ``[seq_length, batch_size*num_tasks, num_hidden]`` or 
            ``[batch_size*num_tasks, num_hidden]``.
        p_var (float): The percentage of explained variance that should be
            explained by the number of principal components that this function
            returns.
        n_pcs (int): The number of PCs for which to report the explained
            variance.
        do_kernel_pca (bool, optional): If True, kernel PCA will also be used
            to copmute the number of hidden dimensions.

    Returns:
        (tuple): Tuple containing:

        - **num_dim**: The number of principal components needed to explain 
            :math:`p_{var}` % variance of the data.
        - The percentage explained variance by the first :math:`n_{pcs}`
            principal components.
        - **knum_dim**: The number of kernel PCA components needed to explain 
            :math:`p_{var}` % variance of the data.

    """
    # Treat all dimensions other than the hidden neurons as different samples.
    x = x.view(-1, x.shape[-1])

    # Make sure that we have more samples than dimensions.
    assert x.shape[0] > x.shape[1]

    # Compute the cumulative explained variances of the PCs.
    expl_vars = compute_pca(x)
    cum_expl_vars = np.cumsum(expl_vars)

    # Find the minimum number of pcs to explain the desired amount of variance.
    num_dim = next(x[0] for x in enumerate(cum_expl_vars) if x[1] >= p_var)

    # Results on kernel PCA.
    if do_kernel_pca:
        print('Doing kernel PCA, this might take a while...')
        kexpl_vars = compute_kpca(x)
        cum_kexpl_vars = np.cumsum(kexpl_vars)
        knum_dim = next(x[0] for x in enumerate(cum_kexpl_vars) if x[1] >=p_var)
        print('Doing kernel PCA, this might take a while. Done.')
    else:
        knum_dim = None

    return num_dim, cum_expl_vars[n_pcs], knum_dim


def get_num_dimensions_per_ts(x, p_var=0.75):
    """Get the dimensionality of the hidden activations per timestep.

    Here we apply the function :func:`get_num_dimensions` separately for each
    timestep.

    Args:
        (....): See docstring of function :func:`get_num_dimensions`.
        x (torch.Tensor): The test hidden activations on all tasks. It has
            dimensions ``[seq_length, batch_size*num_tasks, num_hidden]``.

    Returns:
        (list): The number of principal components needed to explain 
            :math:`p_{var}` % variance of the data in each timestep.
    
    """
    seq_length = x.shape[0]

    num_dims = np.zeros(seq_length)
    for t in range(seq_length):
        num_dims[t], _, _ = get_num_dimensions(x[t, :, :], p_var=p_var)

    return num_dims


def plot_fisher_vs_task(results, seed_groups, task_id=-1, path='', 
        fisher_log_scale=False):
    """Plot mean fisher values against number of tasks.

    Args:
        results (dict): The summary dictionary with the results of the runs.
        seed_groups (dict): The grouping of runs from different random seeds.
        task_id (int): The number of tasks trained to be taken into account.
            By default the analysis is done when all tasks have been trained on.
        path (str): The path where to save the figures.
        fisher_log_scale (boolean): Whether to use a log scale for the 
            magnitudes of the Fisher values.

    """
    num_runs = len(seed_groups.keys())
    num_seeds = len(seed_groups[list(seed_groups.keys())[0]])
    num_tasks = results[list(results.keys())[0]]['num_tasks']

    cmapp = plt.get_cmap('Oranges', num_runs)
    color = [cmapp(i/num_runs) for i in range(num_runs)]

    p = np.array([key for key in seed_groups.keys()])
    p = np.sort(p)

    plt.figure()
    for nr, pi in enumerate(p):

        fisher = []
        for out_dir in seed_groups[pi]:
            fisher.append(results[out_dir]['mean_fisher'][:task_id])
        fisher = np.array(fisher)

        # Normalize by largest value.
        fisher /= np.max(np.mean(fisher, axis=0))
        std_fisher = np.std(fisher, axis=0)
        mean_fisher = np.mean(fisher, axis=0)
        
        plt.plot(np.arange(num_tasks)[:task_id], mean_fisher, color=color[nr], \
            label='p = %i'%pi)
        plt.fill_between(np.arange(num_tasks)[:task_id], mean_fisher-std_fisher, 
            mean_fisher+std_fisher, alpha=0.1, color=color[nr])

    plt.xlabel('num task')
    plt.ylabel('normalized mean fisher')
    if fisher_log_scale:
        plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(path, 'fisher_vs_task.pdf'))


def plot_across_runs(results, seed_groups, task_id=-1, path='', 
        fisher_log_scale=False, do_kernel_pca=False):
    """Plot results across runs with different sequence lengths.

    Specifically, the following is plotted as a funcion of the pattern length:
        - the performance
        - the average fisher weights
        - the dimensionality of the hidden space

    Args:
        results (dict): The summary dictionary with the results of the runs.
        seed_groups (dict): The grouping of runs from different random seeds.
        task_id (int): The number of tasks trained to be taken into account.
            By default the analysis is done when all tasks have been trained on.
        path (str): The path where to save the figures.
        fisher_log_scale (boolean): Whether to use a log scale for the 
            magnitudes of the Fisher values.
        do_kernel_pca (bool, optional): If True, kernel PCA will also be used
            to copmute the number of hidden dimensions.

    """
    num_runs = len(seed_groups.keys())
    num_seeds = len(seed_groups[list(seed_groups.keys())[0]])
    num_tasks = results[list(results.keys())[0]]['num_tasks']
    
    # Extract the name of the complexity measure for this run.
    complexity_measure_name = results[list(results.keys())[0]][\
        'complexity_measure_name']
    complexity_measure_name = complexity_measure_name.replace('_', ' ')

    if num_tasks == 2:
        color = ['b', 'r']
    else:
        cmapp = plt.get_cmap('Blues', num_tasks)
        color = [cmapp(i/num_tasks) for i in range(num_tasks)]

    if num_tasks > 1:
        plot_fisher_vs_task(results, seed_groups, task_id=task_id, path=path,
            fisher_log_scale=fisher_log_scale)

    num_subplots = 4
    if do_kernel_pca:
        num_subplots += 1

    fig, axes = plt.subplots(num_subplots, 1, figsize=(7, num_subplots*2.5))
    for task_id in range(num_tasks):

        # Create arrays gathering the results.
        p = np.zeros(num_runs)
        acc = np.zeros((num_seeds, num_runs))
        num_hidden_dim = np.zeros((num_seeds, num_runs))
        knum_hidden_dim = np.zeros((num_seeds, num_runs))
        expl_var = np.zeros((num_seeds, num_runs))
        fisher = np.zeros((num_seeds, num_runs))
        for i, pi in enumerate(seed_groups.keys()):
            p[i] = pi
            for oi, out_dir in enumerate(seed_groups[pi]):
                acc[oi, i] = results[out_dir]['final_acc'][task_id]
                num_hidden_dim[oi, i] = \
                    results[out_dir]['num_hidden_dim'][task_id]
                expl_var[oi, i] = results[out_dir]['expl_var'][task_id]
                knum_hidden_dim[oi, i] = \
                    results[out_dir]['knum_hidden_dim'][task_id]
                fisher[oi, i] = results[out_dir]['mean_fisher'][task_id]
        acc_std = np.std(acc, axis=0)
        acc = np.mean(acc, axis=0)
        std_fisher = np.std(fisher, axis=0)
        mean_fisher = np.mean(fisher, axis=0)
        num_hidden_dim_std = np.std(num_hidden_dim, axis=0)
        num_hidden_dim = np.mean(num_hidden_dim, axis=0)
        expl_var_std = np.std(expl_var, axis=0)
        expl_var = np.mean(expl_var, axis=0)
        knum_hidden_dim_std = np.std(knum_hidden_dim, axis=0)
        knum_hidden_dim = np.mean(knum_hidden_dim, axis=0)

        # Sort the lists by increasing sequence length.
        seq_order = np.argsort(p)
        p = np.array(p[seq_order])
        acc = np.array(acc[seq_order])
        acc_std = np.array(acc_std[seq_order])
        num_hidden_dim = np.array(num_hidden_dim[seq_order])
        expl_var = np.array(expl_var[seq_order])
        knum_hidden_dim = np.array(knum_hidden_dim[seq_order])
        mean_fisher = np.array(mean_fisher[seq_order])
        std_fisher = np.array(std_fisher[seq_order])

        ### Plot the performance vs. sequence length.
        axes[0].plot(p, acc, color=color[task_id])
        axes[0].fill_between(p, acc-acc_std, acc+acc_std, alpha=0.1, \
            color=color[task_id])

        ### Plot the hidden dimensionality vs. sequence length.
        axes[1].plot(p, num_hidden_dim, color=color[task_id])
        axes[1].fill_between(p, num_hidden_dim-num_hidden_dim_std, \
            num_hidden_dim+num_hidden_dim_std, alpha=0.1, color=color[task_id])

        ### Plot the explained variance by first 10 PCs vs. sequence length.
        axes[2].plot(p, expl_var, color=color[task_id])
        axes[2].fill_between(p, expl_var-expl_var_std, \
            expl_var+expl_var_std, alpha=0.1, color=color[task_id])

        if do_kernel_pca:
            axes[3].plot(p, knum_hidden_dim, color=color[task_id])
            axes[3].fill_between(p, knum_hidden_dim-knum_hidden_dim_std, \
                knum_hidden_dim+knum_hidden_dim_std, alpha=0.1,
                color=color[task_id])

        ### Plot the fisher weights vs. sequence length.
        axes[num_subplots-1].plot(p, mean_fisher, color=color[task_id])
        axes[num_subplots-1].fill_between(p, mean_fisher-std_fisher, \
            mean_fisher+std_fisher, alpha=0.1, color=color[task_id])

    axes[0].set_ylabel('% final accuracy')
    axes[0].set_xticks(p)
    axes[1].set_ylabel('num dimensions')
    axes[1].set_xticks(p)
    axes[2].set_ylabel('explained var')
    axes[2].set_xticks(p)
    if do_kernel_pca:
        axes[3].set_ylabel('num dimensions (kPCA)')
        axes[3].set_xticks(p)
    axes[num_subplots-1].set_ylabel('mean fisher')
    axes[num_subplots-1].set_xlabel(complexity_measure_name)
    axes[num_subplots-1].set_xticks(p)
    if fisher_log_scale:
        axes[num_subplots-1].set_yscale('log')

    # Prepare legend.
    custom_lines = [Line2D([0], [0], color=color[i], lw=4) for i in \
        range(num_tasks)]
    custom_lines_legend = ['%i tasks'%i for i in range(1, num_tasks+1)]
    axes[num_subplots-1].legend(custom_lines, custom_lines_legend, \
        loc='upper left')

    plt.savefig(os.path.join(path, 'complexity_effects.png'))


def plot_dim_per_ts_all_runs(results, seed_groups, task_id=-1, path='', 
    seed_number=None):
    """Plot the dimensionality in each timestep for all sequence lengths.

    Args:
        results (dict): The summary dictionary with the results of the runs.
        seed_groups (dict): The grouping of runs from different random seeds.
        task_id (int): The number of tasks trained to be taken into account.
            By default the analysis is done when all tasks have been trained on.
        path (str): The path where to save the figures.
        seed_number (int): The number of the seed iteration to be used.

    """
    num_runs = len(seed_groups.keys())
    num_seeds = len(seed_groups[list(seed_groups.keys())[0]])

    cmapp = plt.get_cmap('Oranges', num_runs)
    color = [cmapp(i/num_runs) for i in range(num_runs)]

    # Create arrays gathering the results.
    p = np.array([key for key in seed_groups.keys()])
    p = p[np.argsort(p)]

    seq_length = int(np.max(p)*2 + 1)
    flag_ts = int(np.floor(seq_length/2))

    if seed_number is None:
        seed_range = range(num_seeds)
    else:
        seed_range = [seed_number]

    plt.figure()
    for i, pi in enumerate(p):
        hidden_dim = []
        for ii in seed_range:
            out_dir = seed_groups[pi][ii]
            hidden_dim.append(results[out_dir]['num_hidden_dim_per_ts'][task_id])
        hidden_dim_std = np.std(np.array(hidden_dim), axis=0)
        hidden_dim = np.mean(np.array(hidden_dim), axis=0)    
        
        # Make sure ts 0 corresponds to stop flag.
        x = np.arange(len(hidden_dim))
        if results[out_dir]['masked'] != -1:
            x -= pi
        else:
            x -= int((len(hidden_dim))/2)

        # Display both the input sequence length and actual pattern length.
        pat = pi
        if results[out_dir]['masked'] != -1:
            pat = results[out_dir]['masked']
        plt.plot(x, hidden_dim, label='i = %i, p = %i'%(pi, pat),color=color[i])
        plt.fill_between(x, hidden_dim-hidden_dim_std, \
            hidden_dim+hidden_dim_std, alpha=0.1, color=color[i])
    plt.xlabel('ts')
    plt.ylabel('num dimensions')
    seq_length_lim = np.max(p) + (10 - np.max(p)%10)
    plt.xticks(np.arange(-seq_length_lim, seq_length_lim, 10), \
        fontname="Liberation Sans")
    # For some bizarre reason, minus signs don't show up when using Arial font
    # and rendering the plots as pdfs (for pngs it works fine).
    axes = plt.gca()
    plt.plot([0, 0], axes.get_ylim(), '--k', label='stop flag')
    plt.legend()
    suffix = ''
    if task_id != -1:
        suffix = '_task_%i'%task_id
    if seed_number is not None:
        suffix += '_seed%i'%seed_number
    plt.savefig(os.path.join(path, 'dim_per_ts%s.pdf'%suffix))


def plot_accuracy_per_ts(results, seed_groups, task_id=-1, path='',
        seed_number=None):
    """Plot the accuracy in each timestep for all sequence lengths.

    Args:
        results (dict): The summary dictionary with the results of the runs.
        seed_groups (dict): The grouping of runs from different random seeds.
        task_id (int): The number of tasks trained to be taken into account.
            By default the analysis is done when all tasks have been trained on.
        path (str): The path where to save the figures.
        seed_number (int): The number of the seed iteration to be used.

    """
    num_runs = len(seed_groups.keys())
    num_seeds = len(seed_groups[list(seed_groups.keys())[0]])

    cmapp = plt.get_cmap('Oranges', num_runs)
    color = [cmapp(i/num_runs) for i in range(num_runs)]

    # Create arrays gathering the results.
    p = np.array([key for key in seed_groups.keys()])
    p = p[np.argsort(p)]

    seq_length = int(np.max(p)*2 + 1)
    flag_ts = int(np.floor(seq_length/2))

    if seed_number is None:
        seed_range = range(num_seeds)
    else:
        seed_range = [seed_number]

    plt.figure()
    for i, pi in enumerate(p):
        hidden_dim = []
        for ii in seed_range:
            out_dir = seed_groups[pi][ii]
            hidden_dim.append(results[out_dir]['accs_per_ts'][task_id])
        hidden_dim_std = np.std(np.array(hidden_dim), axis=0)
        hidden_dim = np.mean(np.array(hidden_dim), axis=0)
        
        # Make sure ts 0 corresponds to stop flag.
        x = np.arange(len(hidden_dim))

        # Display both the input sequence length and actual pattern length.
        pat = pi
        if results[out_dir]['masked'] != -1:
            pat = results[out_dir]['masked']
        plt.plot(x, hidden_dim, label='i = %i, p = %i'%(pi, pat),color=color[i])
        plt.fill_between(x, hidden_dim-hidden_dim_std, \
            hidden_dim+hidden_dim_std, alpha=0.1, color=color[i])
    plt.xlabel('ts')
    plt.ylabel('accuracy %')

    plt.legend()
    suffix = ''
    if task_id != -1:
        suffix = '_task_%i'%task_id
    if seed_number is not None:
        suffix += '_seed%i'%seed_number
    plt.savefig(os.path.join(path, 'acc_per_ts%s.pdf'%suffix))


def get_bptt_steps(permutation, input_len):
    """Get the number of BPTT steps for permute time tasks.

    For tasks where only time is permuted, this function returns the number
    of steps used in BPTT to compute the loss for each timestep in the output.
    For example for the following input sequence:

        ``A B C D E``

    and a permutation leading the following output sequence:

        ``D C E A B``

    The number of BPTT steps is (taking into account the stop flag):

        ``3 5 4 9 9``

    Args:
        permutation (list): The permutation used inside the CopyTask handler.
        input_len (int): The length of the patterns.

    Returns:
        (list): The number of BPTT steps for each output timesteps.

    """
    seq_width = int(len(permutation)/input_len)
    permutation_rsh = permutation.reshape((input_len, seq_width))[:, 0]

    # The following gives us the indices of where each timestep
    # in the original pattern has been moved to.
    permutation_idx = list(np.argsort(permutation_rsh))

    # This gives the value of the timestep that is represented in
    # each output timestep:
    # example: permutation_idx = 4 0 2 3 1 gives
    #          new_loc_idx = 1 4 2 3 0
    new_loc_idx = []
    for iii in range(input_len):
        new_loc_idx.append(permutation_idx.index(iii))

    # Now we compute the number of BPTT steps for each timestep.
    bptt_steps = -(new_loc_idx - np.arange(input_len)) + input_len + 1

    assert np.mean(bptt_steps) == input_len + 1

    return bptt_steps


def plot_accuracy_vs_bptt_steps(results, seed_groups, task_id=-1, path='',
        seed_number=None):
    """Plot the accuracy per timesteps vs. the number of BPTT steps.

    Note that this analyses make sense for the case of permutations in time, so
    we only consider that case.

    Args:
        results (dict): The summary dictionary with the results of the runs.
        seed_groups (dict): The grouping of runs from different random seeds.
        task_id (int): The number of tasks trained to be taken into account.
            By default the analysis is done when all tasks have been trained on.
        path (str): The path where to save the figures.
        seed_number (int): The number of the seed iteration to be used.

    """
    num_runs = len(seed_groups.keys())
    num_seeds = len(seed_groups[list(seed_groups.keys())[0]])

    cmapp = plt.get_cmap('Oranges', num_runs)
    color = [cmapp(i/num_runs) for i in range(num_runs)]

    # Create arrays gathering the results.
    p = np.array([key for key in seed_groups.keys()])
    p = p[np.argsort(p)]

    seq_length = int(np.max(p)*2 + 1)
    flag_ts = int(np.floor(seq_length/2))

    if seed_number is None:
        seed_range = range(num_seeds)
    else:
        seed_range = [seed_number]

    plt.figure()
    for i, pi in enumerate(p):
        hidden_dim = []
        for ii in seed_range:
            out_dir = seed_groups[pi][ii]
            hidden_dim.append(results[out_dir]['accs_per_ts'][task_id])
            if ii == 0:

                # Now we compute the number of BPTT steps for each timestep.
                permutation = results[out_dir]['permutation'][task_id]
                bptt_steps = get_bptt_steps(permutation, pi)

            else:
                assert (results[out_dir]['permutation'][task_id] == \
                    permutation).all()

        hidden_dim_std = np.std(np.array(hidden_dim), axis=0)
        hidden_dim = np.mean(np.array(hidden_dim), axis=0)    
        
        # Make sure ts 0 corresponds to stop flag.
        x = np.arange(len(hidden_dim))

        # Display both the input sequence length and actual pattern length.
        pat = pi
        if results[out_dir]['masked'] != -1:
            pat = results[out_dir]['masked']
        plt.scatter(bptt_steps, hidden_dim, label='i = %i, p = %i'%(pi, pat), \
            color=color[i])

    plt.xlabel('bptt steps')
    plt.ylabel('accuracy %')

    plt.legend()
    suffix = ''
    if task_id != -1:
        suffix = '_task_%i'%task_id
    if seed_number is not None:
        suffix += '_seed%i'%seed_number
    plt.savefig(os.path.join(path, 'acc_vs_bptt_steps%s.pdf'%suffix))


def str_to_list(string):
    """Convert a string list into an actual list.

    Args:
        string (str): The list in string format.

    Returns:
        (list): The corresponding list.

    """
    if string[0] == '[' and string[-1] == ']':
        string = string[1:-2]

    li = list(string.split(" ")) 
    for i, item in enumerate(li):
        li[i] = float(item)

    return li


def group_runs(results):
    """Group runs that are identical, except for the random seed.

    This grouping is done in a simplistic way, simply grouping together the
    runs that have the same complexity measure, assuming all the other
    parameters will also be the same.

    Furthermore, we ensure that for all complexity measure values, the number
    of runs is the same (such that we always average across the same number of
    random seeds).

    Args:
        results (dict): The summarized results of the runs.

    Returns:
        (dict): The sorting of the runs according to the complexity measure.

    """

    # Loop over all runs and assign to correct p.
    seed_groups = {}
    for out_dir in results.keys():
        p = results[out_dir]['complexity_measure']
        if p not in seed_groups:
            seed_groups[p] = [out_dir]
        else:
            seed_groups[p].append(out_dir)

    # Make sure that all the levels of complexity have the same number of runs.
    min_num_seeds = np.min([len(seed_groups[p]) for p in seed_groups.keys()])
    print('\nThe analysis was done with %i '%min_num_seeds + \
        'seeds for each type of run.')
    for p in seed_groups.keys():
        seed_groups[p] = seed_groups[p][:min_num_seeds]

    return seed_groups


def analyse_single_run(out_dir, device, writer, logger, analysis_kwd, 
        get_loss_func, accuracy_func, generate_tasks_func,
        redo_analyses=False, do_kernel_pca=False):
    """Analyse the hidden dimensionality for an individual run.

    Args:
        out_dir (str): The path to the output directory.
        device: The device.
        writer: The tensorboard writer.
        logger: The logger.
        analysis_kwd (dict): The dictionary containing important keywords for
            the current analysis.
        get_loss_func (func): A handler to generate the loss function.
        accuracy_func (func): A handler to the accuracy function.
        generate_tasks_func (func): A handler to a datahandler generator.
        redo_analyses (boolean, optional): If True, analyses will be redone 
            even if they had been stored previously.
        do_kernel_pca (bool, optional): If True, kernel PCA will also be used
            to copmute the number of hidden dimensions.

    Returns:
        (tuple): Tuple containing:

        - **results**: The dictionary of results for the current run.
        - **settings**: The dictionary with the values of the parameters that
          are specified in `analysis_kwd['fixed_params']`.

    """
    dirname = os.path.basename(out_dir)

    ### Initialize the results of the run.
    results = { 'complexity_measure':[],
                'complexity_measure_name':[],
                'num_tasks':[],
                'final_acc':[],
                'final_acc_std':[],
                'during_acc':[],
                'accs_per_ts':[],
                'mean_fisher':[],
                'std_fisher':[],
                'median_fisher':[],
                'num_hidden_dim':[],           # PCA-based
                'expl_var':[],                 # PCA-based
                'knum_hidden_dim':[],          # kernel PCA-based
                'num_hidden_dim_per_ts':[],    # PCA-based
                'masked':[],
                'permutation':[]
               }

    ### Prepare the data and the networks.
    # Load the config
    if not os.path.exists(out_dir):
        raise ValueError('The directory "%s" does not exist.'%out_dir)
    with open(out_dir + "/config.pickle", "rb") as f:
        config = pickle.load(f)

    # Do some sanity checks in the parameters.
    assert config.use_ewc
    for key, value in analysis_kwd['forced_params']:
        assert getattr(config, key) == value

    # Overwrite the directory it it's not the same as the original.
    if config.out_dir != out_dir:
        config.out_dir = out_dir

    # Check for old command line arguments and make compatible with new version.
    config = train_args_sequential.update_cli_args(config)

    # Ensure all runs have comparable properties
    if 'num_tasks' not in analysis_kwd['fixed_params']:
        analysis_kwd['fixed_params'].append('num_tasks')
    settings = {}
    for key in analysis_kwd['fixed_params']:
        settings[key] = getattr(config, key)

    if os.path.exists(out_dir + "/pca_results.pickle") and \
            not redo_analyses:
        with open(out_dir + "/pca_results.pickle", "rb") as f:
            results = pickle.load(f)
        print('PCA analyses have been done and stored previously, ' +
            'loading them.')
    else:
        # Populate the dictionary.
        results['complexity_measure'] = getattr(config, \
            analysis_kwd['complexity_measure'])
        results['complexity_measure_name'] = \
            analysis_kwd['complexity_measure_name']
        results['masked'] = config.pat_len
        results['num_tasks'] = config.num_tasks
        results['final_acc'] = []
        results['final_acc_std'] = []
        results['during_acc'] = []
        results['during_acc_std'] = []
        results['accs_per_ts'] = []
        results['mean_fisher'] = []
        results['std_fisher'] = []
        results['median_fisher'] = []
        results['num_hidden_dim'] = []
        results['expl_var'] = []
        results['knum_hidden_dim'] = []
        results['num_hidden_dim_per_ts'] = []
        results['permutation'] =[]

        # Define functions.
        task_loss_func = get_loss_func(config, device, logger)
        accuracy_func = accuracy_func

        # Generate datahandlers
        dhandlers = generate_tasks_func(config, writer)
        config.show_plots = True
        plc.visualise_data(dhandlers, config, device)

        # Generate the network
        target_net, hnet, _ = stu.generate_networks(config, dhandlers, \
            device)

        for task_id in range(config.num_tasks):
            results['permutation'].append(dhandlers[task_id].permutation)

            # Load the checkpointed model for the corresponding task.
            mnet, hnet = load_models(out_dir, device, logger, target_net, hnet, 
                task_id=task_id)

            # Gather fisher values.
            diag_hh_fisher = get_hh_fisher_estimates(mnet)
            results['mean_fisher'].append(np.mean(diag_hh_fisher))
            results['std_fisher'].append(np.std(diag_hh_fisher))
            results['median_fisher'].append(np.median(diag_hh_fisher))

            ### Obtain hidden activations and performances.
            _, accs, accs_per_ts = test(dhandlers, device, config, logger, \
                writer, mnet, hnet, store_activations=True, \
                accuracy_func=accuracy_func, task_loss_func=task_loss_func, 
                num_trained=task_id, return_acc_per_ts=True)
            # We only measure the final accuracy up to the current task, since
            # we are simulating a continual learning setting with less tasks.

            results['final_acc'].append(np.mean(accs[:task_id+1]))
            results['final_acc_std'].append(np.std(accs[:task_id+1]))
            results['during_acc'].append(accs[task_id])
            results['accs_per_ts'].append(accs_per_ts[0])

            # Load internal hidden activations.
            act = get_int_activations(out_dir)
            n_hidden = np.sum(misc.str_to_ints(config.rnn_arch))
            assert act.shape[-1] == n_hidden

            # Perform PCA on all the hidden activations.
            num_dim, expl_var, knum_dim = get_num_dimensions(act, \
                do_kernel_pca=do_kernel_pca)
            results['num_hidden_dim'].append(num_dim)
            results['expl_var'].append(expl_var)
            results['knum_hidden_dim'].append(knum_dim)

            # Perform PCA on hidden activations per timestep.
            results['num_hidden_dim_per_ts'].append(\
                get_num_dimensions_per_ts(act))

            # Get hidden dimensionality on current task using the final model.
            # FIXME here we overwrite the file "int_activations.pickle" that
            # was generated when testing the model of the current task.
            os.remove(os.path.join(out_dir, 'int_activations.pickle'))
            mnet, hnet = load_models(out_dir, device, logger, \
                target_net, hnet)
            _ = test(dhandlers, device, config, logger, writer, mnet, 
                hnet, store_activations=True, \
                accuracy_func=accuracy_func, task_loss_func=task_loss_func, 
                num_trained=task_id, return_acc_per_ts=True)
            act = get_int_activations(out_dir)
            if task_id == 0:
                all_task_activations = act
            else:
                # Concatenate activations across sample dimension, i.e. hidden
                # actuvations from different tasks will be considered simply
                # as different samples.
                all_task_activations = torch.cat((all_task_activations, act), 1)

        # Compute number of dimensions for activations pulled from all tasks.
        if config.num_tasks > 1:
            num_dim, expl_var, knum_dim = get_num_dimensions(\
                all_task_activations, do_kernel_pca=do_kernel_pca)
            results['num_hidden_dim_all_tasks'] = num_dim
            results['expl_var_all_tasks'] = expl_var
            results['knum_hidden_dim_all_tasks'] = knum_dim

        # Store pickle results.
        with open(out_dir + '/pca_results.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results, settings


def run(config, analysis_kwd, *args):
    """Run the script.

    Args:
        config: The default config for the current dataset.
        analysis_kwd (dict): A dictionary with settings for current analysis.
    """

    misc.configure_matplotlib_params(font_size=15)

    ### Parse the command-line arguments.
    parser = argparse.ArgumentParser(description= \
        'Studying the performance of EWC for different complexity settings.')
    parser.add_argument('out_dir', type=str, default='./out/hyperparam_search',
                        help='The output directory of the runs. ' +
                             'Default: %(default)s.')
    parser.add_argument('--redo_analyses', action='store_true',
                        help='If enabled, all analyses will be done even ' +
                             'if some previous results had been stored and ' +
                             'could have been loaded.')
    parser.add_argument('--do_kernel_pca', action='store_true',
                        help='If enabled, kernel PCA will also be used to ' +
                             'compute the number of hidden dimensions.')
    cmd_args = parser.parse_args()

    ### Set up environment using some general command line arguments.
    device, writer, logger = sutils.setup_environment(config)

    # Define directory where to store the results of all current analyses.
    results_dir = os.path.join(cmd_args.out_dir, 'pca_analyses')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    ### Check if the current directory corresponds to a single run or not.
    # The name of folders has recently been changed. Allow old and new namings.
    out_dirs = glob.glob(cmd_args.out_dir + '/20*')
    out_dirs.extend(glob.glob(cmd_args.out_dir + '/sim*'))
    if len(out_dirs)==0:
        out_dirs = [cmd_args.out_dir]

    ### Store the results of the different runs in a same dictionary.
    results = {}
    for i, out_dir in enumerate(out_dirs):
        results[out_dir], settings = analyse_single_run(out_dir, device, 
            writer, logger, analysis_kwd, *args,    
            redo_analyses=cmd_args.redo_analyses, 
            do_kernel_pca=cmd_args.do_kernel_pca)

        # Ensure all runs have comparable properties
        if i == 0:
            common_settings = settings.copy()
        for key in settings.keys():
            assert settings[key] == common_settings[key]

    ### Check if there are identical runs with different random seeds.
    seed_groups = group_runs(results)

    # Store pickle results.
    with open(os.path.join(results_dir, 'results.pickle'), 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(results_dir, 'seed_groups.pickle'), 'wb') as handle:
        pickle.dump(seed_groups, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Display across tasks dimensionality.
    if common_settings['num_tasks'] > 1:
        num_hidden_dim_sum_tasks = []
        num_hidden_dim_across_tasks = []
        for i, out_dir in enumerate(out_dirs):
            num_hidden_dim_sum_tasks.append(\
                np.sum(results[out_dir]['num_hidden_dim']))
            num_hidden_dim_across_tasks.append(\
                results[out_dir]['num_hidden_dim_all_tasks'])

        print('Sum of hidden dimensions in all tasks: %.2f +- %.2f'%\
            (np.mean(num_hidden_dim_sum_tasks), \
                np.std(num_hidden_dim_sum_tasks)))
        print('Hidden dimension across all tasks: %.2f +- %.2f'%\
            (np.mean(num_hidden_dim_across_tasks), \
                np.std(num_hidden_dim_across_tasks)))

    ### Plot.
    for task_id in range(common_settings['num_tasks']):
        plot_dim_per_ts_all_runs(results, seed_groups, task_id=task_id, \
            path=results_dir)

        plot_accuracy_per_ts(results, seed_groups, task_id=task_id, 
            path=results_dir)

        if settings['permute_time'] and not settings['permute_width'] and not \
                settings['permute_xor']:
            plot_accuracy_vs_bptt_steps(results, seed_groups, task_id=task_id, 
                path=results_dir)

    # Only generate the rest of the plots if we have several runs.
    if not len(out_dirs)==1:
        plot_across_runs(results, seed_groups, path=results_dir, 
            do_kernel_pca=cmd_args.do_kernel_pca)


if __name__=='__main__':
    pass
