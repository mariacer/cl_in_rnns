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
# @title           :sequential/ht_analyses/state_space_plotting.py
# @author          :mc
# @contact         :mariacer@ethz.ch
# @created         :27/06/2020
# @version         :1.0
# @python_version  :3.6.8
"""
Hidden state analyses plotting utils
------------------------------------

This script gathers plotting functions for the state space analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D
import math

from sequential.ht_analyses.pca_utils import get_num_dimensions

def plot_importance_vs_task(results, seed_groups, task_id=-1, path='',
                            log_scale=False):
    """Plot mean importance values against number of tasks.

    Args:
        results (dict): The summary dictionary with the results of the runs.
        seed_groups (dict): The grouping of runs from different random seeds.
        task_id (int): The number of tasks trained to be taken into account.
            By default the analysis is done when all tasks have been trained on.
        path (str): The path where to save the figures.
        log_scale (boolean): Whether to use a log scale for the magnitudes of 
            the importance values.
    """
    num_tasks = results[list(results.keys())[0]]['num_tasks']
    if task_id == -1:
        # The importance values are not computed for the last task, so we don't
        # need to take that one into account.
        # FIXME We might wanna change that. E.g., we could always compute the
        # importance values of the last task if models are stored.
        task_id = num_tasks-1

    p = np.array([key for key in seed_groups.keys()])
    p = np.sort(p)

    plt.figure()
    plot_importance_ho = True
    for nr, pi in enumerate(p):

        imp_values = []
        imp_values_ho = []
        for out_dir in seed_groups[pi]:
            imp_values.append(results[out_dir]['mean_importance'][:task_id])
            if 'mean_importance_ho' in results[out_dir].keys():
                imp_values_ho.append(\
                    results[out_dir]['mean_importance_ho'][:task_id])
            else:
                plot_importance_ho = False

        imp_values = np.array(imp_values)

        # Normalize by largest value.
        imp_values /= np.max(np.mean(imp_values, axis=0))
        imp_values_std = np.std(imp_values, axis=0)
        imp_values_mean = np.mean(imp_values, axis=0)
            
        plt.plot(np.arange(num_tasks-1)[:task_id], imp_values_mean, color='k',
            label='hh')
        plt.fill_between(np.arange(num_tasks-1)[:task_id],
                         imp_values_mean-imp_values_std, 
                         imp_values_mean+imp_values_std,
                         alpha=0.1, color='k')

        if plot_importance_ho:
            imp_values_ho = np.array(imp_values_ho)
            if not math.isnan(np.array(imp_values_ho).mean()):
                imp_values_ho /= np.max(np.mean(imp_values_ho, axis=0))
                imp_values_std_ho = np.std(imp_values_ho, axis=0)
                imp_values_mean_ho = np.mean(imp_values_ho, axis=0)
                plt.plot(np.arange(num_tasks-1)[:task_id], imp_values_mean_ho,
                         color='r', label='ho')
                plt.fill_between(np.arange(num_tasks-1)[:task_id],
                    imp_values_mean_ho-imp_values_std_ho, 
                    imp_values_mean_ho+imp_values_std_ho,
                    alpha=0.1, color='k')

    plt.xlabel('num task')
    plt.ylabel('norm. mean importance')
    plt.legend()
    if log_scale:
        plt.yscale('log')
    plt.savefig(os.path.join(path, 'importance_vs_task.pdf'))


def plot_dimension_vs_task(results, seed_groups, label=None, color='b',
        task_id=-1, path='', kernel=False, onto_task_1=False, cum=False,
        internal=True):
    """Plot hidden dimensionality vs. task.

    If the results and seed groups of a second experiment are provided, both
    will be plotted together for comparison.

    If ``onto_task_1`` is ``True` we plot ``results['expl_var_across_tasks']``.
    Else we plot ``results['expl_var']``.
    If ``kernel`` is ``True``, then we plot the equivalent values for the kernel
    case.

    Args:
        results (dict or list): The summary dictionary with the results of the
            runs. If list, it contains the dictionary from several runs. In
            this case, ``seed_groups``, ``label`` and ``color`` are all expected
            to be lists with equal length.
        seed_groups (dict or list): The grouping of runs from different random
            seeds.
        label (str or list): The label of the experiment.
        color (str or list): The color for the plots.
        task_id (int): The number of tasks trained to be taken into account.
            By default the analysis is done when all tasks have been trained on.
        path (str): The path where to save the figures.
        kernel (bool): If ``True``, prints the kernel PCA results.
        onto_task_1 (bool, optional): If ``True``, the explained variance when
            projecting into the pcs of the first task is plotted. Else, the
            explained variance when projecting onto the tasks native pcs
            are plotted.
        cum (boolean, optional): If ``True``, the pcs to project onto are
            computed cumulatively using data from all previously learned tasks.
            Else, only task 1 is considered.
        internal (bool, optional): If ``True``, the internal recurrent
            activations of the Elman layer will be considered. Else, the output
            recurrent activations are considered.
    """
    if isinstance(results, list):
        assert len(results) == len(seed_groups) == len(label) == len(color)
    else:
        results = [results]
        seed_groups = [seed_groups]
        label = [label]
        color = [color]

    num_experiments = len(results)
    num_tasks = results[0][list(results[0].keys())[0]]['num_tasks']
    n_pcs_considered = \
        results[0][list(results[0].keys())[0]]['n_pcs_considered_across_tasks']

    # FIXME Why are we enforcing this? We don't make use of Fisher values in
    # this function, so we could use `task_id = num_tasks`?
    # Also, it is checked in subfunction correctly, but already overwritten
    # here.
    if task_id == -1:
        task_id = num_tasks-1

    assert len(seed_groups[0].keys()) == 1

    def plot_experiment(results, seed_groups, task_id=-1, color='b', label=None,
            plot_all_tasks=False, n_pcs_considered=None, kernel=False, cum=False,
            internal=True):
        """Plot the explained variance in a single experiment."""
        key = 'expl_var'
        if kernel:
            key = 'kexpl_var'
        if onto_task_1:
            key += '_accross_tasks'
        if cum:
            key += '_cum'
        if not internal:
            key += '_yt'

        num_hidden = len(results[list(results.keys())[0]][key][task_id])
        num_tasks = results[list(results.keys())[0]]['num_tasks']
        pi = list(seed_groups.keys())[0]

        if task_id == -1:
            task_id = num_tasks
        task_range = [task_id]
        if plot_all_tasks:
            task_range = range(num_tasks)

        if plot_all_tasks:
            cmapp = plt.get_cmap('Oranges', len(task_range)+1)
            color = [cmapp(i/len(task_range)) \
                     for i in range(1, len(task_range)+1)]
            label = ['task %i'%(i+1) for i in task_range]
            assert len(color) == len(label)
        else:
            color = [color]
            label = [label]

        x_range = np.arange(num_hidden)
        if n_pcs_considered is not None:
            x_range = n_pcs_considered
        for i, task in enumerate(task_range):
            expl_variance = []
            for out_dir in seed_groups[pi]:
                expl_variance_seed = results[out_dir][key][task]
                if onto_task_1:
                    expl_variance.append(expl_variance_seed)
                else: 
                    expl_variance_od = []
                    for n_pcs in n_pcs_considered:
                        expl_variance_od.append(expl_variance_seed[n_pcs])
                    expl_variance.append(expl_variance_od)
            expl_variance = np.array(expl_variance)
            expl_variance_mean = np.mean(expl_variance, axis=0)
            expl_variance_std = np.std(expl_variance, axis=0)
            plt.plot(x_range, expl_variance_mean, color=color[i],
                     label=label[i])
            plt.fill_between(x_range, \
                expl_variance_mean-expl_variance_std,
                expl_variance_mean+expl_variance_std, alpha=0.1, color=color[i])

    plt.figure()
    if num_experiments == 1:
        plot_experiment(results[0], seed_groups[0], task_id=task_id,
            color=color[0], label=label[0], plot_all_tasks=True,
            n_pcs_considered=n_pcs_considered, kernel=kernel, cum=cum,
            internal=internal)
    if num_experiments > 1:
        for ne in range(num_experiments):
            plot_experiment(results[ne], seed_groups[ne], task_id=task_id,
                color=color[ne], label=label[ne], internal=internal,
                n_pcs_considered=n_pcs_considered, kernel=kernel, cum=cum)
    plt.legend()
    xlabel = 'number of pcs'
    if onto_task_1:
        xlabel = 'number of task 1 pcs'
    plt.xlabel(xlabel)
    plt.ylabel('expl. variance (%)')
    suffix = ''
    if kernel:
        suffix = '_kernel'
    type_plot = '_vs_task'
    if onto_task_1:
        type_plot = '_onto_task1'
    if cum:
        suffix += '_cum'
    if not internal:
        suffix += '_yt'
    plt.savefig(os.path.join(path, 'dimension%s%s.pdf'%(type_plot, suffix)))
    plt.close()

def plot_supervised_dimension_vs_task(results, seed_groups, path='',
                                      key='loss', stop_bit=False,
                                      for_publication=False):
    """Plot the loss or accuracy as a function of the number of supervised
    dimensions.

    Args: 
        (....): See docstring of function :func:`plot_dimension_vs_task`.
        key (str, optional): Whether to plot the `loss` or `accu`.
        stop_bit (str, optional): If ``True``, the results obtained when looking
            at the stop bit only will be plotted (``accu_n_dim_sup_at_stop``).
            Else, the results obtained when pulling all timesteps together will
            be plotted (``accu_n_dim_supervised``).
        for_publication (bool, optional): If True, the plotting settings
            for publication figures will be used.
    """
    if for_publication:
        fig_size = [1.5 * 1.7, 1.5 * 4.8*1.7/6.4]
        from sequential.plotting_sequential import configure_matplotlib_params
        configure_matplotlib_params(fig_size=fig_size)

    num_experiments = len(seed_groups.keys())
    num_tasks = results[list(results.keys())[0]]['num_tasks']

    def plot_experiment(results, seed_groups, task_id=-1, plot_all_tasks=False,
                        key='loss', num_tasks=1, num_experiments=1):
        """Plot the explained variance in a single experiment."""
        if stop_bit:
            key = '%s_n_dim_sup_at_stop' % key
        else:
            key = '%s_n_dim_supervised' % key

        num_hidden = len(results[list(results.keys())[0]]['expl_var'][task_id])
        p = np.sort(list(seed_groups.keys()))
        num_runs = len(seed_groups[p[0]])

        # Currently, we can only plot either for many tasks in one experiment,
        # or a single task in many experiments.
        assert num_tasks == 1 or num_experiments == 1

        if task_id == -1:
            task_id = num_tasks
        task_range = [task_id]
        if plot_all_tasks:
            task_range = range(num_tasks)

        if num_experiments == 1:
            plot_ids = task_range
        elif num_tasks == 1:
            plot_ids = range(num_experiments)

        if plot_all_tasks:
            cmapp = plt.get_cmap('Oranges', len(plot_ids)+1)
            if for_publication and num_tasks > 1:
                cmapp = plt.get_cmap('Blues', len(plot_ids)+1)
            color = [cmapp(i/len(plot_ids)) \
                     for i in range(1, len(plot_ids)+1)]
            if num_tasks == 1:
                label = ['p = %i'%p[i] for i in plot_ids]
            elif num_experiments == 1:
                label = ['task %i'%(i+1) for i in plot_ids]
            assert len(color) == len(label)
        else:
            color = ['b']
            label = [None]

        x_range = np.arange(num_hidden)
        for i, task in enumerate(plot_ids):
            if num_experiments == 1:
                loss_vs_dim = np.nan*np.zeros((num_tasks, num_hidden))
                for ii, out_dir in enumerate(seed_groups[p[0]]):
                    aux = results[out_dir][key][task]
                    loss_vs_dim[ii,:len(aux)] = aux
            elif num_tasks == 1:
                num_runs = len(seed_groups[p[0]])
                loss_vs_dim = np.nan*np.zeros((num_runs, num_hidden))
                for ii, out_dir in enumerate(seed_groups[p[i]]):
                    aux = results[out_dir][key][0]
                    loss_vs_dim[ii,:len(aux)] = aux
            loss_vs_dim_mean = np.nanmean(loss_vs_dim, axis=0)
            loss_vs_dim_std = np.nanstd(loss_vs_dim, axis=0)
            n_dim = num_hidden
            if for_publication:
                n_dim = 100
            plt.plot(range(n_dim), loss_vs_dim_mean[:n_dim], color=color[i],
                     label=label[i])
            plt.fill_between(range(n_dim), \
                loss_vs_dim_mean[:n_dim]-loss_vs_dim_std[:n_dim],
                loss_vs_dim_mean[:n_dim]+loss_vs_dim_std[:n_dim], alpha=0.1, 
                    color=color[i])

    plt.figure()
    plot_experiment(results, seed_groups, plot_all_tasks=True, key=key,
        num_tasks=num_tasks, num_experiments=num_experiments)
    plt.legend()
    plt.xlabel('number of supervised dims.')
    if key == 'loss':
        plt.ylabel('loss')
    elif key == 'accu':
        plt.ylabel('accuracy (%)')
    title = '%s_vs_supervised_dimensions'%key
    if stop_bit:
        title += '_at_stop'
    plt.savefig(os.path.join(path, title + '.pdf'))
    plt.close()


def plot_across_runs(results, seed_groups, task_id=-1, path='',
                     log_scale=False, do_kernel_pca=False, n_pcs=10,
                     p_var=0.75):
    """Plot results across runs with different sequence lengths.

    Specifically, the following is plotted as a function of the complexity
    measure (e.g., pattern length):

    - the performance
    - the average importance values
    - the dimensionality of the hidden space

    Args:
        results (dict): The summary dictionary with the results of the runs.
        seed_groups (dict): The grouping of runs from different random seeds.
        task_id (int): The number of tasks trained to be taken into account.
            By default the analysis is done when all tasks have been trained on.
        path (str): The path where to save the figures.
        log_scale (bool): Whether to use a log scale for the 
            magnitudes of the importance values.
        do_kernel_pca (bool, optional): If True, kernel PCA will also be used
            to copmute the number of hidden dimensions.
        n_pcs (int): The number of PCs for which to report the explained
            variance.
        p_var (float): The percentage of explained variance that should be
            explained by the number of principal components that this function
            returns.

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

    num_subplots = 4
    if do_kernel_pca:
        num_subplots += 1

    fig, axes = plt.subplots(num_subplots, 1, figsize=(7, num_subplots*2.5))
    for task_id in range(num_tasks):

        # Create arrays gathering the results.
        p = np.zeros(num_runs)
        loss = np.zeros((num_seeds, num_runs))
        acc = np.zeros((num_seeds, num_runs))
        num_hidden_dim = np.zeros((num_seeds, num_runs))
        knum_hidden_dim = np.zeros((num_seeds, num_runs))
        expl_var = np.zeros((num_seeds, num_runs))
        importance = np.zeros((num_seeds, num_runs))
        for i, pi in enumerate(seed_groups.keys()):
            p[i] = pi
            for oi, out_dir in enumerate(seed_groups[pi]):
                loss[oi, i] = results[out_dir]['final_loss'][task_id]
                acc[oi, i] = results[out_dir]['final_acc'][task_id]
                num_hidden_dim[oi, i] = get_num_dimensions(results[out_dir]\
                    ['expl_var'][task_id], p_var=p_var)
                expl_var[oi, i] = results[out_dir]['expl_var'][task_id][n_pcs]
                knum_hidden_dim[oi, i] = get_num_dimensions(results[out_dir]\
                    ['kexpl_var'][task_id], p_var=p_var)
                importance[oi, i] = results[out_dir]['mean_importance'][task_id]
        loss_std = np.std(loss, axis=0)
        loss = np.mean(loss, axis=0)
        acc_std = np.std(acc, axis=0)
        acc = np.mean(acc, axis=0)
        std_importance = np.std(importance, axis=0)
        mean_importance = np.mean(importance, axis=0)
        num_hidden_dim_std = np.std(num_hidden_dim, axis=0)
        num_hidden_dim = np.mean(num_hidden_dim, axis=0)
        expl_var_std = np.std(expl_var, axis=0)
        expl_var = np.mean(expl_var, axis=0)
        knum_hidden_dim_std = np.std(knum_hidden_dim, axis=0)
        knum_hidden_dim = np.mean(knum_hidden_dim, axis=0)

        # Sort the lists by increasing sequence length.
        seq_order = np.argsort(p)
        p = np.array(p[seq_order])
        loss = np.array(loss[seq_order])
        loss_std = np.array(loss_std[seq_order])
        acc = np.array(acc[seq_order])
        acc_std = np.array(acc_std[seq_order])
        num_hidden_dim = np.array(num_hidden_dim[seq_order])
        num_hidden_dim_std = np.array(num_hidden_dim_std[seq_order])
        expl_var = np.array(expl_var[seq_order])
        expl_var_std = np.array(expl_var_std[seq_order])
        knum_hidden_dim = np.array(knum_hidden_dim[seq_order])
        knum_hidden_dim_std = np.array(knum_hidden_dim_std[seq_order])
        mean_importance = np.array(mean_importance[seq_order])
        std_importance = np.array(std_importance[seq_order])

        ### Plot the performance vs. sequence length.
        if np.any(np.isnan(acc)):
            axes[0].plot(p, loss, color=color[task_id])
            axes[0].fill_between(p, loss-loss_std, loss+loss_std, alpha=0.1, \
                color=color[task_id])
        else:
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

        ### Plot the importance weights vs. sequence length.
        axes[num_subplots-1].plot(p, mean_importance, color=color[task_id])
        axes[num_subplots-1].fill_between(p, mean_importance-std_importance, \
            mean_importance+std_importance, alpha=0.1, color=color[task_id])

    if np.any(np.isnan(acc)):
        axes[0].set_ylabel('final loss')
    else:
        axes[0].set_ylabel('% final accuracy')
    axes[0].set_xticks(p)
    axes[1].set_ylabel('num dimensions')
    axes[1].set_xticks(p)
    axes[2].set_ylabel('explained var')
    axes[2].set_xticks(p)
    if do_kernel_pca:
        axes[3].set_ylabel('num dimensions (kPCA)')
        axes[3].set_xticks(p)
    axes[num_subplots-1].set_ylabel('mean importance')
    axes[num_subplots-1].set_xlabel(complexity_measure_name)
    axes[num_subplots-1].set_xticks(p)
    if log_scale:
        axes[num_subplots-1].set_yscale('log')

    # Prepare legend.
    custom_lines = [Line2D([0], [0], color=color[i], lw=4) for i in \
        range(num_tasks)]
    custom_lines_legend = ['%i tasks'%i for i in range(1, num_tasks+1)]
    axes[num_subplots-1].legend(custom_lines, custom_lines_legend, \
        loc='upper left')

    plt.savefig(os.path.join(path, 'complexity_effects.png'))

def plot_per_ts(results, seed_groups, key_name, task_id=-1, path='',
                seed_number=None, kernel=False, internal=True):
    """Plot a given variable (specified by ``key_name``) per timestep.

    We either plot across many tasks, or across many pattern lenghts, depending
    on the number of tasks and the number of runs provided.

    If ``key_name='dimension'`` we plot ``results['expl_var_per_ts']`` or its
    kernel version.
    If ``key_name='accuracy'`` we plot ``results['accuracy_per_ts']`` or its
    kernel version.

    Args:
        results (dict): The summary dictionary with the results of the runs.
        seed_groups (dict): The grouping of runs from different random seeds.
        key_name (str): The data within results dictionary to be plotted.
        task_id (int): The number of tasks trained to be taken into account.
            By default the analysis is done when all tasks have been trained on.
        path (str): The path where to save the figures.
        seed_number (int): The number of the seed iteration to be used.
        kernel (bool): If ``True``, prints the kernel PCA results.
        internal (bool, optional): If True, the internal recurrent activations
            of the Elman layer will be considered. Else, the output recurrent
            activations are considered.
    """
    num_runs = len(seed_groups.keys())
    num_seeds = len(seed_groups[list(seed_groups.keys())[0]])
    num_tasks = results[list(results.keys())[0]]['num_tasks']

    # Depending on `key_name`, fix namings.
    if key_name == 'dimension':
        key = 'expl_var_per_ts'
        if kernel:
            key = 'kexpl_var_per_ts'
        y_label = 'num dimensions'
    elif key_name == 'accuracy':
        key = 'accs_per_ts'
        y_label = 'accuracy %'

    # Create arrays gathering the results.
    p = np.array([key for key in seed_groups.keys()])
    p = p[np.argsort(p)]
    assert len(p) == num_runs

    if seed_number is None:
        seed_range = range(num_seeds)
    else:
        seed_range = [seed_number]

    # Define the iterators depending on whether we have many runs or tasks.
    if num_runs > 1:
        num_plots = num_runs
        if task_id == -1:
            task_id = num_tasks
        tasks = task_id*np.ones(num_runs).astype(int)
    else:
        num_plots = num_tasks
        p = p*np.ones(num_tasks)
        tasks = range(num_tasks)
    assert len(p) == len(tasks) == num_plots

    cmapp = plt.get_cmap('Oranges', num_plots+1)
    color = [cmapp(i/num_plots) for i in range(1,num_plots+1)]

    def plot_curve_per_ts(i, pi, taski, key, internal=True):
        colori = color[i]
        if taski == None:
            key += '_all_tasks'
            colori = 'b'
        if not internal:
            key += '_yt'

        hidden_dim = []
        for ii in seed_range:
            out_dir = seed_groups[pi][ii]
            if key_name == 'accuracy':
                hidden_dim.append(results[out_dir][key][taski])
            elif key_name == 'dimension':
                if taski is None:
                    nh_per_ts = [get_num_dimensions(x) for x in \
                        results[out_dir][key]]
                else:
                    nh_per_ts = [get_num_dimensions(x) for x in \
                        results[out_dir][key][taski]]
                hidden_dim.append(nh_per_ts)
        hidden_dim_std = np.std(np.array(hidden_dim), axis=0)
        hidden_dim = np.mean(np.array(hidden_dim), axis=0)

        # Make sure ts 0 corresponds to stop flag.
        x = np.arange(len(hidden_dim))
        if key_name == 'dimension':
            if 'masked' in results[out_dir].keys() and \
                    results[out_dir]['masked'] != -1:
                if results[out_dir]['pad_after_stop']:
                    x -= results[out_dir]['masked']
                else:
                    x -= pi
            else:
                x -= int((len(hidden_dim))/2)

        # Display both the input sequence length and actual pattern length.
        pat = pi
        if 'masked' in results[out_dir].keys() and \
                results[out_dir]['masked'] != -1:
            pat = results[out_dir]['masked']
        if taski is None:
            label = 'all tasks'
        elif num_runs > 1:
            label = 'i = %i, p = %i'%(pi, pat)
        elif num_tasks > 1:
            label = 'task %i'%(taski+1)
        plt.plot(x, hidden_dim, label=label,color=colori)
        plt.fill_between(x, hidden_dim-hidden_dim_std, \
            hidden_dim+hidden_dim_std, alpha=0.1, color=colori)

    plt.figure()
    for i, (pi, taski) in enumerate(zip(p, tasks)):
        plot_curve_per_ts(i, pi, taski, key, internal=internal)

    # Plot the dimension when pulling hidden activities from all tasks together.
    if key_name == 'dimension' and num_runs == 1:
        plot_curve_per_ts(i, pi, None, key, internal=internal)

    plt.xlabel('ts')
    plt.ylabel(y_label)
    seq_length_lim = np.max(p) + (10 - np.max(p)%10)
    plt.xticks(np.arange(-seq_length_lim, seq_length_lim, 5), \
        fontname="Liberation Sans")
    axes = plt.gca()
    plt.plot([0, 0], axes.get_ylim(), '--k', label='stop flag')
    plt.legend()
    suffix = ''
    if task_id != -1:
        suffix = '_task_%i'%task_id
    else:
        suffix = '_all_tasks'
    if seed_number is not None:
        suffix += '_seed%i'%seed_number
    if kernel:
        suffix += '_kernel'
    if not internal:
        suffix += '_yt'
    plt.savefig(os.path.join(path, '%s_per_ts%s.pdf'%(key_name, suffix)))


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

    cmapp = plt.get_cmap('Oranges', num_runs+1)
    color = [cmapp(i/num_runs) for i in range(1,num_runs+1)]

    # Create arrays gathering the results.
    p = np.array([key for key in seed_groups.keys()])
    p = p[np.argsort(p)]

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

        #hidden_dim_std = np.std(np.array(hidden_dim), axis=0)
        hidden_dim = np.mean(np.array(hidden_dim), axis=0)    

        # Make sure ts 0 corresponds to stop flag.
        #x = np.arange(len(hidden_dim))

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


def print_dimension_vs_task(results, seed_groups, p_var=0.75,
                            do_kernel_pca=False):
    """Display the intrinsic dimensionality of the subtasks.

    Args:
        results (dict): The dictionary with the PCA results.
        seed_groups (dict): The grouping of runs from different random seeds.
        p_var (float): The percentage of explained variance that should be
            explained by the number of principal components that this function
            returns.
        do_kernel_pca (bool, optional): If True, kernel PCA will also be used
            to copmute the number of hidden dimensions.
    """
    for key in seed_groups.keys():
        out_dirs = seed_groups[key]

        num_hidden_dim_sum_tasks = []
        num_hidden_dim_across_tasks = []
        knum_hidden_dim_sum_tasks = []
        knum_hidden_dim_across_tasks = []
        for i, out_dir in enumerate(out_dirs):

            # Compute sum of dimensionality of individual tasks for PCA and kPCA
            nhd = [get_num_dimensions(x, p_var=p_var) for x in \
                results[out_dir]['expl_var']]
            num_hidden_dim_sum_tasks.append(np.sum(nhd))
            # Compute the dimensionality of all tasks pulled together.
            nhd_all_tasks = get_num_dimensions(results[out_dir]\
                ['expl_var_all_tasks'], p_var=p_var)
            num_hidden_dim_across_tasks.append(nhd_all_tasks)

            if do_kernel_pca:
                knhd = [get_num_dimensions(x, p_var=p_var) for x in \
                    results[out_dir]['kexpl_var']]
                knum_hidden_dim_sum_tasks.append(np.sum(knhd))
                knhd_all_tasks = get_num_dimensions(results[out_dir]\
                    ['kexpl_var_all_tasks'], p_var=p_var)
                knum_hidden_dim_across_tasks.append(knhd_all_tasks)

        msg1 = msg2 = ''
        if do_kernel_pca:
            msg1 = '  (kPCA): %.2f +- %.2f'%\
            (np.mean(knum_hidden_dim_sum_tasks), \
                np.std(knum_hidden_dim_sum_tasks))
            msg2 =  '   (kPCA): %.2f +- %.2f'%\
            (np.mean(knum_hidden_dim_across_tasks), \
                np.std(knum_hidden_dim_across_tasks))
        print('\nRun type: %s'%str(key))
        print('Sum of hidden dimensions in all tasks (PCA): %.2f +- %.2f'%\
            (np.mean(num_hidden_dim_sum_tasks), \
                np.std(num_hidden_dim_sum_tasks)) + msg1)
        print('Hidden dimension across all tasks     (PCA): %.2f +- %.2f'%\
            (np.mean(num_hidden_dim_across_tasks), \
                np.std(num_hidden_dim_across_tasks)) + msg2)

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
    # FIXME Masking not considered, right?
    bptt_steps = -(new_loc_idx - np.arange(input_len)) + input_len + 1

    assert np.mean(bptt_steps) == input_len + 1

    return bptt_steps