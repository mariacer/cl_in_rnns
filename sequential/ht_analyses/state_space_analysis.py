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
# @title           :sequential/ht_analyses/state_space_analysis.py
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

This script can be used to either: 
1) study the relationship between the dimensionality of the hidden space, and 
the memory requirements of the task (in which case the provided directory 
should contain runs with varying levels of task difficulty, such as pattern 
length), 
2) study the subspaces used by different subtasks (in which case we expect runs 
to have learned multiple tasks, but we don't need runs with varying levels of 
difficulty).

Specifically, this script analyses the hidden dimensionality of checkpointed
models in a certain run, for which the path needs to be provided.
One can equally provide a path to a folder containing the results of
several individual runs, and then the results will be averaged.

Run as follows:

.. code-block:: 

    python3 state_space_analysis.py path/to/results/folder/

For running these analyses, one needs to have run before the following:

.. code-block::

    python3 hpsearch.py --grid_module=ewc_study_config

Making sure that the following arguments activated for all runs:
``--store_final_models --store_during_models``.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import argparse
import os
import pickle
import glob
import numpy as np
import warnings

import sequential.ht_analyses.state_space_plotting as ssp
from sequential.ht_analyses.pca_utils import get_expl_var, \
    get_expl_var_per_ts, get_expl_var_across_tasks
import sequential.plotting_sequential as plc
import sequential.train_utils_sequential as stu
from sequential.train_sequential import test
import utils.sim_utils as sutils
from utils import logger_config
import utils.ewc_regularizer as ewc
import torch
from utils import misc
from sequential.plotting_sequential import configure_matplotlib_params
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
            checkpointed model to be loaded. If None, the final model is
            loaded.

    Returns:
        (tuple): Tuple containing:

        - **mnet**: The checkpointed main network.
        - **hnet**: The checkpointed hypernetwork.

    """
    # Since the main network doesn't have fisher value attributes, but these
    # have been checkpointed, we manually add them before loading the model.
    # Note that when using online EWC, `task_id` has no effect on the buffer 
    # naming, since the Fisher is accumulated over all tasks (whereas the 
    # original EWC had a Fisher per task).
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

    Note that this includes both weights and biases, if the latter exist.

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


def get_int_activations(out_dir, task_id=-1):
    """Get the internal hidden activations for all trained tasks.

    Given a certain output directory, this function loads the stored hidden
    activations.

    Args:
        out_dir (str): The directory to analyse.
        task_id (int, optional): The id of the task up to which to load the 
            activations.

    Return:
        (tuple): Tuple containing:

        - **activations** (list): The hidden activations of all tasks. 
        - **all_activations** (torch.Tensor): The hidden activations of all
            tasks, concatenated.
    """

    with open(out_dir + "/int_activations.pickle", "rb") as f:
        activations = pickle.load(f)
    if task_id == -1:
        task_id = len(activations)
    else:
        task_id += 1

    # Select only activations up to latest trained task.
    activations = activations[:task_id]

    # Concatenate the activations across all tasks.
    all_activations = torch.tensor(())
    for act in activations:
        all_activations = torch.cat((all_activations, act), dim=1)

    return activations, all_activations


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
        (tuple): Tuple containing:

        - **seed_groups** (dict): The sorting of the runs according to the 
          complexity measure.
        - **min_num_seeds** (int): The number of seeds in each type of run.
        - **num_runs** (int): The number of different runs.

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
    for p in seed_groups.keys():
        seed_groups[p] = seed_groups[p][:min_num_seeds]
    num_runs = len(seed_groups.keys())

    return seed_groups, min_num_seeds, num_runs


def analyse_single_run(out_dir, device, writer, logger, analysis_kwd, 
        get_loss_func, accuracy_func, generate_tasks_func, n_samples=-1,
        redo_analyses=False, do_kernel_pca=False, timesteps_for_analysis=None,
        copy_task=True):
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
        n_samples (int): The number of samples to be used.
        timesteps_for_analysis (str, boolean): The timesteps to be used for the
            PCA analyses.
        copy_task (boolean, optional): Indicates whether we are analysing the
            Copy Task or not.

    Returns:
        (tuple): Tuple containing:

        - **results**: The dictionary of results for the current run.
        - **settings**: The dictionary with the values of the parameters that
          are specified in `analysis_kwd['fixed_params']`.

    """

    ### Prepare the data and the networks.
    # Load the config
    if not os.path.exists(out_dir):
        raise ValueError('The directory "%s" does not exist.'%out_dir)
    with open(out_dir + "/config.pickle", "rb") as f:
        config = pickle.load(f)
    # Overwrite the directory it it's not the same as the original.
    if config.out_dir != out_dir:
        config.out_dir = out_dir
    # Check for old command line arguments and make compatible with new version.
    config = train_args_sequential.update_cli_args(config)

    stop_bit=None
    if copy_task:
        # Get the index of the stop bit.
        stop_bit = getattr(config, analysis_kwd['complexity_measure'])

    ### Sanity checks.
    #  Do some sanity checks in the parameters.
    assert config.use_ewc
    for key, value in analysis_kwd['forced_params']:
        assert getattr(config, key) == value
    # Ensure all runs have comparable properties
    if 'num_tasks' not in analysis_kwd['fixed_params']:
        analysis_kwd['fixed_params'].append('num_tasks')

    ### Create the settings dictionary.
    settings = {}
    for key in analysis_kwd['fixed_params']:
        settings[key] = getattr(config, key)

    ### Load or create the results dictionary.
    if os.path.exists(out_dir + "/pca_results.pickle") and not redo_analyses:
        ### Load existing results.
        with open(out_dir + "/pca_results.pickle", "rb") as f:
            results = pickle.load(f)
        print('PCA analyses have been done and stored previously and reloaded.')
    else:
        ### Prepare the environment.
        # Define functions.
        task_loss_func = get_loss_func(config, device, logger)
        accuracy_func = accuracy_func
        # Generate datahandlers
        dhandlers = generate_tasks_func(config, logger, writer=writer)
        config.show_plots = True
        plc.visualise_data(dhandlers, config, device)
        # Generate the networks
        target_net, hnet, _ = stu.generate_networks(config, dhandlers, device)

        ### Initialize the results dictionary.
        results = {}
        if copy_task:
            results['masked'] = config.pat_len
            results['accs_per_ts'] = []
            results['permutation'] = []
            results['expl_var_per_ts'] = []
            results['kexpl_var_per_ts'] = []
        results['complexity_measure'] = getattr(config, \
            analysis_kwd['complexity_measure'])
        results['complexity_measure_name'] = \
            analysis_kwd['complexity_measure_name']
        results['num_tasks'] = config.num_tasks
        results['final_acc'] = []
        results['mean_fisher'] = []
        results['expl_var'] = []
        results['kexpl_var'] = []

        # Iterate over all tasks and accumulate results in lists within the
        # results dictionary values.
        all_during_act = []
        for task_id in range(config.num_tasks):

            if copy_task:
                results['permutation'].append(dhandlers[task_id].permutation)

            ### Load the checkpointed model for the corresponding task.
            mnet, hnet = load_models(out_dir, device, logger, target_net, hnet, 
                task_id=task_id)
            diag_hh_fisher = get_hh_fisher_estimates(mnet)
            results['mean_fisher'].append(np.mean(diag_hh_fisher))

            ### Obtain hidden activations and performances.
            # We only measure the final accuracy up to the current task, since
            # we are simulating a continual learning setting with less tasks.
            _, accs, accs_per_ts = test(dhandlers, device, config, logger, \
                writer, mnet, hnet, store_activations=True, \
                accuracy_func=accuracy_func, task_loss_func=task_loss_func, 
                num_trained=task_id, return_acc_per_ts=True)
            results['final_acc'].append(np.mean(accs[:task_id+1]))
            if copy_task:
                results['accs_per_ts'].append(accs_per_ts[0])

            ### Load internal hidden activations.
            tasks_act, act = get_int_activations(out_dir, task_id=task_id)
            n_hidden = np.sum(misc.str_to_ints(config.rnn_arch))
            assert act.shape[-1] == n_hidden
            all_during_act.append(act)

            # Perform PCA on all the hidden activations of the current task.
            expl_var, kexpl_var = get_expl_var(act, \
                do_kernel_pca=do_kernel_pca, n_samples=n_samples,
                timesteps=timesteps_for_analysis, stop_bit=stop_bit)
            results['expl_var'].append(expl_var)
            results['kexpl_var'].append(kexpl_var)

            if copy_task:
                # Perform PCA on hidden activations per timestep.
                expl_var_per_ts, kexpl_var_per_ts = get_expl_var_per_ts(act, \
                    n_samples=n_samples, do_kernel_pca=do_kernel_pca)
                results['expl_var_per_ts'].append(expl_var_per_ts)
                results['kexpl_var_per_ts'].append(kexpl_var_per_ts)

        ### Get hidden dimensionality using the final model.
        # Note, here we overwrite the file "int_activations.pickle" that
        # was generated when testing the model of the current task.
        os.remove(os.path.join(out_dir, 'int_activations.pickle'))
        mnet, hnet = load_models(out_dir, device, logger, target_net, hnet)
        _ = test(dhandlers, device, config, logger, writer, mnet, 
            hnet, store_activations=True,
            accuracy_func=accuracy_func, task_loss_func=task_loss_func, 
            num_trained=task_id, return_acc_per_ts=True)
        tasks_act, act = get_int_activations(out_dir, task_id=task_id)

        # Compute number of dimensions for activations from all tasks (here we 
        # use `act` which accumulates the activations across all tasks).
        if config.num_tasks > 1:
            assert len(tasks_act) == config.num_tasks 
            expl_var, k_expl_var = get_expl_var(act, 
                do_kernel_pca=do_kernel_pca, n_samples=n_samples, 
                timesteps=timesteps_for_analysis, stop_bit=stop_bit)
            results['expl_var_all_tasks'] = expl_var
            results['kexpl_var_all_tasks'] = expl_var

            if copy_task:
                expl_var_per_ts, kexpl_var_per_ts = get_expl_var_per_ts(act, \
                    do_kernel_pca=do_kernel_pca, n_samples=n_samples)
                results['expl_var_per_ts_all_tasks'] = expl_var_per_ts
                results['kexpl_var_per_ts_all_tasks'] = kexpl_var_per_ts

        # Compute explained variance when projecting hidden states of other
        # tasks into the pcs of task 1 after learning task 1 only.
        expl_var_accross_tasks, kexpl_var_accross_tasks, n_pcs_considered = \
            get_expl_var_across_tasks(all_during_act, n_samples=n_samples, 
                timesteps=timesteps_for_analysis, stop_bit=stop_bit,
                do_kernel_pca=False)
        results['expl_var_accross_tasks'] = expl_var_accross_tasks
        results['kexpl_var_accross_tasks'] = kexpl_var_accross_tasks
        results['n_pcs_considered_across_tasks'] = n_pcs_considered

        # Store pickle results.
        with open(out_dir + '/pca_results.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results, settings


def run(config, *args, copy_task=True):
    """Run the script.

    Args:
        config: The default config for the current dataset.
        copy_task (boolean, optional): Indicates whether we are analysing the
            Copy Task or not.
    """

    fig_size = [2.8, 2.55]
    configure_matplotlib_params(fig_size=fig_size)

    ### Parse the command-line arguments.
    parser = argparse.ArgumentParser(description= \
        'Studying the dimensionality of the hidden space.')
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
    parser.add_argument('--p_var', type=int, default=0.75,
                        help='The amount of variance that needs to be ' +
                             'explained to determine the number of ' +
                             'dimensions. Default: %(default)s.')
    parser.add_argument('--n_pcs', type=int, default=10,
                        help='The number of principal components to be taken '+
                             'into account for the plots and analyses. ' +
                             'Default: %(default)s.')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='The number of samples to be used for the PCA ' +
                             'analyses in the different settings. This ' +
                             'ensures that comparisons when pulling data ' +
                             'from different tasks are made with an equal ' +
                             'amount of data. Default: %(default)s.')
    parser.add_argument('--timesteps_for_analysis', type=str, default='output',
                        choices=['all', 'input', 'output', 'stop'],
                        help='The timesteps to be used for the PCA analyses. '+
                             'Options are: all timesteps, only during input ' +
                             'presentation, only during output presentation, '+
                             'or only upon stop flag. Default: %(default)s.')
    cmd_args = parser.parse_args()

    if not copy_task and cmd_args.timesteps_for_analysis != 'all':
        warnings.warn('A subset of timesteps for the analysis can only be ' +
            'selected for the Copy Task currently. Using all timesteps.')
        setattr(cmd_args, 'timesteps_for_analysis', 'all')

    # Define directory where to store the results of all current analyses.
    results_dir = os.path.join(cmd_args.out_dir, 'pca_analyses')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # Overwrite the output directory.
    config.out_dir = os.path.join(results_dir, 'sim_files')

    ### Set up environment using some general command line arguments.
    device, writer, logger = sutils.setup_environment(config)

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
            writer, logger, *args, redo_analyses=cmd_args.redo_analyses, 
            n_samples=cmd_args.n_samples, do_kernel_pca=cmd_args.do_kernel_pca,
            timesteps_for_analysis=cmd_args.timesteps_for_analysis,
            copy_task=copy_task)

        # Ensure all runs have comparable properties
        if i == 0:
            common_settings = settings.copy()
        for key in settings.keys():
            assert settings[key] == common_settings[key]
    num_tasks = common_settings['num_tasks']

    ### Check if there are identical runs with different random seeds.
    seed_groups, num_seeds, num_runs = group_runs(results)
    print('\nThe analysis was done with %i seeds '%num_seeds + 
        'for each of the %i different runs.'%num_runs)

    ### Pickle the results.
    with open(os.path.join(results_dir, 'results.pickle'), 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(results_dir, 'seed_groups.pickle'), 'wb') as handle:
        pickle.dump(seed_groups, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ### Plot.
    if copy_task:
        for task_id in range(num_tasks):
            if num_runs > 1:
                ssp.plot_per_ts(results, seed_groups, task_id=task_id, 
                    path=results_dir, key_name='accuracy')
                ssp.plot_per_ts(results, seed_groups, task_id=task_id, 
                    path=results_dir, key_name='dimension')
                if cmd_args.do_kernel_pca:
                    ssp.plot_per_ts(results, seed_groups, key_name='dimension',
                        task_id=task_id, path=results_dir, kernel=True)

            if settings['permute_time'] and not \
                    settings['permute_width'] and not settings['permute_xor']:
                ssp.plot_accuracy_vs_bptt_steps(results, seed_groups, 
                    task_id=task_id, path=results_dir)

    # Make multi-task plots.
    if num_tasks > 1:
        ssp.plot_fisher_vs_task(results, seed_groups, path=results_dir)
        ssp.plot_dimension_vs_task(results, seed_groups, path=results_dir)
        ssp.plot_dimension_vs_task(results, seed_groups, path=results_dir, 
            onto_task_1=True)
        ssp.plot_dimension_vs_task(results, seed_groups, path=results_dir)
        ssp.print_dimension_vs_task(results, seed_groups, p_var=cmd_args.p_var)
        if copy_task:
            if num_runs == 1:
                ssp.plot_per_ts(results, seed_groups, path=results_dir, 
                    key_name='accuracy')
                ssp.plot_per_ts(results, seed_groups, path=results_dir, 
                    key_name='dimension')
                if cmd_args.do_kernel_pca:
                    ssp.plot_per_ts(results, seed_groups, key_name='dimension',
                        path=results_dir, kernel=True)

    # Make multi-run plots.
    if num_runs > 1:
        ssp.plot_across_runs(results, seed_groups, path=results_dir, 
            do_kernel_pca=cmd_args.do_kernel_pca, n_pcs=cmd_args.n_pcs,
            p_var=cmd_args.p_var)

if __name__=='__main__':
    pass