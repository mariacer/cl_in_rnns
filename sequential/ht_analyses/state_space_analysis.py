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

#. study the relationship between the dimensionality of the hidden space, and
   the memory requirements of the task (in which case the provided directory
   should contain runs with varying levels of task difficulty, such as pattern
   length),
#. study the subspaces used by different subtasks (in which case we expect runs
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
from sequential.ht_analyses.supervised_dimred_utils import \
    get_loss_vs_supervised_n_dim
import sequential.plotting_sequential as plc
import sequential.train_utils_sequential as stu
from sequential.train_sequential import test
import utils.sim_utils as sutils
import utils.ewc_regularizer as ewc
import utils.si_regularizer as si
import torch
from utils import misc
from sequential.plotting_sequential import configure_matplotlib_params
from sequential import train_args_sequential

def load_models(out_dir, device, logger, mnet, hnet, wembs=None, task_id=None,
                method='ewc'):
    """Small wrapper to load checkpointed models for mnet and hnet.

    Args:
        out_dir (str): The path to the output directory.
        device: The device.
        logger: The logger.
        mnet: The main network.
        hnet: The hypernetwork.
        wembs (list): Word embeddings.
        task_id (int, optional): The id of the last task to be trained for the
            checkpointed model to be loaded. If None, the final model is
            loaded.
        method (str, optional): The weight-importance method being used.
            Either ``'ewc'`` or ``'si'``.

    Returns:
        (tuple): Tuple containing:

        - **mnet**: The checkpointed main network.
        - **hnet**: The checkpointed hypernetwork.

    """
    # Since the main network doesn't have importance value attributes, but these
    # have been checkpointed, we manually add them before loading the model.
    # Note that when using online EWC, `task_id` has no effect on the buffer
    # naming, since the Fisher is accumulated over all tasks (whereas the
    # original EWC had a Fisher per task).
    online = True
    for i, p in enumerate(mnet.internal_params):
        if method == 'ewc':
            buff_w_name, buff_f_name = ewc._ewc_buffer_names(task_id, i, online)
        else:
            buff_w_name, buff_f_name, _, _ = si._si_buffer_names(i)
        mnet.register_buffer(buff_w_name, torch.empty_like(p))
        mnet.register_buffer(buff_f_name, torch.empty_like(p))

    # Load the models.
    _, mnet, hnet, _ = stu.load_models(out_dir, device, logger, \
        mnet, hnet=hnet, wembs=wembs, return_models=True, task_id=task_id)

    return mnet, hnet


def get_importance_values(mnet, connection_type='hh', method='ewc'):
    """Extract the importance values for the weights.

    For EWC, get the diagonal fisher estimates for the weights.
    For SI, get the omegas.

    Note that this includes both weights and biases, if the latter exist.

    Args:
        mnet: The main network.
        connection_type (str, optional): The type of connections. 'hh' for 
            hidden-to-hidden connections, 'ho' for hidden-to-output connections
            of the Elman layers.
        method (str, optional): The weight-importance method being used.
            Either ``'ewc'`` or ``'si'``.

    Returns:
        (list): The importance values of all requested weights.
    """
    # Get the list of meta information about regularized weights.
    n_cm = mnet._num_context_mod_shapes()
    regged_tnet_meta = mnet._param_shapes_meta[n_cm:]

    importance_values = []
    for i, p in enumerate(regged_tnet_meta):
        if method == 'ewc':
            _, buff_f_name = ewc._ewc_buffer_names(None, i, True)
        elif method == 'si':
            buff_f_name, _, _, _ = si._si_buffer_names(i)
        if 'info' in p.keys() and \
                p['info'] == connection_type:
            importance_value = getattr(mnet, buff_f_name)

            # We flatten the matrices and gather all params in a single list.
            importance_values.extend(torch.flatten(importance_value))

    return importance_values


def get_activations(out_dir, task_id=-1, internal=True, vanilla_rnn=True):
    """Get the hidden activations for all trained tasks.

    Given a certain output directory, this function loads the stored hidden
    activations.
    Note that for the Copy Task, given a during model, the hidden activations
    of all tested tasks are going to be indistinguishable because the inputs
    are the same across tasks. However in other datasets this is not the case.

    If network is an LSTM, :math:`h_t` are considered the external activations
    ("activations.pickle") and :math:`c_t` the internal activations
    ("int_activations.pickle"). For vanilla RNNs, :math:`h_t` are considered
    the internal activations ("int_activations.pickle") and :math:`y_t` are the
    external activations ("activations.pickle").

    Args:
        out_dir (str): The directory to analyse.
        task_id (int, optional): The id of the task up to which to load the 
            activations.
        internal (bool, optional): If ``True``, the internal recurrent
            activations :math:`h_t` of the Elman layer will be loaded. Else,
            the output recurrent activations  :math:`y_t` are returned.

            Note:
                For an LSTM network, the function returns the internal state
                :math:`c_t` if ``internal`` is ``True`` and :math:`h_t`
                otherwise.
        vanilla_rnn (bool, optional): Whether the network used to compute the
            activations is vanilla or LSTM. This will indicate what the hidden
            and internal hidden activations mean.

    Return:
        (tuple): Tuple containing:

        - **activations** (list): The hidden activations of all tasks.
        - **all_activations** (torch.Tensor): The hidden activations of all
            tasks, concatenated.
    """
    filename = ''
    if (vanilla_rnn and internal) or (not vanilla_rnn and not internal):
        filename = 'int_activations'
    elif (vanilla_rnn and not internal) or (not vanilla_rnn and internal):
        filename = 'activations'

    with open(os.path.join(out_dir, "%s.pickle"%filename), "rb") as f:

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
        if len(seed_groups[p]) > min_num_seeds:
            warnings.warn('Complexity value %s has %d seeds. But only ' \
                % (str(p), len(seed_groups[p])) + '%d seeds will be ' \
                % (min_num_seeds) + 'considered.')
        seed_groups[p] = seed_groups[p][:min_num_seeds]
    num_runs = len(seed_groups.keys())

    return seed_groups, min_num_seeds, num_runs


def pca_analysis_single_task(act, results, stop_bit=None, do_kernel_pca=False,
        n_samples=-1, timesteps=None, internal=True,
        do_supervised_dimred=False):
    """Do the PCA analyses on a single task.

    This function will perform PCA on the provided hidden activations, and if we
    are dealing with the copy task it will do this analysis per timestep too.

    Args:
        act (torch.Tensor): The recurrent activations.
        results (dict): The results.
        internal (bool, optional): If ``True``, we are working with the internal
            recurrent dynamics. Else, with the output of the Elman layer. See
            docstring of function :func:`get_activations`.
        do_supervised_dimred (bool, optional): If ``True``, supervised linear
            dimensionality reduction will be used to compute the number of
            task-relevant hidden dimensions.
        (...): See docstring of function :func:`analyse_single_run`.

    Return:
        (dict): The results.
    """
    suffix = ''
    if not internal:
        suffix = '_yt'

    # Perform PCA on all the hidden activations of the current task.
    expl_var, kexpl_var = get_expl_var(act, \
        do_kernel_pca=do_kernel_pca, n_samples=n_samples,
        timesteps=timesteps, stop_bit=stop_bit)
    results['expl_var%s'%suffix].append(expl_var)
    results['kexpl_var%s'%suffix].append(kexpl_var)

    # Perform PCA on hidden activations per timestep.
    expl_var_per_ts, kexpl_var_per_ts = get_expl_var_per_ts(act, \
        n_samples=n_samples, do_kernel_pca=do_kernel_pca)
    results['expl_var_per_ts%s'%suffix].append(expl_var_per_ts)
    results['kexpl_var_per_ts%s'%suffix].append(kexpl_var_per_ts)

    return results


def pca_analysis_all_tasks(act, all_during_act, results, do_kernel_pca=False,
        n_samples=-1, timesteps=None, stop_bit=None, copy_task=False,
        internal=True):
    """Do the PCA analyses with the final model.

    This function performs PCA on the provided hidden activations, and if we are
    dealing with the copy task it will do this analysis per timestep too.

    Args:
        act (torch.Tensor): The recurrent activations from the final model.
        all_during_act (list): The recurrent activations from the during models.
        results (dict): The results.
        (...): See docstring of function `pca_analysis_single_task`.

    Return:
        (dict): The results.
    """
    suffix = ''
    if not internal:
        suffix = '_yt'

    num_tasks = len(all_during_act)

    # Compute number of dimensions for activations from all tasks (here we
    # use `act` which accumulates the activations across all tasks).
    if num_tasks > 1:
        expl_var, k_expl_var = get_expl_var(act,
            do_kernel_pca=do_kernel_pca, n_samples=n_samples,
            timesteps=timesteps, stop_bit=stop_bit)
        results['expl_var_all_tasks%s'%suffix] = expl_var
        results['kexpl_var_all_tasks%s'%suffix] = k_expl_var

        expl_var_per_ts, kexpl_var_per_ts = get_expl_var_per_ts(act, \
            do_kernel_pca=do_kernel_pca, n_samples=n_samples)
        results['expl_var_per_ts_all_tasks%s'%suffix] = expl_var_per_ts
        results['kexpl_var_per_ts_all_tasks%s'%suffix] = kexpl_var_per_ts

    # Compute explained variance when projecting hidden states of other
    # tasks into the pcs of task 1 after learning task 1 only.
    expl_var_accross_tasks, kexpl_var_accross_tasks, n_pcs_considered = \
        get_expl_var_across_tasks(all_during_act, n_samples=n_samples, 
            timesteps=timesteps, stop_bit=stop_bit, do_kernel_pca=False)
    results['expl_var_accross_tasks%s'%suffix] = expl_var_accross_tasks
    results['kexpl_var_accross_tasks%s'%suffix] = kexpl_var_accross_tasks
    results['n_pcs_considered_across_tasks%s'%suffix] = n_pcs_considered

    return results


def analyse_single_run(out_dir, device, writer, logger, analysis_kwd,
        get_loss_func, accuracy_func, generate_tasks_func, n_samples=-1,
        redo_analyses=False, do_kernel_pca=False, do_supervised_dimred=False,
        timesteps_for_analysis=None, copy_task=True, num_tasks=-1,
        sup_dimred_criterion=None, sup_dimred_args={}):
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
        redo_analyses (boolean, optional): If ``True``, analyses will be redone
            even if they had been stored previously.
        do_kernel_pca (bool, optional): If ``True``, kernel PCA will also be
            used to compute the number of hidden dimensions.
        do_supervised_dimred (bool, optional): If ``True``, supervised linear
            dimensionality reduction will be used to compute the number of
            task-relevant hidden dimensions.
        n_samples (int): The number of samples to be used.
        timesteps_for_analysis (str, optional): The timesteps to be used for the
            PCA analyses.
        copy_task (bool, optional): Indicates whether we are analysing the
            Copy Task or not.
        num_tasks (int, optional): The number of tasks to be considered.
        sup_dimred_criterion (int, optional): If provided, this value will 
            be used as stopping criterion when looking for the number of 
            necessary supervised components to describe the hidden activity.
        sup_dimred_args (dict): Optional arguments (e.g., optimization
            arguments) passed to the supervised dimensionality reduction
            :func:`sequential.ht_analyses.supervised_dimred_utils.\
get_loss_vs_supervised_n_dim`.

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
    with open(os.path.join(out_dir, "config.pickle"), "rb") as f:
        config = pickle.load(f)
    # Overwrite the directory it it's not the same as the original.
    if config.out_dir != out_dir:
        config.out_dir = out_dir
    # Check for old command line arguments and make compatible with new version.
    config = train_args_sequential.update_cli_args(config)

    print('Working on output directory "%s".' % out_dir)

    # Overwrite the number of tasks.
    if num_tasks == -1:
        num_tasks = config.num_tasks

    if sup_dimred_criterion == -1:
        sup_dimred_criterion = None

    stop_bit=None
    if copy_task:
        # Get the index of the stop bit.
        #stop_bit = getattr(config, analysis_kwd['complexity_measure'])
        # If we do not enforce the condition below, we have to determine the
        # location of the stop bit on a sample-by-sample basis.
        assert config.input_len_step == 0 and config.input_len_variability == 0
        stop_bit = config.first_task_input_len
        if config.pad_after_stop:
            stop_bit = config.pat_len

    ### Sanity checks.
    # Do some sanity checks in the parameters.
    assert config.use_ewc or config.use_si
    if config.use_ewc:
        method = 'ewc'
    elif config.use_si:
        method = 'si'
    for key, value in analysis_kwd['forced_params']:
        assert getattr(config, key) == value
    # Ensure all runs have comparable properties
    if 'num_tasks' not in analysis_kwd['fixed_params']:
        analysis_kwd['fixed_params'].append('num_tasks')

    ### Create the settings dictionary.
    settings = {}
    for key in analysis_kwd['fixed_params']:
        settings[key] = getattr(config, key)
        if key == 'num_tasks':
            settings[key] = num_tasks

    ### Load or create the results dictionary.
    if os.path.exists(os.path.join(out_dir, "pca_results.pickle")) and \
            not redo_analyses:
        ### Load existing results.
        with open(os.path.join(out_dir, "pca_results.pickle"), "rb") as f:
            results = pickle.load(f)
        print('PCA analyses have been done and stored previously and reloaded.')
        assert num_tasks == -1 or results['num_tasks'] == num_tasks

        if 'mean_fisher' in results:
            results['mean_importance'] = results['mean_fisher']
            results['mean_importance_ho'] = results['mean_fisher_ho']
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
        shared = argparse.Namespace()
        # FIXME might not work for all datasets (e.g., PoS tagging).
        shared.feature_size = dhandlers[0].in_shape[0]
        target_net, hnet, _ = stu.generate_networks(config, shared, dhandlers,
                                                    device)

        ### Initialize the results dictionary.
        results = {}
        if copy_task:
            results['masked'] = config.pat_len
            results['pad_after_stop'] = config.pad_after_stop
            results['accs_per_ts'] = []
            results['permutation'] = []
        results['expl_var_per_ts'] = []
        results['kexpl_var_per_ts'] = []
        results['expl_var_per_ts_yt'] = []
        results['kexpl_var_per_ts_yt'] = []
        results['complexity_measure'] = getattr(config, \
            analysis_kwd['complexity_measure'])
        results['complexity_measure_name'] = \
            analysis_kwd['complexity_measure_name']
        results['num_tasks'] = num_tasks
        results['final_acc'] = []
        results['final_loss'] = []
        results['mean_importance'] = []
        results['mean_importance_ho'] = []
        results['expl_var'] = []
        results['kexpl_var'] = []
        results['expl_var_yt'] = []
        results['kexpl_var_yt'] = []
        if do_supervised_dimred:
            # Note, in the code 'loss_n_dim_supervised' plays, for the
            # supervised dimensionality reduction, the same role as 'expl_var'
            # for the standard PCA analysis, i.e. we store the explained
            # variance (resp. loss) as a function of how many dimensions are
            # taken into account, and then select a threshold for the explained
            # variance (resp. loss) to determine the number of intrinsic
            # dimensions.
            results['loss_n_dim_supervised'] = []
            results['accu_n_dim_supervised'] = []
            if copy_task:
                results['accu_n_dim_sup_at_stop'] = []
                results['loss_n_dim_sup_at_stop'] = []

        # Iterate over all tasks and accumulate results in lists within the
        # results dictionary values.
        all_during_act = []
        all_during_act_yt = []
        for task_id in range(num_tasks):

            if copy_task:
                results['permutation'].append(dhandlers[task_id].permutation)

            ### Load the checkpointed during model for the corresponding task.
            # Note, the return values of the function below are just references
            # to the variables `target_net` and `hnet`, which are modified in-
            # place.
            mnet, hnet = load_models(out_dir, device, logger, target_net, hnet,
                wembs=None, task_id=task_id, method=method)
            # FIXME Should we disentangle weight matrices and bias vectors?
            hh_imp_values = get_importance_values(mnet, connection_type='hh',
                method=method)
            results['mean_importance'].append(np.mean(hh_imp_values))
            ho_imp_values = get_importance_values(mnet, connection_type='ho',
                method=method)
            if ho_imp_values != []:
                results['mean_importance_ho'].append(np.mean(ho_imp_values))
            else:
                results['mean_importance_ho'].append(np.nan)

            ### Obtain hidden activations and performances.
            # We only measure the final accuracy up to the current task, since
            # we are simulating a continual learning setting with less tasks.
            loss, accs, accs_per_ts = test(dhandlers, device, config, None,
                logger, writer, mnet, hnet, store_activations=True, \
                accuracy_func=accuracy_func, task_loss_func=task_loss_func,
                num_trained=task_id, return_acc_per_ts=True)
            results['final_loss'].append(np.mean(loss[:task_id+1]))
            if accs is None:
                results['final_acc'].append(None)
            else:
                results['final_acc'].append(np.mean(accs[:task_id+1]))
            if copy_task:
                results['accs_per_ts'].append(accs_per_ts[task_id])

            ### Load the internal activations.
            tasks_act, act = get_activations(out_dir, task_id=task_id,
                vanilla_rnn=config.use_vanilla_rnn)
            n_hidden = np.sum(misc.str_to_ints(config.rnn_arch))
            assert act.shape[-1] == n_hidden
            all_during_act.append(act)
            tasks_act_yt, act_yt = get_activations(out_dir, task_id=task_id,
                internal=False, vanilla_rnn=config.use_vanilla_rnn)
            all_during_act_yt.append(act_yt)

            ### Do PCA analyses.
            # Do analyses on internal recurrent activations.
            results = pca_analysis_single_task(act, results,
                do_kernel_pca=do_kernel_pca, n_samples=n_samples,
                timesteps=timesteps_for_analysis, stop_bit=stop_bit,
                do_supervised_dimred=do_supervised_dimred)

            # Do analyses on output recurrent activations.
            results = pca_analysis_single_task(act_yt, results,
                do_kernel_pca=do_kernel_pca, n_samples=n_samples,
                timesteps=timesteps_for_analysis, stop_bit=stop_bit,
                internal=False, do_supervised_dimred=do_supervised_dimred)

            if do_supervised_dimred:
                if not copy_task:
                    raise NotImplementedError('TODO need to adapt the ' +
                        'loss computation for tasks other than the Copy Task.')
                # Do supervised dimensionality reduction on during models.
                loss_dim, accu_dim = get_loss_vs_supervised_n_dim(mnet,
                        hnet, task_loss_func, accuracy_func, dhandlers, config,
                        device, task_id=task_id, criterion=sup_dimred_criterion,
                        writer_dir=out_dir, **sup_dimred_args)
                results['loss_n_dim_supervised'].append(loss_dim)
                results['accu_n_dim_supervised'].append(accu_dim)
                if copy_task:
                    loss_dim, accu_dim = get_loss_vs_supervised_n_dim(mnet,
                            hnet, task_loss_func, accuracy_func, dhandlers,
                            config, device, stop_timestep=stop_bit,
                            task_id=task_id, criterion=sup_dimred_criterion,
                            writer_dir=out_dir, **sup_dimred_args)
                    results['loss_n_dim_sup_at_stop'].append(loss_dim)
                    results['accu_n_dim_sup_at_stop'].append(accu_dim)

        ### Get hidden dimensionality using the final model.
        # Note, here we overwrite the files "int_activations.pickle" and
        # "activations.pickle" that were generated when testing the model of
        # the current task.
        os.remove(os.path.join(out_dir, 'int_activations.pickle'))
        os.remove(os.path.join(out_dir, 'activations.pickle'))
        mnet, hnet = load_models(out_dir, device, logger, target_net, hnet,
                                 wembs=None, method=method)
        _ = test(dhandlers, device, config, shared, logger, writer, mnet,
            hnet, store_activations=True,
            accuracy_func=accuracy_func, task_loss_func=task_loss_func,
            num_trained=task_id, return_acc_per_ts=True)

        # Load internal activations.
        tasks_act, act = get_activations(out_dir, task_id=task_id,
            vanilla_rnn=config.use_vanilla_rnn)
        tasks_act_yt, act_yt = get_activations(out_dir, task_id=task_id,
            internal=False, vanilla_rnn=config.use_vanilla_rnn)

        ### Do PCA analyses on final models.
        results = pca_analysis_all_tasks(act, all_during_act, results,
                do_kernel_pca=do_kernel_pca, n_samples=n_samples,
                timesteps=timesteps_for_analysis, stop_bit=stop_bit,
                copy_task=copy_task)
        results = pca_analysis_all_tasks(act_yt, all_during_act_yt, results,
                do_kernel_pca=do_kernel_pca, n_samples=n_samples,
                timesteps=timesteps_for_analysis, stop_bit=stop_bit,
                copy_task=copy_task, internal=False)

        if do_supervised_dimred and len(all_during_act) > 1:
            ### Do supervised dimensionality reduction on final models.
            # Only do if we dealt with more than one task.
            loss_dim, accu_dim = get_loss_vs_supervised_n_dim(mnet, hnet,
                    task_loss_func, accuracy_func, dhandlers, config, device,
                    criterion=sup_dimred_criterion, writer_dir=out_dir,
                    **sup_dimred_args)
            results['loss_n_dim_supervised_all_tasks'] = loss_dim
            results['accu_n_dim_supervised_all_tasks'] = accu_dim

        # Store pickle results.
        with open(os.path.join(out_dir, 'pca_results.pickle'), 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results, settings


def run(config, *args, dataset='copy'):
    """Run the script.

    Args:
        config: The default config for the current dataset.
        dataset (str, optional): The dataset being analysed.
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
    parser.add_argument('--do_supervised_dimred', action='store_true',
                        help='If enabled, supervised dimensionality reduction '+
                             'will be performed to compute the number of '+
                             'task-relevant hidden dimensions.')
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
                        choices=['all', 'input', 'output', 'stop', 'last', \
                                 'stop_plus_one'],
                        help='The timesteps to be used for the PCA analyses. '+
                             'Options are: all timesteps, only during input ' +
                             'presentation, only during output presentation, '+
                             'only upon stop flag, only one timestep after ' +
                             'the flag or only the very last output timestep.'+
                             ' Default: %(default)s.')
    parser.add_argument('--num_tasks', type=int, default=-1,
                        help='Number of tasks to consider. Default: ' +
                             '%(default)s.')
    parser.add_argument('--sup_dimred_criterion', type=int, default=-1,
                        help='Accuracy to be obtained when projecting the ' +
                             'hidden activity into a lower-dimensional ' +
                             'subspace in a supervised fasion. For a value '+
                             'of -1, no criterion is used and so all ' +
                             'possible number of dimensions are explored. '+
                             '%(default)s.')
    parser.add_argument('--sdr_orth_strength', type=float, default=1e3,
                        help='The strength of the orthogonal regularizer ' +
                             'when doing supervised dimensionality reduction ' +
                             '(option "do_supervised_dimred"). ' +
                             'Default: %(default)s.')
    parser.add_argument('--sdr_lr', type=float, default=1e-2,
                        help='The learning rate when doing supervised ' +
                             'dimensionality reduction ' +
                             '(option "do_supervised_dimred"). ' +
                             'Default: %(default)s.')
    parser.add_argument('--sdr_batch_size', type=int, default=64,
                        help='The batch size when doing supervised ' +
                             'dimensionality reduction ' +
                             '(option "do_supervised_dimred"). ' +
                             'Default: %(default)s.')
    parser.add_argument('--sdr_n_iter', type=int, default=100,
                        help='The number of training iterations per ' +
                             'projection column when doing supervised ' +
                             'dimensionality reduction ' +
                             '(option "do_supervised_dimred"). ' +
                             'Default: %(default)s.')
    cmd_args = parser.parse_args()

    if (dataset == 'audioset' and cmd_args.timesteps_for_analysis not in \
            ['all', 'last']) or (dataset == 'seq_smnist' and \
            cmd_args.timesteps_for_analysis != 'all') \
            or (dataset == 'student_teacher' \
            and cmd_args.timesteps_for_analysis != 'all'):
        warnings.warn('A subset of timesteps for the analysis can only be ' +
            'selected for the Copy Task currently. Using all timesteps.')
        # Note that for Sequential SMNIST, the last timestep needs to be 
        # correctly selected, which is not currently implemented.
        setattr(cmd_args, 'timesteps_for_analysis', 'all')

    copy_task = False
    if dataset == 'copy':
        copy_task = True

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
    out_dirs = glob.glob(os.path.join(cmd_args.out_dir, '20*'))
    out_dirs.extend(glob.glob(os.path.join(cmd_args.out_dir, 'sim*')))
    if len(out_dirs)==0:
        out_dirs = [cmd_args.out_dir]

    ### Store the results of the different runs in a same dictionary.
    results = {}
    for i, out_dir in enumerate(out_dirs):
        sdr_args = {
            'lambda_ortho': cmd_args.sdr_orth_strength,
            'lr': cmd_args.sdr_lr,
            'n_iter': cmd_args.sdr_n_iter,
            'batch_size': cmd_args.sdr_batch_size
        }

        results[out_dir], settings = analyse_single_run(out_dir, device,
            writer, logger, *args, redo_analyses=cmd_args.redo_analyses,
            n_samples=cmd_args.n_samples, do_kernel_pca=cmd_args.do_kernel_pca,
            timesteps_for_analysis=cmd_args.timesteps_for_analysis,
            copy_task=copy_task, num_tasks=cmd_args.num_tasks,
            do_supervised_dimred=cmd_args.do_supervised_dimred,
            sup_dimred_criterion=cmd_args.sup_dimred_criterion,
            sup_dimred_args=sdr_args)

        # Ensure all runs have comparable properties
        if i == 0:
            common_settings = settings.copy()
        for key in settings.keys():
            assert settings[key] == common_settings[key]
    num_tasks = common_settings['num_tasks']
    if cmd_args.num_tasks != -1:
        num_tasks = cmd_args.num_tasks
        assert num_tasks <= common_settings['num_tasks']

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
    for task_id in range(num_tasks):
        if num_runs > 1:
            if copy_task:
                ssp.plot_per_ts(results, seed_groups, task_id=task_id,
                    path=results_dir, key_name='accuracy')
            ssp.plot_per_ts(results, seed_groups, task_id=task_id,
                path=results_dir, key_name='dimension')
            ssp.plot_per_ts(results, seed_groups, task_id=task_id,
                path=results_dir, key_name='dimension', internal=False)
            if cmd_args.do_kernel_pca:
                ssp.plot_per_ts(results, seed_groups, key_name='dimension',
                    task_id=task_id, path=results_dir, kernel=True)
                ssp.plot_per_ts(results, seed_groups, key_name='dimension',
                    task_id=task_id, path=results_dir, kernel=True,
                    internal=False)

    if copy_task:
        if settings['permute_time'] and not \
                settings['permute_width'] and not settings['permute_xor']:
            ssp.plot_accuracy_vs_bptt_steps(results, seed_groups,
                task_id=task_id, path=results_dir)

    # Make multi-task plots.
    if num_tasks > 1:
        ssp.plot_importance_vs_task(results, seed_groups, path=results_dir)
        ssp.plot_dimension_vs_task(results, seed_groups, path=results_dir)
        ssp.plot_dimension_vs_task(results, seed_groups, path=results_dir,
            internal=False)
        ssp.plot_dimension_vs_task(results, seed_groups, path=results_dir,
            onto_task_1=True)
        ssp.print_dimension_vs_task(results, seed_groups, p_var=cmd_args.p_var)
        if num_runs == 1:
            ssp.plot_per_ts(results, seed_groups, path=results_dir,
                key_name='dimension')
            ssp.plot_per_ts(results, seed_groups, path=results_dir,
                key_name='dimension', internal=False)
            if cmd_args.do_kernel_pca:
                ssp.plot_per_ts(results, seed_groups, key_name='dimension',
                    path=results_dir, kernel=True)
            if copy_task:
                ssp.plot_per_ts(results, seed_groups, path=results_dir,
                    key_name='accuracy') 
    if cmd_args.do_supervised_dimred:
        ssp.plot_supervised_dimension_vs_task(results, seed_groups,
            path=results_dir)
        ssp.plot_supervised_dimension_vs_task(results, seed_groups,
            path=results_dir, key='accu')
        ssp.plot_supervised_dimension_vs_task(results, seed_groups,
            path=results_dir, key='accu', stop_bit=True)

    # Make multi-run plots.
    if num_runs > 1:
        ssp.plot_across_runs(results, seed_groups, path=results_dir,
            do_kernel_pca=cmd_args.do_kernel_pca, n_pcs=cmd_args.n_pcs,
            p_var=cmd_args.p_var)

    # To get the evolution of importance values type in the command line:
    # >> ``tensorboard --logdir path``
    # where `path` is the provided `out_dir` command line argument.

if __name__=='__main__':
    pass