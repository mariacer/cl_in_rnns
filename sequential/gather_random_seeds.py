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
# @title           :sequential/gather_random_seeds.py
# @author          :mc
# @contact         :mariacer@ethz.ch
# @created         :05/05/2020
# @version         :1.0
# @python_version  :3.6.8
"""
Gather results of a given experiment for different random seeds.
----------------------------------------------------------------

Given a certain configuration this script runs the same experiment with
different random seed values, to assess robustness and gather results for
publication.

The configuration can be provided directly, loaded from a run, or chosen as the
best performing configuration from a set of hyperparameter results.
If the configuration is loaded from a specific run folder, the results of this 
folder are copied, so that we can avoid running one of the experiments.
"""
import argparse
from argparse import Namespace
from distutils.dir_util import copy_tree
import importlib
import os
import pickle
import pandas as pd
import shutil
from subprocess import call
import numpy as np
from warnings import warn

from hpsearch import hpsearch
from sequential import train_args_sequential
import utils.misc as misc

def get_single_run_config(out_dir):
    """Load the config file from a specified experiment.

    Args:
        out_dir (str): The path to the experiment.

    Returns:
        The Namespace object containing argument names and values.
    """
    print('Loading the configuration of run: %s'%out_dir)

    if not os.path.exists(os.path.join(out_dir, "config.pickle")):
        # Currently, we can't read the config from the results csv files.
        raise NotImplementedError('The run "%s" does not contain a '%out_dir + 
            '"config.pickle" file. If working locally, please make sure '+
            'you copy the output folder of the current run from the cluster.')

    with open(os.path.join(out_dir, "config.pickle"), "rb") as f:
        config = pickle.load(f)

    # Check for old command line arguments and make compatible with new version.
    config = train_args_sequential.update_cli_args(config)
    config.show_plots = False

    config.store_during_models = True
    config.store_final_models = True

    return config

def get_hpsearch_config(out_dir):
    """Load the config file from a hyperparameter search.

    This file loads the results of the hyperparameter search, and select the
    configuration that lead to the best mean final accuracies.

    Args:
        out_dir (str): The path to the experiment.

    Returns:
        (tuple): Tuple containing:

        - **config**: The config of the best run.
        - **best_out_dir**: The path to the best run.
    """
    result_file = os.path.join(out_dir, 'search_results.csv')
    post_result_file = os.path.join(out_dir, 'postprocessing_results.csv')

    # Load performance results
    if os.path.exists(post_result_file):
        df = pd.read_csv(post_result_file, sep=';')
    else:
        try:
            df = pd.read_csv(result_file, sep=';')
        except:
            df = pd.read_csv(result_file, sep=',')
    # Get the best performing run.
    best_index = df['mean_final_accuracy'].idxmax()

    best_out_dir = df['out_dir'][best_index]
    best_folder_name = os.path.basename(os.path.normpath(best_out_dir))

    # Check whether the path to the best run needs to be changed. It might be
    # that the path given in the hpsearch summary file is outdated, for
    # example in case we have renamed the folder of the hpsearches.
    best_out_dir = os.path.join(out_dir, best_folder_name)
    # The post-processing might have marked the run for deletion.
    if not os.path.exists(best_out_dir):
        best_out_dir = os.path.join(out_dir, 'TO_BE_DELETED', best_folder_name)
    assert os.path.exists(best_out_dir)

    return get_single_run_config(best_out_dir), best_out_dir

def delete_object_from_text(filename, name, start_delimiter, end_delimiter):
    """Delete a given list or dictionary from text.

    Args:
        filename (str): The name of the text file.
        name (str): The name of the object to be deleted.
        start_delimiter (str): The start delimiter character of the object.
        end_delimiter (str): The end delimiter character of the object.

    Returns:
        (int): The initial line where object was found in the text.

    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Find position of grid.
    interval = [None, None]
    num_open_delimiter = -1
    for i, line in enumerate(lines):
        if name in line and '=' in line and start_delimiter in line:
            interval[0] = i
            num_open_delimiter = 0

        num_open_delimiter += line.count(start_delimiter)
        num_open_delimiter -= line.count(end_delimiter)
        if num_open_delimiter == 0:
            interval[1] = i
            break

    # Delete grid.
    with open(filename, "w") as f:
        for i, line in enumerate(lines):
            if i < interval[0] or i > interval[1]:
                f.write(line)
    with open(filename, 'r') as file:
        lines = file.readlines()

    return interval[0]

def write_new_grid_to_text(filename, config, location, random_seeds, cmd_args,
        kwds=None):
    """Write new grid to the text file.

    This function writes a new grid based on the provided config to the
    hpsearch config file. It ignores some keyworks that might be part of a
    stored config after evaluation, but that are not part of the command line
    arguments.

    Args:
        filename (str): The name of the text file.
        config: The config of the best run.
        location (int): The line where to add the grid.
        random_seeds (list): The list of seeds to use.
        kwds (list): The list of keywords to add to the grid.

    """
    if kwds == None:
        kwds = list(vars(config).keys())
    for key in ['tnet_weights', 'out_dir', 'classification', 'input_dim', \
            'out_dim', 'hnet_out', 'compression_ratio', 'coresets', 'mode']:
        if key in kwds:
            kwds.remove(key)

    # Make sure we don't use the same seed as in the original run.
    if config.random_seed in random_seeds:
        random_seeds.remove(config.random_seed)
        random_seeds.append(max(random_seeds)+1)

    with open(filename, 'r') as file:
        lines = file.readlines()

    # Get the list of keys that will be added to the grid.
    grid_keys = []
    for key in vars(config).keys():
        if key in kwds:
            grid_keys.append(key)

    with open(filename, "w") as f:
        for i, line in enumerate(lines):
            if i == location:
                f.write('grid = { \n')
                for key in grid_keys:
                    value = getattr(config, key)
                    if key == 'random_seed':
                        value = random_seeds
                        new_line = '     "%s" : %s'%(key, str(value))
                    elif cmd_args.vary_data_seed and key == 'data_random_seed':
                        # Note, will be overwritten in the conditions!
                        value = [-1]
                        new_line = '     "%s" : %s'%(key, str(value))
                    elif type(value) == str or type(value) == list:
                        new_line = '     "%s" : ["%s"]'%(key, str(value)) 
                    else:
                        new_line = '     "%s" : [%s]'%(key, str(value)) 

                    end_line = ',\n'
                    if key == grid_keys[-1]: # last key
                        end_line = '\n'
                    new_line += end_line

                    f.write(new_line)
                f.write('} \n')
            f.write(line)

    return random_seeds

def write_new_conditions_to_text(filename, location, random_seeds, cmd_args):
    """Remove conditions from the text file.

    This functon removes all conditions from the hpsearch config.

    Args:
        filename (str): The name of the text file.
        location (int): The line where to add the grid.

    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    with open(filename, "w") as f:
        for i, line in enumerate(lines):
            if i == location:
                if not cmd_args.vary_data_seed:
                    f.write('conditions = [] \n')
                else:
                    f.write('conditions = [ \n')
                    for rs in random_seeds:
                        f.write('({\'random_seed\': ['+ str(rs) +
                                ']}, {\'data_random_seed\': ['+ str(rs) +
                                '],}),\n')
                    f.write(']\n')

            if not 'conditions = conditions' in line:
                f.write(line)

def get_command_line(grid_module_name, results_dir, cmd_args):
    """Generate the command line for the hpsearch.

    Args:
        grid_module_name (str): The name of the module for the hpsearch.
        results_dir (str): The path to the results directory.
        cmd_args: The command line arguments.

    Returns:
        (str): The command line to be executed.

    """
    # Fix resources argument to make fit for command line.
    resources = cmd_args.resources.replace(" ", "")
    resources = resources.strip('"')

    cluster_cmd_prefix = ''
    cluster_cmd_suffix = ''
    non_cluster_cmd_suffix = ''
    if cmd_args.run_cluster and cmd_args.scheduler == 'lsf':
        cluster_cmd_prefix = 'bsub -n 1 -W %s:00 '%cmd_args.num_tot_hours + \
            '-e random_seeds.err -o random_seeds.out -R "rusage[mem=8000]" '

        resources = cmd_args.resources.replace(" ", "")
        cluster_cmd_suffix = ' --run_cluster ' + \
            '--scheduler=%s '%cmd_args.scheduler +\
            '--num_jobs=%s '%cmd_args.num_jobs +\
            '--num_hours=%s '%cmd_args.num_hours + \
            '--resources="\\"%s\\"" '%resources + \
            '--num_searches=1000 '
    elif cmd_args.run_cluster:
        assert cmd_args.scheduler == 'slurm'
        cluster_cmd_suffix = ' --run_cluster ' + \
            '--scheduler=%s ' % cmd_args.scheduler + \
            '--num_jobs=%s ' % cmd_args.num_jobs + \
            '--num_hours=%s ' % cmd_args.num_hours + \
            '--slurm_mem=%s ' % cmd_args.slurm_mem + \
            '--slurm_gres=%s ' % cmd_args.slurm_gres + \
            '--slurm_partition=%s ' % cmd_args.slurm_partition + \
            '--slurm_qos=%s ' % cmd_args.slurm_qos + \
            '--slurm_constraint=%s ' % cmd_args.slurm_constraint + \
            '--num_searches=1000 '
    else:
        non_cluster_cmd_suffix = \
            '--visible_gpus=%s ' % cmd_args.visible_gpus + \
            '--allowed_load=%f ' % cmd_args.allowed_load + \
            '--allowed_memory=%f ' % cmd_args.allowed_memory + \
            '--sim_startup_time=%d ' % cmd_args.sim_startup_time + \
            '--max_num_jobs_per_gpu=%d ' % cmd_args.max_num_jobs_per_gpu
    cmd_str = 'TMP_CUR_DIR="$(pwd -P)" && pushd ../../hpsearch && ' + \
        cluster_cmd_prefix + \
        'python3 hpsearch.py --grid_module=%s '%grid_module_name + \
        '--out_dir=%s --force_out_dir '%results_dir + \
        '--dont_force_new_dir --run_cwd=$TMP_CUR_DIR ' + \
        cluster_cmd_suffix + non_cluster_cmd_suffix + ' && popd'

    return cmd_str


def write_seeds_summary(results_dir):
    """Write the averages and SEM when aggregating all seeds.

    Args:
        results_dir (str): The results directory.

    """

    # Load results summary file.
    seeds_summary_file = os.path.join(results_dir, 'search_results.csv')
    post_result_file = os.path.join(results_dir, 'postprocessing_results.csv')
    if os.path.exists(post_result_file):
        seeds_summary_file = post_result_file
        # FIXME We should read the performance summaries from all result
        # folders directly.
        warn('Post-processing result file is used for collecting seed ' +
             'information. Note, this file might ignore the original run, ' +
             'that was not part of the hpsearch.')

    try:
        seeds_summary = pd.read_csv(seeds_summary_file, sep=';')
        mean_final_accs = seeds_summary['mean_final_accuracy'].values
    except:
        seeds_summary = pd.read_csv(seeds_summary_file, sep=',')
        mean_final_accs = seeds_summary['mean_final_accuracy'].values
    mean_during_accs = seeds_summary['mean_during_accuracy'].values
    finished = seeds_summary['finished'].values

    # Exclude non finished runs.
    mean_final_accs = mean_final_accs[finished==1]
    mean_during_accs = mean_during_accs[finished==1]

    # Compute the during and final averages, and the standard error of the mean
    ns = len(mean_during_accs)
    during_results = (np.mean(mean_during_accs), \
        np.std(mean_during_accs)/np.sqrt(ns))
    final_results = (np.mean(mean_final_accs), \
        np.std(mean_final_accs)/np.sqrt(ns))

    # Write into a summary text file.
    filename = os.path.join(results_dir, 'seeds_summary_text.txt')
    with open(filename, "w") as f:
        f.write('During accuracy (mean +/- sem): %.2f +- %.2f\n'%during_results)
        f.write('Final accuracy (mean +/- sem):  %.2f +- %.2f\n'%final_results)
        f.write('Number of seeds: %i \n\n'%ns)
        f.write('Publication tables style: \n')
        f.write('%.2f $\pm$  %.2f & %.2f $\pm$  %.2f '% \
            (*during_results, *final_results))


def run(ref_module, results_dir='./out/random_seeds', config=None,
        ignore_kwds=None, forced_params=None):
    """Run the script.

    Args:
        ref_module (str): Name of the reference module which contains the 
            hyperparameter search config that can be modified to gather random 
            seeds.
        results_dir (str, optional): The path where to store the results.
        config: The Namespace object containing argument names and values.
            If provided, all random seeds will be gathered from zero, with no
            reference run.
        ignore_kwds (list, optional): The list of keywords in the config file
            to exclude from the grid.
        forced_params (dict, optional): Dict of key-value pairs specifying
            hyperparameter values that should be fixed across runs
    """
    if ignore_kwds is None:
        ignore_kwds = []
    if forced_params is None:
        forced_params = {}

    ### Parse the command-line arguments.
    parser = argparse.ArgumentParser(description= \
        'Gathering random seeds for the specified experiment.')
    parser.add_argument('--out_dir', type=str, default='',
                        help='The output directory of the run or runs. ' +
                             'For single runs, the configuration will be ' +
                             'loaded and run with different seeds.' +
                             'For multiple runs, i.e. results of ' +
                             'hyperparameter searches, the configuration ' +
                             'leading to the best mean final accuracy ' +
                             'will be selected and run with different seeds. ' +
                             'Default: %(default)s.')
    parser.add_argument('--config_name', type=str,
                        default='hpsearch_random_seeds.py',
                        help='The name of the hpsearch config file. Since ' +
                             'multiple random seed gathering experiments ' +
                             'might be running in parallel, it is important ' +
                             'that this file has a unique name for each ' +
                             'experiment. Default: %(default)s.')
    parser.add_argument('--config_pickle', type=str, default='',
                        help='The path to a pickle file containing a run ' +
                             ' config that will be loaded.')
    parser.add_argument('--num_seeds', type=int, default=10,
                        help='The number of different random seeds.')
    # FIXME `None` is not a valid default value.
    parser.add_argument('--seeds_list', type=str, default=None,
                        help='The list of seeds to use. If specified, ' +
                             '"num_seeds" will be ignored.')
    parser.add_argument('--vary_data_seed', action='store_true',
                        help='If activated, "data_random_seed"s are set ' +
                             'equal to "random_seed"s. Otherwise only ' +
                             '"random_seed"s are varied.')
    parser.add_argument('--num_tot_hours', type=int, metavar='N', default=120,
                        help='If "run_cluster" is activated, then this ' +
                             'option determines the maximum number of hours ' +
                             'the entire search may run on the cluster. ' +
                             'Default: %(default)s.')
    # FIXME Arguments below are copied from hpsearch.
    parser.add_argument('--run_cluster', action='store_true',
                        help='This option would produce jobs for a GPU ' +
                             'cluser running a job scheduler (see option ' +
                             '"scheduler".')
    parser.add_argument('--scheduler', type=str, default='lsf',
                        choices=['lsf', 'slurm'],
                        help='The job scheduler used on the cluster. ' +
                             'Default: %(default)s.')
    parser.add_argument('--num_jobs', type=int, metavar='N', default=8,
                        help='If "run_cluster" is activated, then this ' +
                             'option determines the maximum number of jobs ' +
                             'that can be submitted in parallel. ' +
                             'Default: %(default)s.')
    parser.add_argument('--num_hours', type=int, metavar='N', default=24,
                        help='If "run_cluster" is activated, then this ' +
                             'option determines the maximum number of hours ' +
                             'a job may run on the cluster. ' +
                             'Default: %(default)s.')
    parser.add_argument('--resources', type=str,
                        default='"rusage[mem=8000, ngpus_excl_p=1]"',
                        help='If "run_cluster" is activated and "scheduler" ' +
                             'is "lsf", then this option determines the ' +
                             'resources assigned to job in the ' +
                             'hyperparameter search (option -R of bsub). ' +
                             'Default: %(default)s.')
    parser.add_argument('--slurm_mem', type=str, default='8G',
                        help='If "run_cluster" is activated and "scheduler" ' +
                             'is "slurm", then this value will be passed as ' +
                             'argument "mem" of "sbatch". An empty string ' +
                             'means that "mem" will not be specified. ' +
                             'Default: %(default)s.')
    parser.add_argument('--slurm_gres', type=str, default='gpu:1',
                        help='If "run_cluster" is activated and "scheduler" ' +
                             'is "slurm", then this value will be passed as ' +
                             'argument "gres" of "sbatch". An empty string ' +
                             'means that "gres" will not be specified. ' +
                             'Default: %(default)s.')
    parser.add_argument('--slurm_partition', type=str, default='',
                        help='If "run_cluster" is activated and "scheduler" ' +
                             'is "slurm", then this value will be passed as ' +
                             'argument "partition" of "sbatch". An empty ' +
                             'string means that "partition" will not be ' +
                             'specified. Default: %(default)s.')
    parser.add_argument('--slurm_qos', type=str, default='',
                        help='If "run_cluster" is activated and "scheduler" ' +
                             'is "slurm", then this value will be passed as ' +
                             'argument "qos" of "sbatch". An empty string ' +
                             'means that "qos" will not be specified. ' +
                             'Default: %(default)s.')
    parser.add_argument('--slurm_constraint', type=str, default='',
                        help='If "run_cluster" is activated and "scheduler" ' +
                             'is "slurm", then this value will be passed as ' +
                             'argument "constraint" of "sbatch". An empty ' +
                             'string means that "constraint" will not be ' +
                             'specified. Default: %(default)s.')
    parser.add_argument('--visible_gpus', type=str, default='',
                        help='If "run_cluster" is NOT activated, then this ' +
                             'option determines the CUDA devices visible to ' +
                             'the hyperparameter search. A string of comma ' +
                             'separated integers is expected. If the list is ' +
                             'empty, then all GPUs of the machine are used. ' +
                             'The relative memory usage is specified, i.e., ' +
                             'a number between 0 and 1. If "-1" is given, ' +
                             'the jobs will be executed sequentially and not ' +
                             'assigned to a particular GPU. ' +
                             'Default: %(default)s.')
    parser.add_argument('--allowed_load', type=float, default=0.5,
                        help='If "run_cluster" is NOT activated, then this ' +
                             'option determines the maximum load a GPU may ' +
                             'have such that another process may start on ' +
                             'it. The relative load is specified, i.e., a ' +
                             'number between 0 and 1. Default: %(default)s.')
    parser.add_argument('--allowed_memory', type=float, default=0.5,
                        help='If "run_cluster" is NOT activated, then this ' +
                             'option determines the maximum memory usage a ' +
                             'GPU may have such that another process may ' +
                             'start on it. Default: %(default)s.')
    parser.add_argument('--sim_startup_time', type=int, metavar='N', default=60,
                        help='If "run_cluster" is NOT activated, then this ' +
                             'option determines the startup time of ' +
                             'simulations. If a job was assigned to a GPU, ' +
                             'then this time (in seconds) has to pass before ' +
                             'options "allowed_load" and "allowed_memory" ' +
                             'are checked to decide whether a new process ' +
                             'can be send to a GPU.Default: %(default)s.')
    parser.add_argument('--max_num_jobs_per_gpu', type=int, metavar='N',
                        default=1,
                        help='If "run_cluster" is NOT activated, then this ' +
                             'option determines the maximum number of jobs ' +
                             'per GPU that can be submitted in parallel. ' +
                             'Note, this script does not validate whether ' +
                             'other processes are already assigned to a GPU. ' +
                             'Default: %(default)s.')
    cmd_args = parser.parse_args()
    out_dir = cmd_args.out_dir

    if cmd_args.out_dir == '' and cmd_args.config_pickle != '':
        with open(cmd_args.config_pickle, "rb") as f:	
            config = pickle.load(f)

    # Either a config or an experiment folder need to be provided.
    assert config is not None or cmd_args.out_dir != ''
    if cmd_args.out_dir == '':
        out_dir = config.out_dir

    # Make sure that the provided hpsearch config file name does not exist.
    config_name = cmd_args.config_name
    if config_name[-3:] != '.py':
        config_name = config_name + '.py'
    if os.path.exists(config_name):
        overwrite = input('The config file "%s" '% config_name + \
            'already exists! Do you want to overwrite the file? [y/n] ')
        if not overwrite in ['yes','y','Y']:
            exit()

    # The following ensures that we can safely use `basename` later on.
    out_dir = os.path.normpath(out_dir)

    ### Create directory for results.
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # Define a subfolder for the current random seed runs.
    results_dir = os.path.join(results_dir, os.path.basename(out_dir))
    print('Random seeds will be gathered in folder %s.' % results_dir)

    if os.path.exists(results_dir):

        # If random seeds have been gathered already, simply get the results for 
        # publication.
        write_seeds_summary(results_dir)

        raise RuntimeError('Output directory %s already exists! ' %results_dir+\
            'seems like random seeds already have been gathered.')

    ### Get the experiments config.
    num_seeds = cmd_args.num_seeds
    if config is None:

        # Check if the current directory corresponds to a single run or not.
        # FIXME quick and dirty solution to figure out, whether it's a single
        # run.
        single_run = False
        if not os.path.exists(os.path.join(out_dir, 'search_results.csv')) \
                and not os.path.exists(os.path.join(out_dir, \
                    'postprocessing_results.csv')):
            single_run = True

        # Get the configuration.
        if single_run:
            config = get_single_run_config(out_dir)
            best_out_dir = out_dir
        else:
            config, best_out_dir = get_hpsearch_config(out_dir)

        # Since we already have a reference run, we can run one seed less.
        num_seeds -= 1

    if cmd_args.seeds_list is not None:
        seeds_list = misc.str_to_ints(cmd_args.seeds_list)
        cmd_args.num_seeds = len(seeds_list)
    else:
        seeds_list = list(range(num_seeds))

    # Replace config values provided via `forced_params`.
    if len(forced_params.keys()) > 0:
        for kwd, value in forced_params.items():
            setattr(config, kwd, value)

    ### Write down the hp search grid module in its own file.
    ref_module_basename = ref_module[[i for i,e in \
        enumerate(ref_module) if e == '.'][-1]+1:]
    ref_module_path = ref_module[:[i for i,e in \
        enumerate(ref_module) if e == '.'][-1]+1]
    shutil.copy(ref_module_basename + '.py', config_name)

    # Define the kwds to be added to the grid.
    kwds = list(vars(config).keys())
    for kwd in ignore_kwds:
        if kwd in kwds:
            kwds.remove(kwd)

    # Remove old grid and write new grid, and remove conditions.
    grid_loc = delete_object_from_text(config_name, 'grid', '{', '}')
    random_seeds = write_new_grid_to_text(config_name, config, grid_loc, \
        seeds_list, cmd_args, kwds=kwds)
    cond_loc = delete_object_from_text(config_name, 'conditions', \
        '[', ']')
    write_new_conditions_to_text(config_name, cond_loc, random_seeds, cmd_args)
    
    ### Run the hpsearch code with different random seeds.
    hpsearch_module = ref_module_path + config_name[:-3]
    cmd_str = get_command_line(hpsearch_module, results_dir, cmd_args)
    print(cmd_str)

    if cmd_args.run_cluster and cmd_args.scheduler == 'slurm':
        # FIXME hacky solution to write SLURM job script.
        # FIXME might be wrong to give the same `slurm_qos` to the hpsearch,
        # as the job might have to run much longer.
        job_script_fn = hpsearch._write_slurm_script(Namespace(**{
                'num_hours': cmd_args.num_tot_hours,
                'slurm_mem': '8G',
                'slurm_gres': '',
                'slurm_partition': cmd_args.slurm_partition,
                'slurm_qos': cmd_args.slurm_qos,
                'slurm_constraint': cmd_args.slurm_constraint,
            }),
            cmd_str, 'random_seeds')

        cmd_str = 'sbatch %s' % job_script_fn
        print('We will execute command "%s".' % cmd_str)

    # Execute the program.
    print('Starting gathering random seeds...')
    ret = call(cmd_str, shell=True,  executable='/bin/bash')
    print('Call finished with return code %d.' % ret)

    ### Add results of the reference run to our results folder.
    new_best_out_dir = os.path.join(results_dir, os.path.basename(out_dir))
    copy_tree(best_out_dir, new_best_out_dir)

    ### Store results of given run in CSV file.
    # FIXME Extremely ugly solution.
    imported_grid_module = importlib.import_module(hpsearch_module)
    hpsearch._read_config(imported_grid_module)

    results_file = os.path.join(results_dir, 'search_results.csv')
    cmd_dict = dict()
    for k in kwds:
        cmd_dict[k] = getattr(config, k)

    # Get training results.
    performance_dict = hpsearch._SUMMARY_PARSER_HANDLE(new_best_out_dir, -1)
    for k, v in performance_dict.items():
        cmd_dict[k] = v

    # Create or update the CSV file summarizing all runs.
    panda_frame = pd.DataFrame.from_dict(cmd_dict)
    if os.path.isfile(results_file):
        old_frame = pd.read_csv(results_file, sep=';')
        panda_frame = pd.concat([old_frame, panda_frame], sort=True)
    panda_frame.to_csv(results_file, sep=';', index=False)

    # Create a text file aggregating all results for publication.
    write_seeds_summary(results_dir)

if __name__=='__main__':
    run()
