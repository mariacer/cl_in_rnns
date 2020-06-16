#!/usr/bin/env python3
# Copyright 2019 Christian Henning
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
"""
- **title**          :sequential/smnist/hp_search_smnist.py
- **author**         :ch, mc
- **contact**        :henningc, mariacer@ethz.ch
- **created**        :09/01/2020
- **version**        :1.0
- **python_version** :3.6.8

Configuration file for the hyperparameter search of the SMNIST task.

"""
##########################################
### Please define all parameters below ###
##########################################

grid = {
    'n_iter':[10],
    ## Continual Learning Options ###
    # 'beta': [.005, .001, .01, 0.05, 0.1],
    'train_from_scratch': [False],
    # 'multi_head': [False],
    # # 'num_tasks': [5],

    # ### Training Options ###
    # 'batch_size': [200, 64],
    # 'n_iter': [5000, 2500, 10000],
    # 'lr': [1e-3, 1e-4, 5e-4, 1e-2],
    # #'weight_decay': [0.],
    # 'adam_beta1': [.9],
    # # 'clip_grad_norm': [-1, 2.5, 5],

    # ### Recurrent Network Options ###
    # 'rnn_arch': ['"256"', '"128"', '"512"','"64"',],
    # 'net_act': ['tanh'], # 'relu'
    # #'use_vanilla_rnn': [False],

    # ### Hypernet Options ###
    # #'hyper_chunks' : [-1],
    # 'hnet_arch' : ['"10"', '"50"', '"10,10"', '"100"'], #['""']
    # 'hnet_act' : ['relu'],
    # 'temb_size' : [2, 4, 16, 32],
    # 'emb_size' : [32],
    #'hnet_noise_dim': [-1],
    #'hnet_dropout_rate': [-1],
    #'temb_std': [-1],

    ### Evaluation options ###
    #'val_iter' : [500],

    ### Miscellaneous options ###
    'use_cuda' : [True],
    #'deterministic_run': [False],
    #'show_plots': [False],
    #'data_random_seed': [42],
    #'random_seed': [42],
    #'store_activations': [False],
    #'use_masks': [False],
    #'multitask': [False],
    #'train_tnet_once': [False],
    #'reinit_tnet': [False],
    'hnet_all': [True],

    ### EWC Options ###
    'use_ewc': [False],
    #'ewc_gamma': [1.],
    #'ewc_lambda': [5000.],
    #'n_fisher': [-1],
    #'tbptt_fisher': [-1],
    #'ts_weighting_fisher': ['none'], # 'none', 'last', 'discount'

    ### SI Options ###
    'use_si': [False],
    # 'si_lambda': [1.],
    # 'si_task_loss_only': [False],

    ## Context-Modulation Options ###
    'use_context_mod': [False]
}

"""Parameter grid for grid search.

Define a dictionary with parameter names as keys and a list of values for
each parameter. For flag arguments, simply use the values :code:`[False, True]`.
Note, the output directory is set by the hyperparameter search script.
Therefore, it always assumes that the argument `--out_dir` exists and you
**should not** add `out_dir` to this `grid`!

Example:
    .. code-block:: python

        grid = {'option1': [10], 'option2': [0.1, 0.5],
                'option3': [False, True]}

    This dictionary would correspond to the following 4 configurations:

    .. code-block:: console

        python3 SCRIPT_NAME.py --option1=10 --option2=0.1
        python3 SCRIPT_NAME.py --option1=10 --option2=0.5
        python3 SCRIPT_NAME.py --option1=10 --option2=0.1 --option3
        python3 SCRIPT_NAME.py --option1=10 --option2=0.5 --option3

If fields are commented out (missing), the default value is used.
Note, that you can specify special :attr:`conditions` below.
"""

conditions = [    
]

"""Define exceptions for the grid search.

Sometimes, not the whole grid should be searched. For instance, if an `SGD`
optimizer has been chosen, then it doesn't make sense to search over multiple
`beta2` values of an Adam optimizer.
Therefore, one can specify special conditions or exceptions.
Note* all conditions that are specified here will be enforced. Thus, **they
overwrite the** :attr:`grid` **options above**.

How to specify a condition? A condition is a key value tuple: whereas as the
key as well as the value is a dictionary in the same format as in the
:attr:`grid` above. If any configurations matches the values specified in the
"key" dict, the values specified in the "values" dict will be searched instead.

Note, if arguments are commented out above but appear in the conditions, the
condition will be ignored.
"""

####################################
### DO NOT CHANGE THE CODE BELOW ###
####################################
### This code only has to be adapted if you are setting up this template for a
### new simulation script!

# Name of the script that should be executed by the hyperparameter search.
# Note, the working directory is set seperately by the hyperparameter search
# script, so don't include paths.
_SCRIPT_NAME = 'train_split_smnist.py'

# This file is expected to reside in the output folder of the simulation.
_SUMMARY_FILENAME = 'performance_overview.txt'

# These are the keywords that are supposed to be in the summary file.
# A summary file always has to include the keyword `finished`!.
_SUMMARY_KEYWORDS = [
    # Track all performance measures with respect to the best mean accuracy.
    'mean_final_accuracy',
    'std_final_accuracy',
    'mean_during_accuracy',
    'std_during_accuracy',
    'min_final_accuracy',
    'min_during_accuracy',
    'final_accuracy',
    'during_accuracy',
    'num_train_iter',
    'finished'
]

# The name of the command-line argument that determines the output folder
# of the simulation.
_OUT_ARG = 'out_dir'

# In case you need a more elaborate parser than the default one define by the
# function :func:`hpsearch.hpsearch._get_performance_summary`, you can pass a
# function handle to this attribute.
# Value `None` results in the usage of the default parser.
_SUMMARY_PARSER_HANDLE = None # Default parser is used.
#_SUMMARY_PARSER_HANDLE = _get_performance_summary # Custom parser is used.

def _performance_criteria(summary_dict, performance_criteria):
    """Evaluate whether a run meets a given performance criteria.

    This function is needed to decide whether the output directory of a run is
    deleted or kept.

    Args:
        summary_dict: The performance summary dictionary as returned by
            :attr:`_SUMMARY_PARSER_HANDLE`.
        performance_criteria (float): The performance criteria. E.g., see
            command-line option `performance_criteria` of script
            :mod:`hpsearch.hpsearch_postprocessing`.

    Returns:
        bool: If :code:`True`, the result folder will be kept as the performance
        criteria is assumed to be met.
    """
    performance = float(summary_dict['mean_final_accuracy'][0])
    return performance > performance_criteria

# A function handle, that is used to evaluate the performance of a run.
_PERFORMANCE_EVAL_HANDLE = _performance_criteria

# A key that must appear in the `_SUMMARY_KEYWORDS` list. If `None`, the first
# entry in this list will be selected.
# The CSV file will be sorted based on this keyword. See also attribute
# `_PERFORMANCE_SORT_ASC`.
_PERFORMANCE_KEY = 'mean_final_accuracy'
assert(_PERFORMANCE_KEY is None or _PERFORMANCE_KEY in _SUMMARY_KEYWORDS)
# Whether the CSV should be sorted ascending or descending based on the
# `_PERFORMANCE_KEY`.
_PERFORMANCE_SORT_ASC = False

# FIXME: This attribute will vanish in future releases.
# This attribute is only required by the `hpsearch_postprocessing` script.
# A function handle to the argument parser function used by the simulation
# script. The function handle should expect the list of command line options
# as only parameter.
# Example:
# from sequential.smnist import train_args_smnist as targs
# f = lambda argv : targs.parse_cmd_arguments(argv=argv)
# _ARGPARSE_HANDLE = f
from sequential.smnist import train_args_smnist as targs
f = lambda argv : targs.parse_cmd_arguments(argv=argv)
_ARGPARSE_HANDLE = f

if __name__ == '__main__':
    pass


