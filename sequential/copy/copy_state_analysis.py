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
# @title           :sequential/copy/copy_state_analysis.py
# @author          :mc
# @contact         :mariacer@ethz.ch
# @created         :16/04/2020
# @version         :1.0
# @python_version  :3.6.8
"""
Study dimensionality of hidden states in copy task variants
-----------------------------------------------------------

In this script, we perform an analysis of the dimensionality of the hidden
states of a recurrent network, for different continual learning experiments
where the permuted copy task has an increasing pattern length.

Run as follows:

.. code-block:: 

    python3 copy_state_analysis.py path/to/results/folder/

For running these analyses, one needs to have run before the following:

.. code-block::

    python3 hpsearch.py --grid_module=ewc_study_config

Making sure that the different runs have different complexities, and that they
all have the following arguments activated for all runs:
`--store_final_models --store_during_models`.

"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import train_utils_copy as ctu
import train_args_copy
from sequential.ht_analyses.state_space_analysis import run

if __name__=='__main__':

    ### Get default config for current dataset.
    config = train_args_copy.parse_cmd_arguments(default=True)

    ### Extract important functions.
    task_loss_func = ctu.get_copy_loss_func
    accuracy_func = ctu.get_accuracy
    generate_tasks_func = ctu.generate_copy_tasks

    ### Create dictionary explaining important settings for the current analysis
    # complexity_measure: Indicates the name of the hyperparameter across 
    #       which complexity differs across runs (ex: 
    #       "first_task_seq_length" for the copy task)
    # complexity_measure_name: Indicates the name of the complexity measure that 
    #       is used.
    # forced_params: List of key-value pairs specifying hyperparameter 
    #       values that should be fixed across runs
    # fixed_params: List of hyperparameter names for which values should be
    #       identical across tasks (used for sanity checks)
    analysis_kwd = {'complexity_measure': 'first_task_input_len',
                    'complexity_measure_name': 'pattern_length',
                    'forced_params': [],
                    'fixed_params': ['num_tasks', 'seq_width', 'permute_time',\
                         'permute_width', 'permute_xor']}

    run(config, analysis_kwd, task_loss_func, accuracy_func, \
        generate_tasks_func)