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
- **title**          :sequential/smnist/hpconfig_smnist.py
- **author**         :be,ch, mc
- **contact**        :behret,henningc, mariacer@ethz.ch
- **created**        :15/04/2020
- **version**        :1.0
- **python_version** :3.6.8

Configuration file for the hyperparameter search of the SMNIST task.
"""
from sequential.copy import hp_search_copy as hpcog

##########################################
### Please define all parameters below ###
##########################################

grid = {
    ## Continual Learning Options ###
    #'beta': [.005],
    #'train_from_scratch': [False],
    #'multi_head': [False],
    'num_tasks': [5],
    #'num_classes_per_task': [2],

    ### Training Options ###
    #'batch_size': [64],
    'n_iter': [2000],
    #'lr': [1e-3],
    #'weight_decay': [0.],
    #'adam_beta1': [.9],
    #'clip_grad_norm': [-1],

    ### Recurrent Network Options ###
    #'rnn_arch': ['"256"'],
    #'srnn_pre_fc_layers': ['""'],
    #'srnn_post_fc_layers': ['""'],
    #'net_act': ['tanh'], # 'relu'
    #'use_vanilla_rnn': [False],

    ### Hypernet Options ###
    #'hyper_chunks' : [-1],
    #'hnet_arch' : ['"50,50"'],
    #'hnet_act' : ['relu'],
    #'temb_size' : [32],
    #'emb_size' : [32],
    #'hnet_noise_dim': [-1],
    #'hnet_dropout_rate': [-1],
    #'temb_std': [-1],

    ### New Hypernet Options ###
    #'nh_hnet_type' : ['hmlp'], # 'hmlp', 'chunked_hmlp', 'structured_hmlp',
                                # 'hdeconv', 'chunked_hdeconv'
    #'nh_hmlp_arch' : ['"50,50"'],
    #'nh_cond_emb_size' : [32],
    #'nh_chmlp_chunk_size' : [1000],
    #'nh_chunk_emb_size' : ['32'],
    #'nh_use_cond_chunk_embs' : [False],
    #'nh_hdeconv_shape' : ['"512,512,3"'],
    #'nh_hdeconv_num_layers' : [5],
    #'nh_hdeconv_filters' : ['"128,512,256,128"'],
    #'nh_hdeconv_kernels': ['"5"'],
    #'nh_hdeconv_attention_layers': ['"1,3"'],
    #'nh_hnet_net_act': ['relu'],
    #'nh_hnet_no_bias': [False],
    #'nh_hnet_dropout_rate': [-1],
    #'nh_hnet_specnorm': [False],
    #'nh_shmlp_chunk_sizes': ['8'],
    #'nh_shmlp_chunk_fc_layers': [False],
    'use_new_hnet': [False],

    ### Initialization Options ###
    #'std_normal_temb': [1.],
    #'std_normal_emb': [1.],
    #'hyper_fan_init': [False],

    ### Evaluation options ###
    #'val_iter' : [250],
    #'val_set_size': [1000],

    ### Miscellaneous options ###
    'use_cuda' : [True],
    #'deterministic_run': [False],
    #'show_plots': [False],
    #'data_random_seed': [42],
    #'random_seed': [42],
    #'store_activations': [False],
    #'multitask': [False],
    #'train_tnet_once': [False],
    #'reinit_tnet': [False],
    #'input_task_identity': [False],
    #'use_masks': [False],
    #'mask_fraction': [.8],
    #'hnet_all': [False],
    #'calc_hnet_reg_targets_online': [False],
    #'hnet_reg_batch_size': [-1],
    #'last_task_only': [False],
    #'ts_weighting': ['last'], # 'none', 'last', 'last_ten_percent',
                               # 'unpadded', 'discount'
    #'orthogonal_hh_init': [False],
    #'orthogonal_hh_reg': [-1],
    #'early_stopping_thld': [1e-3],
    #'es_warm_up_iter': [5000],
    #'es_best_val_diff': [.01],
    'ssmnist_seq_len': [3],
    #'analyse_hidden_pcs': [False],
    #'store_final_models': [False],
    #'store_during_models': [False],
    #'use_best_models': [False],
    #'during_acc_criterion': ['-1'],

    ### EWC Options ###
    #'use_ewc': [False],
    #'ewc_gamma': [1.],
    #'ewc_lambda': [5000.],
    #'n_fisher': [-1],
    #'tbptt_fisher': [-1],
    #'ts_weighting_fisher': ['last'], # 'none', 'last', 'last_ten_percent',
                                      # 'unpadded', 'discount'

    ### SI Options ###
    #'use_si': [False],
    # 'si_lambda': [1.],
    # 'si_task_loss_only': [False],

    ## Context-Modulation Options ###
    #'use_context_mod': [False],
    #'no_context_mod_outputs': [False],
    #'context_mod_inputs': [False],
    #'context_mod_post_activation': [False],
    #'context_mod_last_step': [False],
    #'checkpoint_context_mod': [False],
    #'context_mod_init': ['constant'], #'constant','normal','uniform','sparse'
    #'offset_gains': [False],
    #'dont_softplus_gains': [False],
    #'context_mod_per_ts': [False],
    #'sparsify_context_mod': [False],
    #'sparsification_reg_strength': [1.],
    #'sparsification_reg_type': ['l1'], # 'l1', 'log'
}

conditions = [

]

####################################
### DO NOT CHANGE THE CODE BELOW ###
####################################
conditions = conditions + hpcog._BASE_CONDITIONS

### This code only has to be adapted if you are setting up this template for a
### new simulation script!

# Name of the script that should be executed by the hyperparameter search.
# Note, the working directory is set seperately by the hyperparameter search
# script, so don't include paths.
_SCRIPT_NAME = 'train_seq_smnist.py'

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
    'compression_ratio',
    'rnn_arch',
    'hnet_arch',
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

