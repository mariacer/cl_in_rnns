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
- **title**          :sequential/smnist/hpconfig_smnist_multitask.py
- **author**         :ch, mc
- **contact**        :henningc, mariacer@ethz.ch
- **created**        :06/01/2020
- **version**        :1.0
- **python_version** :3.6.8

Configuration file for the multitask hyperparameter search of the SMNIST task.
"""
from sequential.copy import hp_search_copy as hpcog
from sequential.copy import hp_search_copy_multitask as hpcopy_mt

##########################################
### Please define all parameters below ###
##########################################

grid = {
    ## Continual Learning Options ###
    #'multi_head': [False],
    #'num_tasks': [5],
    #'num_classes_per_task': [2],

    ### Training Options ###
    #'batch_size': [64],
    #'n_iter': [5000],
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
    #'nh_separate_out_head': [False],
    'use_new_hnet': [False],

    ### Initialization Options ###
    #'std_normal_temb': [1.],
    #'std_normal_emb': [1.],
    #'hyper_fan_init': [False],

    ### Evaluation options ###
    #'val_iter' : [250],
    #'val_set_size': [100],

    ### Miscellaneous options ###
    'use_cuda' : [True],
    #'deterministic_run': [False],
    #'show_plots': [False],
    #'data_random_seed': [42],
    #'random_seed': [42],
    #'store_activations': [False],
    'multitask': [True],
    #'input_task_identity': [False],
    #'use_masks': [False],
    #'mask_fraction': [.8],
    #'hnet_all': [False],
    #'ts_weighting': ['last'], # 'none', 'last', 'last_ten_percent',
                               # 'unpadded', 'discount'
    #'orthogonal_hh_init': [False],
    #'orthogonal_hh_reg': [-1],
    #'store_final_models': [False],
    #'use_best_models': [False],

    ### Context-Modulation Options ###
    #'use_context_mod': [False],
    #'no_context_mod_outputs': [False],
    #'context_mod_inputs': [False],
    #'context_mod_post_activation': [False],
    #'context_mod_last_step': [False],
    'checkpoint_context_mod': [False], # Not implemented for multitask!
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
_SCRIPT_NAME = 'train_split_smnist.py'

# This file is expected to reside in the output folder of the simulation.
_SUMMARY_FILENAME = hpcopy_mt._SUMMARY_FILENAME

# These are the keywords that are supposed to be in the summary file.
# A summary file always has to include the keyword `finished`!.
_SUMMARY_KEYWORDS = hpcopy_mt._SUMMARY_KEYWORDS

# The name of the command-line argument that determines the output folder
# of the simulation.
_OUT_ARG = hpcopy_mt._OUT_ARG

# In case you need a more elaborate parser than the default one define by the
# function :func:`hpsearch.hpsearch._get_performance_summary`, you can pass a
# function handle to this attribute.
# Value `None` results in the usage of the default parser.
_SUMMARY_PARSER_HANDLE = hpcopy_mt._SUMMARY_PARSER_HANDLE

# A function handle, that is used to evaluate the performance of a run.
_PERFORMANCE_EVAL_HANDLE = hpcopy_mt._PERFORMANCE_EVAL_HANDLE

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
# >>> from classifier.imagenet import train_args as targs
# >>> f = lambda argv : targs.parse_cmd_arguments(mode='cl_ilsvrc_cub',
# ...                                             argv=argv)
# >>> _ARGPARSE_HANDLE = f
from sequential.smnist import train_args_smnist as targs
f = lambda argv : targs.parse_cmd_arguments(argv=argv)
_ARGPARSE_HANDLE = f

if __name__ == '__main__':
    pass


