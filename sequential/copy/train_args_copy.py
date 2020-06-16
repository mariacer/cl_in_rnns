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
# title          :sequential/copy/train_args_copy.py
# author         :mc
# contact        :mariacer@ethz.ch
# created        :24/03/2020
# version        :1.0
# python_version :3.6.8
"""
Command-line arguments and default values for the copy task are handled here.
"""
import argparse
import warnings

import utils.cli_args as cli
import sequential.train_args_sequential as seq

def parse_cmd_arguments(default=False, argv=None):
    """Parse command-line arguments.

    Args:
        default (optional): If True, command-line arguments will be ignored and
            only the default values will be parsed.
        argv (optional): If provided, it will be treated as a list of command-
            line argument that is passed to the parser in place of sys.argv.

    Returns:
        The Namespace object containing argument names and values.
    """

    description = 'Continual learning on copy task.'

    parser = argparse.ArgumentParser(description=description)

    cli.cl_args(parser, show_beta=True, dbeta=0.005,
                show_from_scratch=True, show_multi_head=True,
                show_split_head_cl3=False,
                show_cl_scenario=False,
                show_num_tasks=True, dnum_tasks=6)
    cli.train_args(parser, show_lr=True, show_epochs=False,
        dbatch_size=64, dn_iter=5000,
        dlr=1e-3, show_clip_grad_value=False, show_clip_grad_norm=True,
        show_momentum=False, show_adam_beta1=True)
    seq.rnn_args(parser, drnn_arch='256', dnet_act='tanh')
    cli.hypernet_args(parser, dhyper_chunks=-1, dhnet_arch='10,10',
                          dtemb_size=2, demb_size=32, dhnet_act='sigmoid')
    # Args of new hnets.
    nhnet_args = cli.hnet_args(parser, allowed_nets=['hmlp', 'chunked_hmlp',
        'structured_hmlp', 'hdeconv', 'chunked_hdeconv'], dhmlp_arch='50,50',
        show_cond_emb_size=True, dcond_emb_size=32, dchmlp_chunk_size=1000,
        dchunk_emb_size=32, show_use_cond_chunk_embs=True,
        dhdeconv_shape='512,512,3', prefix='nh_',
        pf_name='new edition of a hyper-', show_net_act=True, dnet_act='relu',
        show_no_bias=True, show_dropout_rate=True, ddropout_rate=-1,
        show_specnorm=True, show_batchnorm=False, show_no_batchnorm=False)
    seq.new_hnet_args(nhnet_args)
    cli.init_args(parser, custom_option=False, show_normal_init=False,
                  show_hyper_fan_init=True)
    cli.eval_args(parser, dval_iter=250)
    magroup = cli.miscellaneous_args(parser, big_data=False,
        synthetic_data=True, show_plots=True, no_cuda=True,
        show_publication_style=False)
    seq.ewc_args(parser, dewc_lambda=5000., dn_fisher=-1, dtbptt_fisher=-1,
        show_ts_weighting_fisher=False)
    seq.si_args(parser, dsi_lambda=1.)
    seq.context_mod_args(parser, dsparsification_reg_type='l1', 
        dsparsification_reg_strength=1., dcontext_mod_init='constant')
    seq.miscellaneous_args(magroup, dmask_fraction=0.8, dclassification=True,
                           show_ts_weighting=False, show_use_ce_loss=False,
                           show_early_stopping_thld=True,
                           dearly_stopping_thld=-1)
    copy_sequence_args(parser)

    # Replay arguments.
    rep_args = seq.replay_args(parser, show_all_task_softmax=False)
    cli.generator_args(rep_args, dlatent_dim=100)
    cli.main_net_args(parser, allowed_nets=['simple_rnn'],
        dsrnn_rec_layers='256', dsrnn_pre_fc_layers='',
        dsrnn_post_fc_layers='',
        show_net_act=True, dnet_act='tanh', show_no_bias=True,
        show_dropout_rate=False, show_specnorm=False, show_batchnorm=False,
        prefix='dec_', pf_name='replay decoder')

    args = None
    if argv is not None:
        if default:
            warnings.warn('Provided "argv" will be ignored since "default" ' +
                          'option was turned on.')
        args = argv
    if default:
        args = []
    config = parser.parse_args(args=args)

    ### Check argument values!
    cli.check_invalid_argument_usage(config)
    seq.check_invalid_args_sequential(config)
    check_invalid_args_sequential(config)

    if config.train_from_scratch:
        # FIXME We could get rid of this warning by properly checkpointing and
        # loading all networks.
        warnings.warn('When training from scratch, only during accuracies ' +
                      'make sense. All other outputs should be ignored!')

    return config

def copy_sequence_args(parser):
    """This is a helper function of function :func:`parse_cmd_arguments` to add
    specific arguments to the argument group related to copy task sequences.

    Arguments specified in this function:
        - `first_task_input_len`
        - `input_len_step`
        - `input_len_variability`
        - `seq_width`
        - `pat_len`
        - `permute_width`
        - `permute_time`
        - `use_new_permuted_dhandler`
        - `permute_xor`
        - `permute_xor_iter`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
    """
    heading = 'Copy task random sequence options'

    sgroup = parser.add_argument_group(heading)
    sgroup.add_argument('--first_task_input_len', type=int, default=5,
                        help='The mean input length for the first task. ' +
                             'Default: %(default)s')
    sgroup.add_argument('--input_len_step', type=int, default=7,
                        help='The step in input lengths between succesive ' +
                             'tasks. For example, for a value of 7 and an ' +
                             'input length of 5 for the first task, the ' +
                             'mean input length of the second task is 12. ' +
                             'Default: %(default)s')
    sgroup.add_argument('--input_len_variability', type=int, default=2,
                        help='The range of input length variability for ' +
                             'the training sequences. For a value of 2 and a ' +
                             'mean input length of 5, the range of ' +
                             'input sequence lengths will lie in [5-2, 5+2]. ' +
                             'Default: %(default)s')
    sgroup.add_argument('--seq_width', type=int, default=7,
                        help='The width of the sequences (excluding the stop ' +
                             'flag). Default: %(default)s')
    sgroup.add_argument('--pat_len', type=int, default=-1,
                        help='The number of timesteps in the training copy ' +
                             'patterns that shouldnt be zeroed out. Note that '+
                             'this option does not change the length of the ' +
                             'inputs, only the actual duration of the ' +
                             'random patterns within the inputs, thus '+
                             'reducing the memory requirement. If no value is '+
                             'given, "pat_len" will equal the ' +
                             'length of each pattern. Default: %(default)s')
    sgroup.add_argument('--random_pad', action='store_true',
                        help='If active, the entire input sequence will ' +
                             'consist of random binary bits, and will not ' +
                             'be zero padded. Note that this only has an ' +
                             'effect if "pat_len"!=-1, and in this case the '+
                             'loss still takes into account the ' +
                             'reconstruction of an output pattern of length '+
                             '"pat_len".')
    sgroup.add_argument('--permute_width', action='store_true',
                        help='If enabled, the random patterns will be ' +
                             'permuted differently along the width direction '+
                             'for different tasks to obtain a continual ' +
                             'learning setting analogous to permuted MNIST. '+
                             'For this type of permutation, the temporal ' +
                             'memory requirements are not altered between ' +
                             'tasks. Note that this option requires that all ' +
                             'tasks have sequences with identical lengths (' +
                             '"input_len_variability=0" and "input_len_step'+
                             '=0"). The sequence lengths can then be set with '+
                             '"first_task_input_len".')
    sgroup.add_argument('--permute_time', action='store_true',
                        help='If enabled, the random patterns will be ' +
                             'permuted differently along the time direction '+
                             'for different tasks to obtain a continual ' +
                             'learning setting analogous to permuted MNIST. '+
                             'For this type of permutation, the temporal ' +
                             'memory requirements can be altered between ' +
                             'tasks. Note that this option requires that all ' +
                             'tasks have sequences with identical lengths (' +
                             '"input_len_variability=0" and "input_len_step'+
                             '=0"). The sequence lengths can then be set with '+
                             '"first_task_input_len".')
    sgroup.add_argument('--use_new_permuted_dhandler', action='store_true',
                        help='If enabled, the new datahandler for the ' +
                             'permuted copy tasks, ' +
                             '"data.timeseries.PermutedCopyList" will be ' +
                             'used. Else, the default '+ 
                             '"data.timeseries.CopyTask" is loaded.')
    sgroup.add_argument('--scatter_pattern', action='store_true',
                        help='If enabled, the output pattern will be made up ' +
                             'of randomly scattered timesteps from the input ' +
                             'sequence. "pat_len" needs to be specified in ' +
                             'this case and determines the size of the ' +
                             'output pattern.')
    sgroup.add_argument('--permute_xor', action='store_true',
                        help='If enabled, the output pattern will be given ' +
                             'by the logical xor of the input pattern and ' +
                             'a random permutation of the input pattern. ' +
                             'The nature of the random permutation is ' +
                             'controlled by the options "permute_width" and ' +
                             '"permute_time".')
    sgroup.add_argument('--permute_xor_iter',  type=int, default=1,
                        help='The number of times the permuted xor operation ' +
                             'is applied. Default: %(default)s')
    sgroup.add_argument('--permute_xor_separate', action='store_true',
                        help='If enabled, and "permute_xor_iter" is greater ' +
                             'than 1, then rather than applying the same ' +
                             'permutation iteratively, a different ' +
                             'permutation is applied at every iteration.')

def check_invalid_args_sequential(config):
    """Sanity check for some command-line arguments specific to training on 
    the copy task.

    Args:
        config (argparse.Namespace): Parsed command-line arguments.
    """
    if config.first_task_input_len <= 0:
        raise ValueError('"first_task_input_len" must be a strictly positive '+
                         'integer.')
    if config.input_len_step < 0:
        raise ValueError('"input_len_step" must be a positive integer.')
    if config.input_len_variability < 0:
        raise ValueError('"input_len_variability" must be a positive integer.')
    if config.seq_width <= 0:
        raise ValueError('"seq_width" must be a strictly positive integer.')
    if config.pat_len!=-1. and config.pat_len < 0:
        raise ValueError('"pat_len" must be a positive integer.')
    if config.permute_width or config.permute_time or config.scatter_pattern:
        # Note, these are design choices that we made for this set of tasks.
        # The code should not break if you deviate from these conditions.
        if config.input_len_variability != 0:
            warnings.warn('For permuted or scatter tasks, the lengths of the ' +
                'sequences has to be identical. "input_len_variability" will '+
                'automatically be set to zero.')
            config.input_len_variability = 0
        if config.input_len_step != 0:
            warnings.warn('For permuted or scatter tasks, the lengths of the ' +
                'sequences in different tasks has to be identical. '+
                '"input_len_step" will automatically be set to zero.')
            config.input_len_step = 0
    if not (config.permute_width or config.permute_time) and \
            hasattr(config, 'permute_xor') and config.permute_xor:
        raise ValueError('Option "permute_xor" only applicable if ' +
                         'permutations are used.')
    if config.scatter_pattern and config.pat_len == -1:
        raise ValueError('"scatter_pattern" is not compatible with "pat_len' +
            '==-1". Please provide a new "pat_len" to specify the length '+
            'of the output patterns.')
    if config.permute_xor_iter == 1 and config.permute_xor_separate:
        warnings.warn('Option "permute_xor_separate" doesn\'t have an effect ' +
                      'if "permute_xor_iter" is not greater than 1.')
    if config.random_pad and config.pat_len==-1.:
        warnings.warn('The option "random_pad" has no effect if "pat_len" '+
            'is equal to -1.')



if __name__=='__main__':
    pass