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
# @title          :sequential/smnist/train_args_seq_smnist.py
# @author         :be
# @contact        :behret@ethz.ch
# @created        :24/03/2020
# @version        :1.0
# @python_version :3.6.8
"""
Command-line arguments and default values for the sequential SMNIST task are 
handled here.
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

    description = 'Continual learning on sequential SMNIST task.'

    parser = argparse.ArgumentParser(description=description)

    cli.cl_args(parser, show_beta=True, dbeta=0.005,
                show_from_scratch=True, show_multi_head=True,
                show_split_head_cl3=False, show_cl_scenario=False,
                show_num_tasks=True, dnum_tasks=45)
    cli.train_args(parser, show_lr=True, show_epochs=False,
        dbatch_size=64, dn_iter=5000,
        dlr=1e-3, show_clip_grad_value=False, show_clip_grad_norm=True,
        show_momentum=False, show_adam_beta1=True)
    seq.rnn_args(parser, drnn_arch='256', dnet_act='tanh')
    cli.hypernet_args(parser, dhyper_chunks=-1, dhnet_arch='50,50',
                          dtemb_size=32, demb_size=32, dhnet_act='relu')
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
    cli.eval_args(parser, dval_iter=250, show_val_set_size=True,
                  dval_set_size=1000)
    magroup = cli.miscellaneous_args(parser, big_data=False,
        synthetic_data=True, show_plots=True, no_cuda=True,
        show_publication_style=False)
    seq.ewc_args(parser, dewc_lambda=5000., dn_fisher=-1, dtbptt_fisher=-1,
        dts_weighting_fisher='last')
    seq.si_args(parser, dsi_lambda=1.)
    seq.context_mod_args(parser, dsparsification_reg_type='l1',
        dsparsification_reg_strength=1., dcontext_mod_init='constant')
    seq.miscellaneous_args(magroup, dmask_fraction=0.8, dclassification=True,
                           dts_weighting='last', show_use_ce_loss=False,
                           show_early_stopping_thld=True)
    # Replay arguments.
    rep_args = seq.replay_args(parser)
    cli.generator_args(rep_args, dlatent_dim=100)
    cli.main_net_args(parser, allowed_nets=['simple_rnn'],
        dsrnn_rec_layers='256', dsrnn_pre_fc_layers='',
        dsrnn_post_fc_layers='',
        show_net_act=True, dnet_act='tanh', show_no_bias=True,
        show_dropout_rate=False, show_specnorm=False, show_batchnorm=False,
        prefix='dec_', pf_name='replay decoder')

    seq_args(parser)

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

    if config.train_from_scratch:
        # FIXME We could get rid of this warning by properly checkpointing and
        # loading all networks.
        warnings.warn('When training from scratch, only during accuracies ' +
                      'make sense. All other outputs should be ignored!')

    return config


def seq_args(parser):
    """This is a helper function of function :func:`parse_cmd_arguments` to add
    specific arguments to the argument group related to seq smnist task.

    Arguments specified in this function:
        - `ssmnist_seq_len` 


    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
    """
    heading = 'SSMNIST options'

    sgroup = parser.add_argument_group(heading)
    sgroup.add_argument('--ssmnist_seq_len', type=int, default=2,
                        help='The number of digits used in a sequence. ' +
                             'Default: %(default)s.')
    sgroup.add_argument('--ssmnist_two_classes', action='store_true',
                        help='If used, every task will have only 2 classes. ' +
                             'Instead of classifying every possible sequence ' +
                             'individually, sequences are randomly grouped ' +
                             'into 2 classes.')

if __name__=='__main__':
    pass