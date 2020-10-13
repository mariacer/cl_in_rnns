#!/usr/bin/env python3
# Copyright 2020 Christian Henning
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
# @title          :sequential/student_teacher/train_args_st.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :07/28/2020
# @version        :1.0
# @python_version :3.6.10
"""
CLI Argument Parsing for Sequential Student-Teacher Experiments
---------------------------------------------------------------

Command-line arguments and default values for the Student-Teacher experiments
are handled here.
"""
import argparse
import warnings

import utils.cli_args as cli
import sequential.train_args_sequential as seq

def parse_cmd_arguments(mode='student_teacher', default=False, argv=None):
    """Parse command-line arguments.

    Args:
        mode (str): The CLI mode of the experiment.
        default (optional): If ``True``, command-line arguments will be ignored
            and only the default values will be parsed.
        argv (optional): If provided, it will be treated as a list of command-
            line argument that is passed to the parser in place of sys.argv.

    Returns:
        The Namespace object containing argument names and values.
    """

    description = 'Continual learning in a Student-Teacher setting.'

    parser = argparse.ArgumentParser(description=description)

    dnum_tasks = 5
    dval_set_size = 100

    cli.cl_args(parser, show_beta=True, dbeta=0.01,
                show_from_scratch=True, show_multi_head=True,
                show_split_head_cl3=False, show_cl_scenario=False,
                show_num_tasks=True, dnum_tasks=dnum_tasks,
                show_num_classes_per_task=False)
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
                  dval_set_size=dval_set_size)
    magroup = cli.miscellaneous_args(parser, big_data=False,
        synthetic_data=True, show_plots=True, no_cuda=True,
        show_publication_style=False)
    seq.ewc_args(parser, dewc_lambda=1e5, dn_fisher=-1, dtbptt_fisher=-1,
        show_ts_weighting_fisher=False, dts_weighting_fisher='none')
    seq.si_args(parser, dsi_lambda=1.)
    seq.context_mod_args(parser, dsparsification_reg_type='l1', 
        dsparsification_reg_strength=1., dcontext_mod_init='constant')
    seq.miscellaneous_args(magroup, dmask_fraction=0.8, dclassification=False,
                           show_ts_weighting=False, dts_weighting='none',
                           show_use_ce_loss=False,
                           show_during_acc_criterion=False)
    # Replay arguments.
    rep_args = seq.replay_args(parser)
    cli.generator_args(rep_args, dlatent_dim=100)
    cli.main_net_args(parser, allowed_nets=['simple_rnn'],
        dsrnn_rec_layers='256', dsrnn_pre_fc_layers='',
        dsrnn_post_fc_layers='',
        show_net_act=True, dnet_act='tanh', show_no_bias=True,
        show_dropout_rate=False, show_specnorm=False, show_batchnorm=False,
        prefix='dec_', pf_name='replay decoder')

    # Dataset specific arguments.
    student_teacher_args(parser)

    args = None
    if argv is not None:
        if default:
            warnings.warn('Provided "argv" will be ignored since "default" ' +
                          'option was turned on.')
        args = argv
    if default:
        args = []
    config = parser.parse_args(args=args)
    config.mode = mode

    ### Check argument values!
    cli.check_invalid_argument_usage(config)
    seq.check_invalid_args_sequential(config)
    check_invalid_args(config)

    if config.train_from_scratch:
        # FIXME We could get rid of this warning by properly checkpointing and
        # loading all networks.
        warnings.warn('When training from scratch, only during metrics ' +
                      'make sense. All final metrics should be ignored!')

    return config

def student_teacher_args(parser):
    """This is a helper function of function :func:`parse_cmd_arguments` to add
    specific arguments to the argument group related to the student-teacher
    dataset.

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
    """
    heading = 'Student-Teacher Dataset'

    sgroup = parser.add_argument_group(heading)
    sgroup.add_argument('--input_feature_dim', type=int, default=7,
                        help='Feature dimensionality of inputs. ' +
                             'Default: %(default)s.')
    sgroup.add_argument('--output_feature_dim', type=int, default=7,
                        help='Feature dimensionality of outputs. ' +
                             'Default: %(default)s.')
    sgroup.add_argument('--orth_teacher_hid_mat', action='store_true',
                        help='Whether the teacher RNNs should have ' +
                             'orthogonal hidden-to-hidden matrices.')
    sgroup.add_argument('--rank_teacher_hid_mat', type=int, default=-1,
                        help='The rank of the hidden-to-hidden matrices used ' +
                             'by the teacher RNNs. If "-1", then no rank ' +
                             'constraint will be applied. ' +
                             'Default: %(default)s.')
    sgroup.add_argument('--max_sv_teacher_hid_mat', type=float, default=-1,
                        help='The maximum singular value of the teacher\'s ' +
                             'hidden-to-hidden matrices. If "-1", then no ' +
                             'singular value constraint will be applied. ' +
                             'Default: %(default)s.')
    sgroup.add_argument('--num_timesteps_in', type=int, default=10,
                        help='The number of timesteps in input sequences. ' +
                             'Default: %(default)s.')
    sgroup.add_argument('--num_timesteps_out', type=int, default=-1,
                        help='The number of timesteps in output sequences. ' +
                             'If "-1", then value of "num_timesteps_in" will ' +
                             'be used. Otherwise, the value has to be ' +
                             'greater or equal to "num_timesteps_in". ' +
                             'Default: %(default)s.')

def check_invalid_args(config):
    """Sanity check for some command-line arguments specific to training in
    the student-teacher setting.

    Args:
        config (argparse.Namespace): Parsed command-line arguments.
    """
    if config.orth_teacher_hid_mat:
        if config.rank_teacher_hid_mat != -1:
            raise ValueError('Options "orth_teacher_hid_mat" and ' +
                             '"rank_teacher_hid_mat" are not compatible.')
        if config.max_sv_teacher_hid_mat != -1:
            raise ValueError('Options "orth_teacher_hid_mat" and ' +
                             '"max_sv_teacher_hid_mat" are not compatible.')
    if config.num_timesteps_out != -1:
        if config.num_timesteps_out < config.num_timesteps_in:
            raise ValueError('Option "num_timesteps_out" has to be greater ' +
                             'or equal to "num_timesteps_in".')

if __name__ == '__main__':
    pass


