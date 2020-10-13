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
# @title          :sequential/train_args_sequential.py
# @author         :mc, be
# @contact        :mariacer@ethz.ch
# @created        :23/03/2020
# @version        :1.0
# @python_version :3.6.8
"""
Common CLI arguments for experiments with sequential data
---------------------------------------------------------

All command-line arguments and default values for this subpackage are handled
in this module.
"""
import warnings

def miscellaneous_args(agroup, dclassification=False, dbeta_fixation=0.5,
                       dmask_fraction=0.8, show_ts_weighting=True,
                       dts_weighting='unpadded', show_use_ce_loss=True,
                       show_early_stopping_thld=False, dearly_stopping_thld=-1,
                       des_warm_up_iter=5000, des_best_val_diff=.01,
                       show_during_acc_criterion=True):
    """This is a helper function of function :func:`parse_cmd_arguments` to add
    specific arguments to the argument group for miscellaneous options.

    Arguments specified in this function:
        - `classification`
        - `store_activations`
        - `multitask`
        - `train_only_heads_after_first`
        - `dont_train_heads`
        - `train_tnet_once`
        - `use_ce_loss`
        - `beta_fixation`
        - `reinit_hnet`
        - `input_task_identity`
        - `use_masks`
        - `mask_fraction`
        - `hnet_all`
        - `last_task_only`
        - `ts_weighting`
        - `early_stopping_thld`

    Args:
        agroup: The argument group returned by function
            :func:`utils.cli_args.miscellaneous_args`.
        dclassification (bool): Default value for option `classification`.
        dbeta_fixation (float): Default value for option `beta_fixation`.
        dmask_fraction (float): Default value for option `mask_fraction`.
        show_ts_weighting (bool): Whether argument `ts_weighting` should be
            shown.
        dts_weighting (str): Default value for option `ts_weighting`.
        show_use_ce_loss (bool): Whether argument `use_ce_loss` and
            `beta_fixation` should be shown.
        show_early_stopping_thld (bool): Whether argument `early_stopping_thld`
            should be shown.
        dearly_stopping_thld (str): Default value for option
            `early_stopping_thld`.
    """
    ### Main network options.
    agroup.add_argument('--classification', type=bool, default=dclassification,
                        help='Whether the task is a classification task. '+
                             'Default: %(default)s')
    agroup.add_argument('--store_activations', action='store_true',
                        help='Whether the activations of the hidden neurons ' +
                             'in the RNN should be stored.')
    agroup.add_argument('--multitask', action='store_true',
                        help='Train the network in a multitask fashion, i.e. ' +
                             'with data from all tasks presented ' +
                             'simultaneously.')
    agroup.add_argument('--train_only_heads_after_first', action='store_true',
                        help='If True, only the final fully-connected layer ' +
                             'will be trained for all tasks after the first ' +
                             'one.')
    agroup.add_argument('--dont_train_heads', action='store_true',
                        help='If True, the output heads will not be trained.')
    agroup.add_argument('--train_tnet_once', action='store_true',
                        help='Train the target network only when training ' +
                             'the first task and keep the weights fixed ' +
                             'afterwards (thus, learning can only affected ' +
                             'modulatory patterns after the first task).')
    if show_use_ce_loss:
        agroup.add_argument('--use_ce_loss', action='store_true',
                            help='Use the cross-entropy loss rather than ' +
                                 'the MSE loss for training.')
        agroup.add_argument('--beta_fixation', type=float,
                            default=dbeta_fixation,
                            help='When using the cross entropy loss, the ' +
                                 'softmax its inverse temperature can be set ' +
                                 'differently for the fixation period ' +
                                 '(typically smaller than 1. ' +
                                 'Default: %(default)s')
    agroup.add_argument('--reinit_tnet', action='store_true',
                        help='Reinitialize the target network before every ' +
                             'task. Note, if using EWC, the network is ' +
                             'anyway pulled back to the old solution.')
    agroup.add_argument('--input_task_identity', action='store_true',
                        help='Provide a one-hot-encoding of the task identity '+
                             'with the input pattern.')
    agroup.add_argument('--use_masks', action='store_true',
                        help='Uses binary masks for context_mod instead of ' +
                             'gains and shifts produced by the hnet.')
    agroup.add_argument('--mask_fraction', type=float, default=dmask_fraction,
                        help='Fraction of units that will be masked when ' +
                             'using the option use_masks. Default: %(default)s')
    agroup.add_argument('--hnet_all', action='store_true',
                        help='If enabled, then all target network weights ' +
                             'will come from the hypernet. Hence, there is ' +
                             'no need for continual learning in the target ' +
                             'network (e.g., via EWC or SI). Instead, ' +
                             'forgetting is prevented via the hnet ' +
                             'regularizer.')
    agroup.add_argument('--calc_hnet_reg_targets_online',
                        action='store_true',
                        help='For our hypernet CL regularizer, this ' +
                             'option will ensure that the targets are ' +
                             'computed on the fly, using the hypernet ' +
                             'weights acquired after learning the ' +
                             'previous task. Note, this option ensures ' +
                             'that there is almost no memory grow with ' +
                             'an increasing number of tasks (except ' +
                             'from an increasing number of task ' +
                             'embeddings). If this option is ' +
                             'deactivated, the more computationally ' +
                             'efficient way is chosen of computing all ' +
                             'main network weight targets (from all ' +
                             'previous tasks) ones before learning a new ' +
                             'task.')
    agroup.add_argument('--hnet_reg_batch_size', type=int, default=-1,
                        metavar='N',
                        help='If not "-1", then this number will ' +
                             'determine the maximum number of previous ' +
                             'tasks that are are considered when ' +
                             'computing the regularizer. Hence, if the ' +
                             'number of previous tasks is greater than ' 
                             'this number, then the regularizer will be ' +
                             'computed only over a random subset of ' +
                             'previous tasks. Default: %(default)s.')
    agroup.add_argument('--last_task_only', action='store_true',
                        help='If activated, the system will only be trained '+
                             'on last task. This is meaningful in the copy' +
                             'task, specially for hyperparameter search since '+
                             'networks that solve well the last task (which ' +
                             'has the longest sequences) will probably '+
                             'solve well the other tasks.')
    if show_ts_weighting:
        agroup.add_argument('--ts_weighting', type=str, default=dts_weighting,
                            choices=['none', 'last', 'last_ten_percent',
                                     'unpadded', 'discount'],
                            help='Weight given to the timesteps when ' +
                                 'computing the loss. Value "none" refers to ' +
                                 'no weighting, "last" refers to only taking ' +
                                 'the last (unpadded) timestep into account, ' +
                                 '"last_ten_percent" only looks at the last '+
                                 '10 percent of (unpadded) timesteps, ' +
                                 '"unpadded" looks at all timesteps except ' +
                                 'the padded ones and "discount" means that' +
                                 'timesteps will be exponentially discounted' +
                                 ' the earlier they occur. ' +
                                 'Default: %(default)s.')
    if show_early_stopping_thld:
        agroup.add_argument('--early_stopping_thld', type=float,
                            default=dearly_stopping_thld,
                        help='Gradient threshold for early stopping. If "-1" ' +
                             'no early stopping will be applied. Otherwise, ' +
                             'a validation measure will be tracked over time ' +
                             'and a straight line will be fit through all ' +
                             'the measures taken so far (where the weight of ' +
                             'old measures is exponentially decaying in this ' +
                             'fit). If the absolute slope of this line is ' +
                             'smaller than the given threshold and it is ' +
                             'roughly the best value seen so far, an early ' +
                             'stopping will be invoked. Default: %(default)s')
        agroup.add_argument('--es_warm_up_iter', type=int,
                            default=des_warm_up_iter,
                            help='If early stopping is used (see ' +
                                 '"early_stopping_thld"), then this option ' +
                                 'defines the number of iterations before ' +
                                 'the early stopping criterion may be ' +
                                 'invoked. Default: %(default)s')
        agroup.add_argument('--es_best_val_diff', type=float,
                            default=des_best_val_diff,
                            help='If early stopping is used (see ' +
                                 '"early_stopping_thld"), then this option ' +
                                 'defines the absolute difference to the ' +
                                 'best validation value seen so far, such ' +
                                 'early stopping might be invoked if the ' +
                                 'slope criterion is met. Default: %(default)s')
    agroup.add_argument('--orthogonal_hh_init', action='store_true',
                        help='Initialize hidden-to-hidden weights of ' +
                             'recurrent layers orthogonally.')
    agroup.add_argument('--orthogonal_hh_reg', type=float, default=-1,
                        help='If "-1", no orthogonal regularization will ' +
                             'be applied. Otherwise, the hidden-to-' +
                             'hidden weights of the recurrent layers are ' +
                             'regularized to be orthogonal with the ' +
                             'given regularization strength. ' +
                             'Default: %(default)s')
    agroup.add_argument('--store_final_models', action='store_true',
                        help='Save the final models into the output folder. ' +
                             'This option might come in handy when doing ' +
                             'post-hoc analysis.')
    agroup.add_argument('--store_during_models', action='store_true',
                        help='Checkpoint the models after training on each ' +
                             'task into the output folder.')
    agroup.add_argument('--use_best_models', action='store_true',
                        help='If activated, the models are checkpointed  ' +
                             'during training whenever the validation ' +
                             'performance improves (see also "val_iter"). ' +
                             'After training, the best model is restored ' +
                             'rather than using the final model. Note, this ' +
                             'is sometimes termed "early stopping", even ' +
                             'though it doesn\'t reduce the actual training ' +
                             'time. Also note, that the validation ' +
                             'performance may not measure the convergence of ' +
                             'regularizers.')
    if show_during_acc_criterion:
        agroup.add_argument('--during_acc_criterion', type=str, default='-1',
                        help='If "-1", the criterion is deactivated. ' +
                             'Otherwise, a list of comma-separated numbers ' +
                             'representing accuracies (between 0 - 100) is ' +
                             'expected. A run will be stopped if the during ' +
                             'accuracy of any task (except the last one) is ' +
                             'smaller than this value. Hence, this is an ' +
                             'easy way to avoid wasting ressources during ' +
                             'hyperparameter search. Note, the list should ' +
                             'either contain a single number or ' +
                             '"num_tasks-1" numbers. A value of "-1" would ' +
                             'deactivate the criterion for a task. ' +
                             'Default: %(default)s')

def rnn_args(parser, drnn_arch='256', dnet_act='tanh',
             show_use_bidirectional_net=False):
    """This is a helper function of function :func:`parse_cmd_arguments` to add
    an argument group for options to a main network.

    Arguments specified in this function:
        - `rnn_arch`
        - `srnn_pre_fc_layers`
        - `srnn_post_fc_layers`
        - `net_act`
        - `use_vanilla_rnn`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
        drnn_arch: Default value of option `rnn_arch`.
        dnet_act: Default value of option `net_act`.
        show_use_bidirectional_net (bool): Whether option
            `show_use_bidirectional_net` should be shown.

    Returns:
        The created argument group, in case more options should be added.
    """
    heading = 'Main recurrent network options'

    ### Main network options.
    # FIXME We should instead use `cli.main_net_args`. However, we already
    # collected a lot of results using these command-line arguments.
    agroup = parser.add_argument_group(heading)
    agroup.add_argument('--rnn_arch', type=str, default=drnn_arch,
                        help='Specifies the dimension of the hidden' +
                             '(recurrent) layer of the recurrent network.' +
                             'Default: %(default)s.')
    agroup.add_argument('--srnn_pre_fc_layers', type=str,
                        default='',
                        help='If using a "simple_rnn" network, ' +
                             'this will specify the sizes of all initial ' +
                             'fully-connected latyers. If left empty, ' +
                             'there will be no initial fully-connected ' +
                             'layers and the first layer is going to be ' +
                             'a recurrent layer. Default: %(default)s.')
    agroup.add_argument('--srnn_post_fc_layers', type=str, default='',
                        help='If using a "simple_rnn" network, ' +
                             'this will specify the sizes of all final ' +
                             'hidden fully-connected layers. Note, the ' +
                             'output layer is also fully-connected, even ' +
                             'if this option is left empty. ' +
                             'Default: %(default)s.')
    agroup.add_argument('--net_act', type=str, default=dnet_act,
                        help='Activation function used in the network. ' +
                             'Default: %(default)s.',
                        choices=['relu', 'tanh', 'linear'])
    agroup.add_argument('--use_vanilla_rnn', action='store_true',
                        help='Whether vanilla rnn cells should be used. ' +
                             'Otherwise, LSTM cells are used.')
    if show_use_bidirectional_net:
        agroup.add_argument('--use_bidirectional_net', action='store_true',
                            help='If set, a bidirectional LSTM or RNN (if ' +
                                 '"use_vanilla_rnn" is used) will be used.')

    return agroup

def new_hnet_args(agroup):
    """This is a helper function of function :func:`parse_cmd_arguments` to add
    arguments to the argument group that characterizes the new hypernetworks.

    Args:
        agroup: The argument group returned by function
            :func:`utils.cli_args.hnet_args`.

    Returns:
        The created argument group, in case more options should be added.
    """
    agroup.add_argument('--nh_shmlp_chunk_sizes', type=str, default=8,
                        help='Only applicable to "structured_hmlp" ' +
                             'hypernetwork. A comma-separated list of ' +
                             'integers, denoting the size of the chunks into ' +
                             'which layers are split.Default: %(default)s.')
    agroup.add_argument('--nh_shmlp_chunk_fc_layers', action='store_true',
                        help='Is a "structured_hmlp" hypernetwork is used, ' +
                             'then this option decides whether fully-' +
                             'connected layers should be chunked as well.')
    agroup.add_argument('--nh_separate_out_head', action='store_true',
                        help='By default, if "--hnet_all" is used, all main ' +
                             'network weights (incl. output head weights) ' +
                             'originate from a conditional hypernetwork. If ' +
                             'this option is activated, then the output head ' +
                             'weights will be task-specific and do not ' +
                             'originate from a shared hypernetwork. Note, if ' +
                             'activated, the main network output will always ' +
                             'correspond to a multihead setting.')
    agroup.add_argument('--use_new_hnet', action='store_true',
                        help='Whether one of the new hypernet ' +
                             'implementations should be used.')

    return agroup

def ewc_args(parser, dewc_lambda=5000., dn_fisher=-1, dtbptt_fisher=-1,
             show_ts_weighting_fisher=True, dts_weighting_fisher='unpadded'):
    """This is a helper function of function :func:`parse_cmd_arguments` to add
    an argument group for options regarding EWC.

    Arguments specified in this function:
        - `use_ewc`
        - `ewc_gamma`
        - `ewc_lambda`
        - `n_fisher`
        - `tbptt_fisher`
        - `ts_weighting_fisher`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
        dewc_lambda (float): Default value of option `ewc_lambda`.
        dn_fisher (int): Default value of option `n_fisher`.
        dtbptt_fisher (int): Default value of option `tbptt_fisher`.
        show_ts_weighting_fisher (bool): Whether argument `ts_weighting_fisher`
            should be shown.
        dts_weighting_fisher (str): Default value of option 
            `ts_weighting_fisher`.

    Returns:
        The created argument group, in case more options should be added.
    """
    agroup = parser.add_argument_group('EWC options')

    agroup.add_argument('--use_ewc', action='store_true',
                         help='If True, ewc will be used.')
    agroup.add_argument('--ewc_gamma', type=float, default=1.,
                         help='Reg strength of Fisher steps in (online) ewc. ' +
                              'Default: %(default)s')
    agroup.add_argument('--ewc_lambda', type=float, default=dewc_lambda,
                         help='Reg strength of (online) ewc. ' +
                              'Default: %(default)s.')
    agroup.add_argument('--n_fisher', type=int, default=dn_fisher,
                        help='Number of training samples to be used for the ' +
                             'estimation of the diagonal Fisher elements. If ' +
                             '"-1", all training samples are used. ' +
                             'Default: %(default)s.')
    agroup.add_argument('--tbptt_fisher', type=int, default=dtbptt_fisher,
                        help='In case truncated BPTT should be used when ' +
                             'computing importance weights using the Fisher ' +
                             'matrix, one can set the number of timesteps to ' +
                             'backprop backwards via this option. "-1" ' +
                             'corresponds to backpropagating through all ' +
                             'timesteps. Default: %(default)s.')
    if show_ts_weighting_fisher:
        agroup.add_argument('--ts_weighting_fisher', type=str,
                            default=dts_weighting_fisher,
                            help='Whether a weighting of the log-likelihood ' +
                                 'of timesteps should be applied when ' +
                                 'computing the Fisher elements. See option ' +
                                 '"ts_weighting" for more details. ' +
                                 'Default: %(default)s.',
                            choices=['none', 'last', 'last_ten_percent',
                                     'unpadded', 'discount'])

    return agroup

def si_args(parser, dsi_lambda=1.):
    """This is a helper function of function :func:`parse_cmd_arguments` to add
    an argument group for options regarding Synaptic Intelligence (SI).

    Args:
        parser (argparse.ArgumentParser): The argument parser to which the
            group should be added.
        dsi_lambda (float): Default value of option `si_lambda`.

    Returns:
        The created argument group, in case more options should be added.
    """
    agroup = parser.add_argument_group('SI options')

    agroup.add_argument('--use_si', action='store_true',
                         help='Use Synaptic Intelligence for the target ' +
                              'network weights.')
    agroup.add_argument('--si_lambda', type=float, default=dsi_lambda,
                         help='Regularization strength for synaptic ' +
                              'intelligence. Default: %(default)s')
    agroup.add_argument('--si_task_loss_only', action='store_true',
                         help='If enabled, synaptic intelligence would ' +
                              'estimate importances based on the task-' +
                              'specific loss only, rather than the total ' +
                              'loss, which incorporates regularizers.')

    return agroup

def context_mod_args(parser, dsparsification_reg_type='l1', 
        dsparsification_reg_strength=1., dcontext_mod_init='constant'):
    """This is a helper function of function :func:`parse_cmd_arguments` to add
    an argument group for options regarding the context modulation.

    Arguments specified in this function:
        - `use_context_mod`
        - `no_context_mod_outputs`
        - `context_mod_inputs`
        - `context_mod_post_activation`
        - `context_mod_last_step`
        - `checkpoint_context_mod`
        - `offset_gains`
        - `dont_softplus_gains`
        - `sparsify_context_mod`
        - `sparsification_reg_strength`

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
        dsparsification_reg_type (str): Default value of option 
            `sparsification_reg_type`.
        dsparsification_reg_strength (float): Default value of option
            `sparsification_reg_strength`.
        dcontext_mod_init (str): Default value of option `context_mod_init`.

    Returns:
        The created argument group, in case more options should be added.
    """

    heading = 'Context modulation options'
    agroup = parser.add_argument_group(heading)

    # here we dont differenciate between the usage of hnet and context mod
    # since we assume we only use the cm case
    agroup.add_argument('--use_context_mod', action='store_true',
                        help='If True, hnet-based context-mod will be used.')
    agroup.add_argument('--no_context_mod_outputs', action='store_true',
                        help='If True, context modulation will not be ' +
                             'applied to the output layer.')
    agroup.add_argument('--context_mod_inputs', action='store_true',
                        help='If True, context modulation will be applied ' +
                             'to the input layer.')
    agroup.add_argument('--context_mod_post_activation', action='store_true',
                        help='If True, context modulation will be applied ' +
                             'after computing the activation function, ' +
                             'else, it will be applied before.')
    agroup.add_argument('--context_mod_last_step', action='store_true',
                        help='If True, context modulation will only be ' +
                             'applied in the last timestep of the sequence. ' +
                             'Else, it is applied at every timestep.')
    agroup.add_argument('--checkpoint_context_mod', action='store_true',
                        help='Train context-modulation without a ' +
                             'hypernetwork. Instead, context-mod weights ' +
                             'will be part of the main network and will be ' +
                             'checkpointed after every task (linear memory ' +
                             'growth).')
    agroup.add_argument('--context_mod_init', type=str,
                        default=dcontext_mod_init,
                        help='What method to use to initialize context-' +
                             'modulation weights. Note, this option is only ' +
                             'applicable in combination with ' +
                             '"checkpoint_context_mod". Reinitialization ' +
                             'will be performed after every task. If ' +
                             '"sparse" is used, then the option ' +
                             '"mask_fraction" is reused to determine the ' +
                             'sparsity level. Default: %(default)s.',
                        choices=['constant', 'normal', 'uniform', 'sparse'])
    agroup.add_argument('--offset_gains', action='store_true',
                        help='If this option is activated, the modulatory ' +
                             'gains produced by the hypernetwork will be ' +
                             'shifted by 1. Note, requires ' +
                             '"dont_softplus_gain" to be set.')
    agroup.add_argument('--dont_softplus_gains', action='store_true',
                        help='If this option is activated, the modulatory ' +
                             'gains produced by the hypernetwork will not ' +
                             'be send through a softplus. Therefore, they ' +
                             'might be positive and negative.')
    agroup.add_argument('--context_mod_per_ts', action='store_true',
                        help='If True, a different context-mod pattern per ' +
                             'timestep will be learned.')
    agroup.add_argument('--sparsify_context_mod', action='store_true',
                        help='If this option is activated, the modulatory ' +
                             'gains are pushed towards zero to sparsify the ' +
                             'backpropagation through neurons in order to ' +
                             'restrict the absolute model capacity used per ' +
                             'task as well to ensure that there are ' +
                             '"unimportant" weights reserved for future tasks.')
    agroup.add_argument('--sparsification_reg_strength', type=float,
                        default=dsparsification_reg_strength,
                        help='The strength of the gain sparsification ' +
                             'regularizer. Default: %(default)s.')
    agroup.add_argument('--sparsification_reg_type', type=str,
                        default=dsparsification_reg_type,
                        help='The type of regularizer to be used in order to ' +
                             'obtain sparse gain patterns. ' +
                             'Default: %(default)s.',
                        choices=['l1', 'log'])

    return agroup

def replay_args(parser, show_all_task_softmax=True):
    """This is a helper function of function :func:`parse_cmd_arguments` to add
    an argument group for options regarding Generative Replay.

    Args:
        parser (argparse.ArgumentParser): The argument parser to which the
            group should be added.
        show_all_task_softmax (bool): Whether option `all_task_softmax` should
            be shown.

            Note:
                Only sensible for classification tasks.

    Returns:
        The created argument group, in case more options should be added.
    """
    agroup = parser.add_argument_group('Replay options')

    agroup.add_argument('--use_replay', action='store_true',
                         help='Whether generative replay should be used as a ' +
                              'method to prevent catastrophic forgetting. ' +
                              'If option "hnet_all" is enabled, a replay ' +
                              'model per task is learned within one task-' +
                              'conditioned hypernetwork. Otherwise, ' +
                              'forgetting within the replay model is ' +
                              'prevented by replaying data from a ' +
                              'checkpointed replay model before starting to ' +
                              'learn a new task.')
    if show_all_task_softmax:
        # This enables what we call CL3.
        agroup.add_argument('--all_task_softmax', action='store_true',
                             help='If enabled, the replay model is used to ' +
                                  'train a multi-head classifier whose ' +
                                  'softmax predictions are computed across ' +
                                  'all output heads.')
    agroup.add_argument('--replay_pm_strength', type=float, default=1.,
                        help='The strength of the prior-matching term if ' +
                             'replay is used. Default: %(default)s.')
    agroup.add_argument('--replay_rec_strength', type=float, default=1.,
                        help='The strength of the reconstruction term if ' +
                             'replay is used. Default: %(default)s.')
    agroup.add_argument('--replay_distill_reg', type=float, default=1.,
                        help='The strength of the soft-target distillation ' +
                             'loss if replay is used. Default: %(default)s.')
    agroup.add_argument('--replay_true_data', action='store_true',
                         help='This is a sanity check. If enabled, actual ' +
                              'data from previous tasks will be replayed. ' +
                              'The autoencoder is still trained, even though ' +
                              'the decoder has no influence in training the ' +
                              'target model.')
    agroup.add_argument('--coreset_size', type=int, default=-1, metavar='N',
                        help='This option is only valid in combination with ' +
                             'option "use_replay". If "-1", then coresets ' +
                             'are deactivated. Otherwise, a positive integer ' +
                             'is expected denoting the size of a coreset per ' +
                             'task. In this case, no decoder (VAE) will be ' +
                             'trained. Instead, data from the coreset is ' +
                             'replayed. Default: %(default)s.')

    return agroup

def check_invalid_args_sequential(config):
    """Sanity check for some command-line arguments specific to training on
    sequential tasks.

    Args:
        config (argparse.Namespace): Parsed command-line arguments.
    """

    if config.train_from_scratch:
        if config.multitask:
            raise ValueError('"Training from scratch" not applicable to ' +
                             'multi-task training.')
        if config.use_ewc:
            raise ValueError('It doesn\'t make sense to use EWC when ' +
                             'training from scratch.')
        if config.use_si:
            raise ValueError('It doesn\'t make sense to use SI when ' +
                             'training from scratch.')
        if config.beta > 0:
            warnings.warn('The hypernetwork-regularizer will be disabled ' +
                          'when training from scratch. So, "beta" is ignored.')
        if config.use_context_mod and config.checkpoint_context_mod:
            raise ValueError('Since all networks are trained from scratch ' +
                             '(including context-mod parameters), ' +
                             'checkpointing them doesn\'t make sense.')
        if config.reinit_tnet:
            raise ValueError('When networks are trained from scratch, they ' +
                             'are anyway reinitialized.')
        if config.store_final_models:
            warnings.warn('Storing only the final model when training ' +
                          'from scratch does not make sense, as all ' +
                          'information about previous tasks is lost. ' +
                          'Use "store_during_models" instead.')

    if config.multitask:
        if config.use_ewc:
            raise ValueError('Doesn\'t make sense to use EWC when training in ' +
                             'a multi-task setting.')
        if config.use_si:
            raise ValueError('Doesn\'t make sense to use SI when training in ' +
                             'a multi-task setting.')
        if config.use_context_mod and config.checkpoint_context_mod:
            raise ValueError('Since all tasks are trained in parallel, ' +
                             'checkpointing them doesn\'t make sense.')
        if config.beta > 0:
            warnings.warn('The hypernetwork-regularizer will be disabled ' +
                          'when training in a multi-task setting. So, ' +
                          '"beta" is ignored.')
        if config.reinit_tnet:
            raise ValueError('Networks can\'t be reinitialized during ' +
                             'training from scratch.')
        if config.store_during_models:
            warnings.warn('There are no during models to be stored when ' +
                             'doing multitask learning. ' +
                             'Use "store_final_models" instead.')
        if hasattr(config, 'during_acc_criterion') and \
                config.during_acc_criterion != '-1':
            warnings.warn('Option "during_acc_criterion" is not ' +
                             'compatible with multitask training and will ' +
                             'be ignored.')

    if config.use_ewc and config.use_si:
        raise ValueError('Cannot use EWC and SI at the same time.')

    if config.use_ewc and config.ewc_lambda == 0 or \
            config.use_si and config.si_lambda == 0:
        warnings.warn('EWC (or SI) was requested but the regularization ' +
                      ' strength isset to zero, such that it will ' +
                      'essentially be ignored from the loss.')

    if config.train_tnet_once:
        if config.use_ewc or config.use_si:
            raise ValueError('EWC (or SI) regularization cannot be used if ' +
                             'target network weights are not learned after ' +
                             'the first task (there is nothing to be ' +
                             'protected).')
        if config.use_masks:
            # Note, error will be thrown below, since context-mod is not used.
            warnings.warn('It doesn\'t make sense to use masks when only ' +
                          'training the first task, because later tasks ' +
                          'might utilize completely untrained parts of the ' +
                          'target network.')
        if not config.use_context_mod:
            raise ValueError('Context modulation has to be used if training ' +
                             'only the first task. Otherwise, there won\'t ' +
                             'be trainable weights for later tasks.')
        if config.multitask:
            raise ValueError('Can\'t only train on one task if multi-task ' +
                             'learning is activated.')
        if config.reinit_tnet:
            warnings.warn('Target net is only trained during first task and ' +
                          'then always reinitialized without training.')

    if hasattr(config, 'use_ce_loss') and config.use_ce_loss:
        assert config.beta_fixation >= 0

    if not config.use_si and config.si_task_loss_only:
        warnings.warn('Option "si_task_loss_only" has no effect if SI is not ' +
                      'used.')

    if config.use_masks:
        if config.use_context_mod:
            raise ValueError('You can\'t use context-modulation and masking ' +
                             'at the same time.')

        if config.context_mod_inputs:
            raise ValueError('Masking cannot be applied if inputs use ' +
                             'context-modulation (otherwise some inputs are ' +
                             'permanently switched off).')

        if not config.no_context_mod_outputs:
            warnings.warn('Masking cannot be applied if outputs use ' +
                          'context-modulation (otherwise some outputs are ' +
                          'permanently switched off). Setting '+
                          '"no_context_mod_outputs" to True.')
            config.no_context_mod_outputs = True

        if not config.dont_softplus_gains:
            warnings.warn('Masking cannot be applied if gains are modified ' +
                          'through a softplus. Setting "dont_softplus_gains" '+
                          'to True.')
            config.dont_softplus_gains = True

    if config.use_context_mod:
        if config.offset_gains and not config.dont_softplus_gains:
            raise ValueError('Option "offset_gains" requires ' +
                             '"dont_softplus_gains" to be set.')

    if not config.use_context_mod:
        if config.sparsify_context_mod:
            raise ValueError('Option "sparsify_context_mod" can only be used ' +
                             'if context-modulation is activated.')

    if config.hnet_all:
        if config.use_masks:
            raise ValueError('Option "hnet_all" is not compatible with ' +
                             'option "use_masks".')
        if config.train_tnet_once:
            raise ValueError('Option "hnet_all" is not compatible with ' +
                             'option "train_tnet_once".')
        if config.reinit_tnet:
            raise ValueError('Option "hnet_all" is not compatible with ' +
                             'option "reinit_tnet".')
        if config.use_ewc:
            raise ValueError('Option "hnet_all" is not compatible with ' +
                             'option "use_ewc".')
        if config.use_si:
            raise ValueError('Option "hnet_all" is not compatible with ' +
                             'option "use_si".')
        if config.checkpoint_context_mod:
            raise ValueError('Option "hnet_all" is not compatible with ' +
                             'option "checkpoint_context_mod".')
        if config.orthogonal_hh_init and \
                (not hasattr(config, 'use_replay') or not config.use_replay):
            warnings.warn('Option "orthogonal_hh_init" has no effect if a ' +
                          'hypernetwork produces the hidden-to-hidden weights.')
    else:
        if config.hyper_fan_init:
            warnings.warn('Option "hyper_fan_init" has no effect if no ' +
                          'hypernetwork is used.')
        if config.nh_separate_out_head:
            # Note, option doesn't make sense for context-mod hnet.
            raise ValueError('Option "nh_separate_out_head" only applicable ' +
                             'when using "--hnet_all".')

    if config.nh_separate_out_head:
        if config.multi_head:
            # FIXME Was easier to implement this way.
            warnings.warn('Option "multi_head" shouldn\'t be used in ' +
                          'conjunction with option "nh_separate_out_head". ' +
                          'Otherwise (due to implementation reasons) a task-' +
                          'specific multi-head is generated rather then a ' +
                          'task-specific single-head, therefore leading too ' +
                          'a lot of unused weights.')
        if hasattr(config, 'use_replay') and config.use_replay:
            raise ValueError('Option "nh_separate_out_head" not applicable ' +
                             'to replay decoder.')

    # Note, the following warnings should simply save us from doing stupid
    # comparisons of single head EWC/SI networks with hnet-protected single-
    # head networks. A proper CL1 comparison demands a multi-head for EWC/SI.
    if config.use_ewc:
        if not config.multi_head and \
                not (config.use_context_mod or config.use_masks):
            warnings.warn('It doesn\'t make sense to use EWC without a ' +
                          'multi-head output when no context modulation or ' +
                          'masking is being used. At least in a CL1 scenario.')
    if config.use_si:
        if not config.multi_head and \
                not (config.use_context_mod or config.use_masks):
            warnings.warn('It doesn\'t make sense to use SI without a ' +
                          'multi-head output when no context modulation or ' +
                          'masking is being used. At least in a CL1 scenario.')

    if hasattr(config, 'use_replay') and config.use_replay:
        # Of course, one can also protect a generative model with EWC or SI
        # (as already studied in other papers). But this is not within the
        # scope of this work.
        if config.use_si:
            raise ValueError('Option "use_si" is not compatible with ' +
                             'option "use_replay".')
        if config.use_ewc:
            raise ValueError('Option "use_ewc" is not compatible with ' +
                             'option "use_replay".')
        if config.use_context_mod:
            raise ValueError('Option "use_context_mod" is not compatible ' +
                             'with option "use_replay".')
        # Same argument as above. Of course masking+SI/EWC can be used to
        # protect a generative model, but we don't assess forgetting in
        # generative models in this work.
        if config.use_masks:
            raise ValueError('Option "use_masks" is not compatible with ' +
                             'option "use_replay".')
        if config.multitask:
            # Doesn't make sense without continual learning.
            raise ValueError('Option "multitask" is not compatible with ' +
                             'option "use_replay".')
        if config.train_tnet_once:
            raise ValueError('Option "train_tnet_once" is not compatible ' +
                             'with option "use_replay".')
        if config.train_from_scratch:
            # Note, this option doesn't make sense when using replay. The
            # classifier would be trained on data from the current task and
            # replayed data from a garbage replay model.
            raise ValueError('Option "train_from_scratch" is not compatible ' +
                             'with option "use_replay". Consider option ' +
                             '"reinit_tnet" which will only reinitialize the ' +
                             'encoder.')
        if hasattr(config, 'all_task_softmax') and \
                config.all_task_softmax and config.multi_head:
            # How do we handle this? A full softmax is not a multi-head
            # system but the main network has the same output size as if it were
            # multi-head -> Needs to be enforced, as we assume that they are
            # mutually-exclusive in the code.
            raise ValueError('Options "all_task_softmax" and "multi_head" ' +
                             'are not compatible.')
        if config.replay_true_data:
            warnings.warn('Option "replay_true_data" is just a sanity check. ' +
                          'It breaks the continual learning assumption that ' +
                          'previous data is available for training new tasks.')
        if config.hnet_all and (config.replay_true_data or \
                                config.coreset_size != -1):
            raise ValueError('Hypernetwork not applicable if replay of ' +
                             'actual data is performed.')
        if config.replay_true_data and config.coreset_size != -1:
            raise ValueError('Options "replay_true_data" and "coreset_size" ' +
                             'are not compatible.')
        if config.input_task_identity:
            # FIXME I think only the plotting (to tensorboard) functions have
            # a problem with that.
            raise NotImplementedError()

    if hasattr(config, 'use_replay') and not config.use_replay:
        if hasattr(config, 'all_task_softmax') and config.all_task_softmax:
            # Note, task-inference can also be done for non-replay method
            # (as shown via option HNET+ENT). But those methods rely on a
            # multi-head output.
            warnings.warn('Option "all_task_softmax" has no effect if not ' +
                          'using replay. Set to "False"')
            config.all_task_softmax = False
        if config.coreset_size != -1:
            raise ValueError('Coresets can only be used if "use_replay" is ' +
                             'activated.')

    if config.use_best_models and hasattr(config, 'early_stopping_thld') and \
            config.early_stopping_thld > 0:
        raise ValueError('Options "use_best_models" and ' +
                         '"early_stopping_thld" are mutually exclusive.')
    if config.train_only_heads_after_first:
        if config.hnet_all or config.train_tnet_once:
            raise ValueError('Option to only train the output '+
                'weights of the RNN is not implemented in this case.')
        if config.use_ewc or config.use_si:
            raise ValueError('It doesnt make sense to use EWC/SI if all '
                             'shared weights are fixed.')

    if config.dont_train_heads:
        if config.train_only_heads_after_first:
            raise ValueError('The current implementation of options ' +
                '"dont_train_heads" and "train_only_heads_after_first" is ' +
                'not compatible currently.')
        # FIXME I don't think that's true and that we can delete the error.
        if config.use_context_mod:
            raise NotImplementedError('The option "dont_train_heads" cannot ' +
                'currently be used with context-mod.')

    if hasattr(config, 'use_bidirectional_net') and \
            config.use_bidirectional_net:
        # Not yet supported by BiLSTM, but should be an easy fix.
        if config.use_ewc and config.tbptt_fisher != -1:
            raise NotImplementedError()


def update_cli_args(config):
    """Update command line arguments of a given config file.

    This function can be used if previously stored config files need to be used
    to run new experiments.

    Args:
        config (argparse.Namespace): Parsed command-line arguments.

    Returns:
        The updated parsed command-line arguments.

    """
    if hasattr(config, 'first_task_seq_length'):
        config.first_task_input_len = config.first_task_seq_length
        delattr(config, 'first_task_seq_length')
    if hasattr(config, 'seq_length_variability'):
        config.input_len_variability = config.seq_length_variability
        delattr(config, 'seq_length_variability')
    if hasattr(config, 'seq_length_step'):
        config.input_len_step = config.seq_length_step
        delattr(config, 'seq_length_step')
    if hasattr(config, 'num_nonzeroed_ts'):
        config.pat_len = config.num_nonzeroed_ts
        delattr(config, 'num_nonzeroed_ts')
    if hasattr(config, 'num_zeroed_ts'):
        # There's no way to go from "num_zeroed_ts" to "pat_len", since
        # "num_zeroed_ts" determines the number of timesteps to silence
        # at the end of the input sequence, for varying lengths, whereas
        # "pat_len" is identical for all input sequences. So we can only
        # process runs that had "num_zeroed_ts" equal to zero.
        assert config.num_zeroed_ts == 0
        config.pat_len = -1
        delattr(config, 'num_zeroed_ts')
    if not hasattr(config, 'srnn_pre_fc_layers'):
        config.srnn_pre_fc_layers = ''
    # FIXME cheap way of figuring out whether this is a copy task config.
    is_copy_config = hasattr(config, 'first_task_input_len')
    if is_copy_config and not hasattr(config, 'seq_out_width'):
        config.seq_out_width = -1
    if is_copy_config and not hasattr(config, 'use_new_permuted_dhandler'):
        config.use_new_permuted_dhandler = False
    if is_copy_config and not hasattr(config, 'scatter_pattern'):
        config.scatter_pattern = False
    if hasattr(config, 'permute_and'):
        if config.permute_and:
            raise NotImplementedError('Option "permute_and" has been deleted.')
        else:
            warnings.warn('Removing deprecated option "permute_and" from ' +
                          'config.')
            delattr(config, 'permute_and')
    if is_copy_config and not hasattr(config, 'permute_xor'):
        config.permute_xor = False
    if is_copy_config and not hasattr(config, 'permute_xor_iter'):
        config.permute_xor_iter = -1
    if is_copy_config and not hasattr(config, 'permute_xor_separate'):
        config.permute_xor_separate = False
    if is_copy_config and not hasattr(config, 'random_pad'):
        config.random_pad = False
    if is_copy_config and not hasattr(config, 'pad_after_stop'):
        config.pad_after_stop = False
    if is_copy_config and not hasattr(config, 'pairwise_permute'):
        config.pairwise_permute = False
    if is_copy_config and not hasattr(config, 'revert_output_seq'):
        config.revert_output_seq = False
    if is_copy_config and not hasattr(config, 'dont_train_heads'):
        config.dont_train_heads = False

    return config


if __name__=='__main__':
    pass