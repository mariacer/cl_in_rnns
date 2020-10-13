#!/usr/bin/env python3
# Copyright 2019 Benjamin Ehret, Maria Cervera

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
# @title           :sequential/train_utils_sequential.py
# @author          :mc, be
# @contact         :mariacer@ethz.ch
# @created         :24/03/2020
# @version         :1.0
# @python_version  :3.6.8
"""
Helper functions for training on timeseries data
------------------------------------------------

Continual learning with hypernetworks for sequential data.
Here we train a hypernetwork to output the weights of a main RNN.
"""
import torch
import numpy as np
import os
import pickle
import pandas as pd
from time import time
from torch.nn import functional as F
from warnings import warn

from data.sequential_dataset import SequentialDataset
from mnets.bi_rnn import BiRNN
from mnets.simple_rnn import SimpleRNN
from hnets.chunked_hyper_model import ChunkedHyperNetworkHandler
from hnets.mlp_hnet import HMLP
from hnets.chunked_mlp_hnet import ChunkedHMLP
from hnets.structured_mlp_hnet import StructuredHMLP
from hnets.chunked_deconv_hnet import ChunkedHDeconv
from hnets.hnet_container import HContainer
import sequential.plotting_sequential as plc
from sequential.rnn_chunking import simple_rnn_chunking
from hnets.hyper_model import HyperNetwork
import utils.misc as misc
from utils import sim_utils as sutils
from utils import torch_ckpts as tckpt

def generate_networks(config, shared, data_handlers, device):
    """Create the target network and potential auxilliary networks.

    Simple helper function to create the classifier using
    :func:`_generate_classifier` and (if needed) the hypernetwork and replay
    decoder.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Miscellaneous information shared across
            functions.
        data_handlers: List of data handlers, one for each task. Needed to
            extract the number of inputs/outputs of the main network and to
            infer the number of tasks.
        device: Torch device.

    Returns:
        (tuple): Tuple containing:

        - **tnet**: The target network.
        - **hnet** (optional): The hypernetwork. ``None`` is returned in case
          no hypernetwork is required.
        - **dnet** (optional): The replay decoder. If no replay decoder is
          needed, the return value is zero.
    """
    assert hasattr(shared, 'feature_size')

    ### Generate main net ###
    tnet = _generate_classifier(config, shared, data_handlers, device)

    ### Generate replay decoder ###
    dnet = None
    if hasattr(config, 'use_replay') and config.use_replay and not \
            (config.replay_true_data or config.coreset_size != -1):
        print('Constructing replay decoder ...')
        if not config.hnet_all and config.multi_head:
            # Task-identity has to be inputted into the decoder in order to be
            # able to select soft targets for classifier.
            in_shape = [config.latent_dim + len(data_handlers)]
        else:
            in_shape = [config.latent_dim]
        # Assuming the input shape is the same for all tasks.
        n_out = shared.feature_size
        if config.input_task_identity:
            # FIXME Should not allow the option together with replay or properly
            # handle the reconstruction of a 1-hot encoding?
            warn('Reconstruction loss might not properly account for option ' +
                 '"input_task_identity".')
            n_out += len(data_handlers)
        out_shape = [n_out]

        dnet = sutils.get_mnet_model(config, 'simple_rnn', in_shape, out_shape,
            device, cprefix='dec_', no_weights=config.hnet_all)

        if config.orthogonal_hh_init:
            dnet.init_hh_weights_orthogonal()

    ### Generate hypernet ###
    hnet = None
    if config.hnet_all or \
            config.use_context_mod and not config.checkpoint_context_mod:
        if dnet is not None:
            print('Constructing hypernetwork for replay model ...')
            hyper_shapes = dnet.hyper_shapes_learned
            hyper_target_net = dnet
        else:
            print('Constructing hypernetwork for classifier ...')
            hyper_shapes = tnet.hyper_shapes_learned
            hyper_target_net = tnet

        if hasattr(config, 'use_new_hnet') and config.use_new_hnet:
            hnet = _get_new_hnet(config, device, hyper_target_net, hyper_shapes)
        else:
            print('Using an old, deprecated hypernetwork. ' +
                  'Consider using option "--use_new_hnet".')
            hnet = _get_old_hnet(config, device, hyper_shapes)

    return tnet, hnet, dnet

def _get_old_hnet(config, device, hyper_shapes):
    """Construct and initialize a hypernetwork.

    Note:
        This function constructs objects from the old deprecated hypernet
        classes from this repository.

    Args:
        (....): See docstring of function :func:`_get_new_hnet`.

    Returns:
        (utils.module_wrappers.CLHyperNetInterface)
    """
    hnet = sutils.get_hnet_model(config, config.num_tasks, device, hyper_shapes)

    config.compression_ratio = hnet.num_weights / hnet.num_outputs
    print('Compression Ratio: %.2f' % config.compression_ratio)

    # Initialization.
    assert hnet.has_task_embs
    for temb in hnet.get_task_embs():
        torch.nn.init.normal_(temb, mean=0., std=config.std_normal_temb)

    if not config.hyper_fan_init and hasattr(hnet, 'chunk_embeddings'):
        for cemb in hnet.chunk_embeddings:
            torch.nn.init.normal_(cemb, mean=0, std=config.std_normal_emb)
    elif config.hyper_fan_init:
        temb_var = config.std_normal_temb**2
        if isinstance(hnet, HyperNetwork):
            hnet.apply_hyperfan_init(method='in', use_xavier=False,
                                     temb_var=temb_var)
        elif isinstance(hnet, ChunkedHyperNetworkHandler):
            hnet.apply_chunked_hyperfan_init(method='in', use_xavier=False,
                temb_var=temb_var, eps=1e-5, cemb_normal_init=False)
        else:
            raise NotImplementedError('No hyperfan-init implemented for ' +
                'hypernetwork of type %s.' % type(hnet))

    return hnet

def _get_new_hnet(config, device, target_net, hyper_shapes):
    """Construct and initialize a hypernetwork.

    Args:
        (....): See docstring of function :func:`generate_networks`.
        target_net (mnets.simple_rnn.SimpleRNN): The network for which weights
            should be produced (required if a structured hypernetwork is build).
        hyper_shapes (list): The shapes of the weights that should be produced
            by the hypernetwork.

    Returns:
        (hnets.hnet_interface.HyperNetInterface)
    """
    chunk_shapes = None
    num_per_chunk = None
    assembly_fct = None
    if config.nh_hnet_type == 'structured_hmlp':
        if config.nh_separate_out_head:
            raise NotImplementedError('Option "nh_separate_out_head" not ' +
                                      'compatible with structured hnets.')
        nh_shmlp_chunk_sizes = misc.str_to_ints(config.nh_shmlp_chunk_sizes)
        if len(nh_shmlp_chunk_sizes) == 1:
            nh_shmlp_chunk_sizes = nh_shmlp_chunk_sizes[0]
        chunk_shapes, num_per_chunk, assembly_fct = simple_rnn_chunking( \
            target_net, chunk_size=nh_shmlp_chunk_sizes,
            fc_chunking=config.nh_shmlp_chunk_fc_layers)

        print('Using the following chunking structure:')
        for i, s in enumerate(chunk_shapes):
            print('Layer %d: %d chunk(s) of shape: %s.' % (i, num_per_chunk[i],
                  s))

    # If requested, exclude output weights before building hypernetwork.
    if config.nh_separate_out_head:
        ow_masks = target_net.get_output_weight_mask()
        ow_inds = [i for i in range(len(ow_masks)) if ow_masks[i] is not None]
        rw_inds = [i for i in range(len(ow_masks)) if ow_masks[i] is None]

        orig_hshapes = list(hyper_shapes)
        ow_shapes = [hyper_shapes[i] for i in ow_inds]
        hyper_shapes = [hyper_shapes[i] for i in rw_inds]

    hnet = sutils.get_hypernet(config, device, config.nh_hnet_type,
        hyper_shapes, config.num_tasks, no_cond_weights=False,
        no_uncond_weights=False, uncond_in_size=0,
        shmlp_chunk_shapes=chunk_shapes, shmlp_num_per_chunk=num_per_chunk,
        shmlp_assembly_fct=assembly_fct,
        verbose=not config.nh_separate_out_head, cprefix='nh_')

    ### Initialization ###
    # Initialize task embeddings.
    if config.nh_cond_emb_size > 0:
        for tid in range(config.num_tasks):
            torch.nn.init.normal_(hnet.get_cond_in_emb(tid), mean=0.,
                                  std=config.std_normal_temb)

    if not config.hyper_fan_init and \
            isinstance(hnet, (ChunkedHMLP, StructuredHMLP, ChunkedHDeconv)):
        # Initialize chunk embeddings.
        nc = config.num_tasks if config.nh_use_cond_chunk_embs else 1
        for t in range(nc):
            tid = t
            if not config.nh_use_cond_chunk_embs:
                tid = None
            if isinstance(hnet,StructuredHMLP):
                cembs = hnet.get_chunk_embs(cond_id=tid)
            else:
                cembs = hnet.get_chunk_emb(cond_id=tid)
                cembs = [cembs]
            for cemb in cembs:
                torch.nn.init.normal_(cemb, mean=0, std=config.std_normal_emb)

    elif config.hyper_fan_init:
        temb_var = config.std_normal_temb**2
        if isinstance(hnet, HMLP):
            hnet.apply_hyperfan_init(method='in', use_xavier=False,
                                     cond_var=temb_var)
        elif isinstance(hnet, ChunkedHMLP):
            hnet.apply_chunked_hyperfan_init(method='in', use_xavier=False,
                cond_var=temb_var, eps=1e-5, cemb_normal_init=False)
        else:
            raise NotImplementedError('No hyperfan-init implemented for ' +
                'hypernetwork of type %s.' % type(hnet))

    ### Build hypernetwork container ###
    # If requested, add task-specific output heads to hypernetwork.
    if config.nh_separate_out_head:
        hnet_rem = hnet
        def assembly_fct(list_of_hnet_tensors, uncond_tensors, cond_tensors):
            assert len(list_of_hnet_tensors) == 1
            hnet_tensors = list_of_hnet_tensors[0]
            all_tensors = []
            cind, hind = 0, 0
            for i in range(len(orig_hshapes)):
                if i in ow_inds:
                    all_tensors.append(cond_tensors[cind])
                    cind += 1
                else:
                    all_tensors.append(hnet_tensors[hind])
                    hind += 1
            return all_tensors

        hnet = HContainer(orig_hshapes, assembly_fct, hnets=[hnet_rem],
                          cond_param_shapes=ow_shapes,
                          num_cond_embs=config.num_tasks).to(device)
        # FIXME We might want to initialize output weights differently.

    config.compression_ratio = hnet.num_params / hnet.num_outputs

    return hnet

def _generate_classifier(config, shared, data_handlers, device):
    """Create a RNN network that will be trained on multiple tasks.

    Args:
        (....): See docstring of function :func:`generate_networks`.

    Returns:
        A RNN instance.
    """
    n_in = shared.feature_size
    n_out = data_handlers[0].out_shape[0]

    # Sanity check: for now, we only support homogeneous output head sizes.
    for dh in data_handlers:
        assert len(dh.out_shape) == 1
        assert len(dh.in_shape) == 1
        # Sanity check below not applicable anymore.
        #assert dh.in_shape[0] == n_in
        assert dh.out_shape[0] == n_out

    max_num_timesteps = None
    if isinstance(data_handlers[0], SequentialDataset):
        max_num_timesteps = data_handlers[0].max_num_ts_in
        for i in range(1, len(data_handlers)):
            if data_handlers[i].max_num_ts_in > max_num_timesteps:
                max_num_timesteps = data_handlers[i].max_num_ts_in
    else:
        raise NotImplementedError()
        assert isinstance(data_handlers[0], CognitiveTasks)
        # FIXME SequentialDataset interface not yet implemented.
        max_num_timesteps = 100

    # If required, add a set of input units to allow a one-hot-encoding of the 
    # task identity.
    if config.input_task_identity:
        n_in += len(data_handlers)

    n_hidden = misc.str_to_ints(config.rnn_arch)
    post_fc_layers = misc.str_to_ints(config.srnn_post_fc_layers)
    if len(post_fc_layers) > 0:
        print('Adding additional fully-connected layers to the main network.')
    pre_fc_layers = misc.str_to_ints(config.srnn_pre_fc_layers)
    if len(pre_fc_layers) > 0:
        print('Prepending additional fully-connected layers to the main ' +
              'network.')
    config.input_dim = n_in
    config.out_dim = n_out

    if config.multi_head:
        n_out = config.out_dim * config.num_tasks
    else: # Single-head.
        n_out = config.out_dim

    if hasattr(config, 'use_replay') and config.use_replay:
        if hasattr(config, 'all_task_softmax') and config.all_task_softmax:
            n_out = config.out_dim * config.num_tasks
        # Note, we add two times the latent dimensions as we output the
        # factorized Gaussian distribution of latent codes.
        n_out += 2 * config.latent_dim

    print('Constructing target network ...')

    # In both cases, we need to have context-mod layers in the main network.
    # but, if we are using masks, then we do not train the context-mod weights.
    use_cm_layers = config.use_context_mod or config.use_masks

    # Note, if we checkpoint context modulation weights (i.e, just store them),
    # then we don't need a hypernetwork and can just treat the context mod
    # weights as internal parameters of the main network.
    context_mod_no_weights = config.use_context_mod and not \
        config.checkpoint_context_mod or config.use_masks

    no_tnet_weights = config.hnet_all
    if hasattr(config, 'use_replay') and config.use_replay:
        # We don't support a second hypernet when using replay.
        assert not config.use_context_mod or config.checkpoint_context_mod
        # Independent of option `hnet_all`, the target network will have it's
        # own weights trained by all data (replayed from previous tasks). The
        # decoder will be hypernet protected.
        no_tnet_weights = False

    cm_args_in_out = {
        'context_mod_inputs': config.context_mod_inputs,
        'no_last_layer_context_mod': config.no_context_mod_outputs,
    }
    cm_args = {
        'use_context_mod': use_cm_layers,
        'context_mod_post_activation': config.context_mod_post_activation,
        'context_mod_no_weights': context_mod_no_weights,
        'context_mod_gain_offset': config.offset_gains,
        'context_mod_gain_softplus': not config.dont_softplus_gains,
    }
    cm_args_rnn_only = {
        'context_mod_num_ts': max_num_timesteps if config.context_mod_per_ts \
            else -1,
        'context_mod_separate_layers_per_ts': not context_mod_no_weights
            or config.use_masks,
    }

    if hasattr(config, 'use_bidirectional_net') and \
            config.use_bidirectional_net:
        last_n_hidden = n_hidden[-1] if len(post_fc_layers) == 0 else \
            post_fc_layers[-1]

        net = BiRNN(
            rnn_args=dict({
                'n_in': n_in,
                'rnn_layers': n_hidden,
                'fc_layers_pre': pre_fc_layers,
                'fc_layers': post_fc_layers,
                'activation': misc.str_to_act(config.net_act),
                'use_lstm': not config.use_vanilla_rnn,
                'use_bias': True,
                'context_mod_inputs': config.context_mod_inputs,
            }, **cm_args, **cm_args_rnn_only),
            mlp_args=dict(**{
                'n_in': 2 * last_n_hidden,
                'n_out': n_out,
                'hidden_layers': [],
                'activation_fn': misc.str_to_act(config.net_act),
                'use_bias': True,
                'no_last_layer_context_mod': config.no_context_mod_outputs,
            }, **cm_args),
            no_weights=no_tnet_weights
        ).to(device)
    else:
        net = SimpleRNN(n_in=n_in,
            rnn_layers=n_hidden,
            fc_layers_pre=pre_fc_layers,
            fc_layers=[*post_fc_layers, n_out],
            no_weights=no_tnet_weights,
            activation=misc.str_to_act(config.net_act),
            use_lstm=not config.use_vanilla_rnn,
            # Note, due to an implementation mismatch, the SimpleRNN initially
            # used always this init rather than the official PyTorch one. To not
            # break all prior hpsearch results, we decided to keep the init this
            # way.
            kaiming_rnn_init=True,
            use_bias=True,
            **cm_args_in_out,
            **cm_args,
            **cm_args_rnn_only
        ).to(device)

    if config.orthogonal_hh_init:
        net.init_hh_weights_orthogonal()

    return net


def save_activations(config, activations, targets=None, modulations=None,
        name_prefix=''):
    """Save the hidden activations and targets.

    Save the hidden activity for each of the tasks, when evaluated on the
    test set after training on all tasks. The targets for the
    tasks are also stored, as these might be needed for the analyses.

    For the context-modulation setting, this data will then be analyzed in
    another script :mod:`bio.cognet.analyse_activations`.

    Args:
        config: Command-line arguments.
        activations (list): The hidden activations in each of the tasks.
        targets (list, optional): The targets of the network for the current
            batch.
        modulations (list, optional): The gains and shifts for each task,
            obtained during final test. Only for context-modulation settings.
        name_prefix (str, optional): Prefix for the name of the file.
    """
    
    if name_prefix != '':
        name_prefix += '_'
    with open(os.path.join(config.out_dir, '%sactivations.pickle'%name_prefix),\
            'wb') as f:
        pickle.dump(activations, f)

    if targets is not None:
        with open(os.path.join(config.out_dir, 'targets.pickle'), 'wb') as f:
            pickle.dump(targets, f)
    if modulations is not None:
        with open(os.path.join(config.out_dir,
                               'modulations.pickle'), 'wb') as f:
            pickle.dump(modulations, f)

def log_results(test_results, config, logger):
    """Log the results.

    Args:
        test_results (np.array): The results to be logged. Either losses or
            accuracies (for classification tasks).
        config: The command line arguments.
        logger: Console (and file) logger.
    """
    # For classification tasks report the accuracy, else report the loss.
    if config.classification:
        var_name = 'accuracy'
    else:
        var_name = 'loss'

    if config.multitask:
        logger.info('Avg %s test: %.3f' % (var_name, np.mean(test_results)))
    else:
        during_results = np.diagonal(test_results)
        final_results = test_results[-1, :]
        logger.info('During %s: %s' % (var_name, \
            np.array2string(during_results, precision=5, separator=',')))
        if not config.train_from_scratch:
            logger.info('Final %s: %s' %(var_name, \
                np.array2string(final_results, precision=5, separator=',')))
        logger.info('Avg %s during training: %.3f' %(var_name, \
            np.mean(during_results)))
        if not config.train_from_scratch:
            logger.info('Avg %s final test: %.3f' %(var_name, \
                np.mean(final_results)))


def save_performance_summary(config, shared, test_results, train_loss,
                             summary_filename='', summary_keywords=[],
                             finished_training=False):
    """Save a summary of the test results.

    Save a summary of the test results achieved with the current run
    configuration. There will be two different files: one containing the run
    configuration as well as simple summary statistics (for hpsearch), and one
    containing more extensive results (for analysis).

    Args:
        config: Command-line arguments.
        shared (argparse.Namespace): Miscellaneous information shared across
            functions.
        test_results: 2d numpy array (n_tasks x n_tasks) containing the test
            results after every training step. Either accuracy or loss.
        train_loss: list of dictionaries containing the training losses for all
            tasks.
        summary_filename (str): The name of the summary file.
        summary_keywords (list): The list of keyword whose values should be
            stored in the csv.
        finished_training (str, optional): Whether training is finished or not.
    """
    if config.classification:
        var_name = 'accuracy'
        var_short_name = 'acc'
    else:
        var_name = 'loss'
        var_short_name = 'loss'

    with open(os.path.join(config.out_dir, 'config.pickle'), 'wb') as f:
        pickle.dump(config, f)
    with open(os.path.join(config.out_dir, 'test_%s.pickle'%var_short_name),\
            'wb') as f:
        pickle.dump(test_results, f)
    with open(os.path.join(config.out_dir, 'train_loss.pickle'), 'wb') as f:
        pickle.dump(train_loss, f)

    # plot results
    if finished_training:
        if not config.multitask:
            plc.plot_train_loss(config.out_dir)
        if config.classification:
            plc.plot_test_results(config.out_dir,
                classification=config.classification)

    # Store summary results in a .csv file.
    results_file = config.out_dir + '/results_summary.csv'

    # TODO horribly ugly code. Fix.
    if not config.multitask:
        results_summary = dict()
        if hasattr(shared, 'f_scores'):
            results_summary = dict(**results_summary, **{
                'mean_final_fscore': [np.mean(shared.f_scores[-1, :])],
                'mean_during_fscore': [np.mean(np.diagonal(shared.f_scores))],
                'final_fscore': [shared.f_scores[-1, :]],
                'during_fscore': [np.diagonal(shared.f_scores)],
            })

        results_summary = dict(**results_summary, **{
            'mean_final_%s'%var_name: [np.mean(test_results[-1, :])],
            'std_final_%s'%var_name: [np.std(test_results[-1, :])],
            'mean_during_%s'%var_name: [np.mean(np.diagonal(test_results))],
            'std_during_%s'%var_name: [np.std(np.diagonal(test_results))],
            'min_final_%s'%var_name: [np.min(test_results[-1, :])],
            'min_during_%s'%var_name: [np.min(np.diagonal(test_results))],
            'during_%s'%var_name:  [np.diagonal(test_results)],
            'final_%s'%var_name: [test_results[-1, :]],
        })
    else:
        results_summary = dict()
        if hasattr(shared, 'f_scores'):
            results_summary = dict(**results_summary, **{
                'mean_final_fscore': [np.mean(shared.f_scores[-1, :])],
                'final_fscore': [shared.f_scores[-1, :]],
            })

        results_summary = dict(**results_summary, **{
            'mean_final_%s'%var_name: [np.mean(test_results)],
            'std_final_%s'%var_name: [np.std(test_results)],
            'min_final_%s'%var_name: [np.min(test_results)],
            'final_%s'%var_name: [test_results],
        })
    results_summary['rnn_arch'] = ['"%s"'%config.rnn_arch]
    results_summary['hnet_arch'] = [np.nan]
    results_summary['compression_ratio'] = [np.nan]
    if config.hnet_all or config.use_context_mod:
        # FIXME We don't do the following (commented code) for several reasons.
        # First of all the current performance summary parser doesn't allow for
        # strings (assumed that they are a list of numbers). Second, there is a
        # command-line argument with the same name.
        #hc = misc.str_to_ints(config.hyper_chunks)
        #if len(hc) == 1 and hc[0] == -1:
        #    hnet_arch = '"%d -> %s"' % (config.temb_size, config.hnet_arch)
        #else:
        #    hnet_arch = '"%d+%d -> %s -> %s"' % (config.temb_size,
        #        config.emb_size, config.hnet_arch, config.hyper_chunks)
        #results_summary['hnet_arch'] = [hnet_arch]
        results_summary['hnet_arch'] = [config.hnet_arch]
        results_summary['compression_ratio'] = [config.compression_ratio]

    df = pd.DataFrame.from_dict(results_summary)
    df.to_csv(results_file, sep=';', index=False)

    # Store summary results in a .txt file
    tp = dict()
    for kw, value in zip(results_summary.keys(), results_summary.values()):
        tp[kw] = value[0]
    tp["finished"] = 1 if finished_training else 0

    with open(os.path.join(config.out_dir, summary_filename), 'w') as f:

        assert('num_train_iter' in summary_keywords)

        for kw in summary_keywords:
            if kw == 'num_train_iter':
                f.write('%s %d\n' % ('num_train_iter', config.n_iter))
                continue
            elif kw in ['rnn_arch', 'hnet_arch'] and tp[kw] == '':
                f.write('%s %s\n' % (kw, '""'))
                continue

            if isinstance(tp[kw], list):
                f.write('%s %s\n' % (kw, misc.list_to_str(tp[kw])))
            elif isinstance(tp[kw], float):
                f.write('%s %f\n' % (kw, tp[kw]))
            elif isinstance(tp[kw], int):
                f.write('%s %d\n' % (kw, tp[kw]))
            elif isinstance(tp[kw], np.ndarray):
                assert tp[kw].ndim == 1 or tp[kw].shape[0] == 1
                f.write('%s %s\n' % (kw,
                                     misc.list_to_str(tp[kw].tolist())))
            else:
                f.write('%s %s\n' % (kw, tp[kw]))

def sequential_nll(loss_type='ce', reduction='sum'):
    r"""Create a custom NLL function for sequential tasks.

    Such an NLL function might be used to invoke function
    :func:`utils.ewc_regularizer.compute_fisher` (see option ``custom_nll``).

    We consider a network that has an output shape of ``[T, B, C]``, where ``T``
    is the length of the sequence (number of timesteps), ``B`` is the batch size
    (we can assume ``B=1`` in function
    :func:`utils.ewc_regularizer.compute_fisher`, but the returned function
    will work with arbitrary batch sizes) and ``C`` is the
    number of output channels (e.g., number of classes for a classification
    problem). Note that, in the multi-head scenario, here we expect that the 
    correct output head has already been selected, such that ``C`` corresponds 
    to the number of classes in the correct output head.

    We first derive the likelihood for a dataset with :math:`N` samples
    :math:`p(\mathcal{D} \mid W) = \prod_{n=1}^N p(\mathbf{y}_n \mid W)`,
    where :math:`W` are the weights of the neural network. Each sample in the
    dataset can be decomposed into targets per timestep as follows
    :math:`\mathbf{y}_n = (y_n^{(1)}, \dots, y_n^{(T)})` (note, individual
    :math:`y_n^{(t)}` might still be vectors of targets, e.g., in case of the
    `copy task <https://arxiv.org/pdf/1410.5401.pdf>`__).


    Similar to function :func:`utils.ewc_regularizer.compute_fisher` (cmp.
    argument ``time_series``), we adopt the following decomposition of the joint
    (also compare docstring of function
    :func:`bio.cognitive.train_utils_cognitive.cognet_mse_nll`, which is a
    specialized implementation of this function)

    .. math::

        p(\mathbf{y}_n \mid W) = p(y_n^{(1)}, \dots, y_n^{(T)} \mid W) = \
            \prod_{t=1}^T p(y_n^{(t)} \mid y_n^{(1)}, \dots, y_n^{(t-1)}, W)

    We now claim, that

    .. math::

        p(y_n^{(t)} \mid y_n^{(1)}, \dots, y_n^{(t-1)}, W) \
            \approx p(y_n^{(t)} \mid \mathbf{h}_n^{(t-1)}, W)

    where :math:`\mathbf{h}_n^{(t-1)}` is the RNN state used to compute the
    likelihood of :math:`y_n^{(t)}`.

    We need to define a loss function that minimizes the negative-log-likelihood
    (NLL)

    .. math::

        \text{NLL} = \sum_{n=1}^N \text{NLL}_n = \
            \sum_{n=1}^N \sum_{t=1}^T \text{NLL}_n^{(t)}

    with :math:`\text{NLL}_n^{(t)} \equiv - \
    \log p(y_n^{(t)} \mid \mathbf{h}_n^{(t-1)}, W)`.

    Note:
        The NLL sums over all samples in the dataset. Hence, whenever one has to
        compute the NLL over a minibatch, it should be renormalized by dividing
        by the batch size and multiplying by the dataset size!

    In the following, we consider how :math:`\text{NLL}_n^{(t)}` can be computed
    for different kinds of problems.

    **Mean-squared-error loss**

    In case of a mean-squared error loss, we assume the likelihood to be
    Gaussian with diagonal covariance. The quantity :math:`y_n^{(t)}` is now a
    ``C``-dimensional target vector. We denote the actual network output for
    timestep :math:`t` by :math:`z_n^{(t)}`, which is also ``C``-dimensional.

    Note:
        Network outputs should be logits, i.e., spanning the whole real line
        since Gaussians have full support.

    We can then define the per sample loss function as follows

    .. math::
        L_n = \frac{1}{T} \sum_{t=1}^T \frac{1}{C} \
            \sum_{c=1}^C m^{(t,c)} (y_n^{(t,c)} - z_n^{(t,c)})^2

    The mask :math:`m^{(t,c)}` shall introduce a greater importance for some
    timesteps/classes.

    Since the optimized loss function represents a MSE, we consider a Gaussian
    likelihood where the masking values enter implicitly as variances:

    .. math::

        \text{NLL}_n^{(t)} &\equiv \
            - \log p(y_n^{(t)} \mid \mathbf{h}_n^{(t-1)}, W) \\
            &= \text{const.} + \frac{1}{2} \sum_{c=1}^C \frac{1}{\sigma_{t,c}^2}
            \big(y_n^{(t,c)} - z_n^{(t,c)} \big)^2 \\
            &= \text{const.} + \frac{1}{C T} \sum_{c=1}^C m^{(t,c)}
            \big(y_n^{(t,c)} - z_n^{(t,c)} \big)^2 \\

    where :math:`m^{(t,c)} = \frac{CT}{2\sigma_{t,c}^2}`, such that if we choose
    :math:`\sigma_{t,c}^2 = \frac{CT}{2m^{(t,c)}}` then the NLL perfectly
    corresponds to the loss function chosen above.

    Hence, by defining a mask :math:`m^{(t,c)}` we implicitly define the
    variances of the Gaussian likelihood function. For instance, a mask value
    :math:`m^{(t,c)} = 0` would correspond to an infinite variance, which makes
    sense since we don't care about the output value in that case.

    **Cross-entropy loss**

    In this case, the target :math:`y_n^{(t)}` represents the target class
    (the label) for timestep :math:`t`. We consider :math:`z_n^{(t)}` as the
    ``C``-dimensional logit output of our network and consider the likelihood
    for each label defined by a softmax

    .. math::

        \tilde{z}_n^{(t)} = \text{softmax}(\beta^{(t)} z_n^{(t)})

    where :math:`\beta^{(t)}` determines the inverse temperature for timestep
    :math:`t`, which we can use to introduce varying importances for timesteps.

    We can now express the NLL for sample :math:`n` at timestep :math:`t` as
    follows

    .. math::

        \text{NLL}_n^{(t)} &\equiv \
            - \log p(y_n^{(t)} \mid \mathbf{h}_n^{(t-1)}, W) \\
            &= - \log \tilde{z}_n^{(t,y_n^{(t)})} \\
            &= - \sum_{c=1}^C [y_n^{(t)} = c] \log \tilde{z}_n^{(t,c)} \\
            &= - \sum_{c=1}^C [y_n^{(t)} = c] \log \
                \big( \text{softmax}(\beta^{(t)} \mathbf{z}_n^{(t)})_c \big)

    where :math:`[\cdot]` denotes the Iverson bracket. One possibility to
    incorporate some kind of masking is to set different temperature values
    :math:`\frac{1}{\beta^{(t)}}` for different timesteps. For instance:

    - Setting :math:`\beta^{(t)} = 0` is equivalent to saying that we don't care
      about the prediction (it will also stop gradient flow through the
      network).
    - Setting :math:`\beta^{(t)}` to a large value corresponds to requesting
      sharper and more confident predictions.

    **Binary cross-entropy loss**

    In this case, the target :math:`y_n^{(t)}` is a binary ``C``-dimensional
    vector that determines the label of **independent** binary decisions, i.e.,
    we aim to minimize

    .. math::

        \text{NLL}_n^{(t)} = - \log p(y_n^{(t)} \mid \mathbf{h}_n^{(t-1)}, W) \
            = - \sum_{c=1}^C \log p(y_n^{(t,c)} \mid \mathbf{h}_n^{(t-1)}, W)

    Again, we consider :math:`z_n^{(t)}` as the ``C``-dimensional logit output
    of the network and define the likelihood per decision via a sigmoid

    .. math::

        \tilde{z}_n^{(t,c)} = \text{sigmoid}(\beta^{(t,c)} z_n^{(t,c)})

    where :math:`\beta^{(t,c)}` is the inverse temperature per
    timestep/decision.

    We can then define the likelihood using the binary-cross entropy

    .. math::

        \text{NLL}_n^{(t)} = \sum_{c=1}^C \big( \
             - y_n^{(t,c)} \log \tilde{z}_n^{(t,c)} \
             - (1 - y_n^{(t,c)}) \log (1 - \tilde{z}_n^{(t,c)}) \
        \big)

    The same considerations as for the usual cross-entropy (see above) apply
    here for setting the inverse temperature.

    Note:
        All loss types require network outputs to be provided as logits!

    Args:
        loss_type (str): Determines which kind of loss function is used. The
            following options are available:

            - ``'ce'``: A function handle that uses the cross-entropy loss is
              returned.
            - ``'bce'``: A function handle that uses the binary cross-entropy
              loss is returned.
            - ``'mse'``: A function handle that uses the mean-squared-error loss
              is returned.
        reduction (str): Whether the NLL loss should be summed ``'sum'`` or
            meaned ``'mean'`` across the batch dimension.

    Returns:
        (func): A function handle as, for instance, requested by option
        ``custom_nll`` of function :func:`utils.ewc_regularizer.compute_fisher`.

        Based on the chosen ``loss_type``, the returned functions might have
        different signatures in terms of keyword arguments. Though, the
        positional arguments are the same for each function.

        - ``'ce'``: A function with the following signature is returned

          .. code-block:: python

              nll(Y, T, data, allowed_outputs, empirical_fisher,
                  ts_factors=None, mask=None)
        - ``'bce'``: A function with the following signature is returned

          .. code-block:: python

              nll(Y, T, data, allowed_outputs, empirical_fisher,
                  ts_factors=None, beta=None)
        - ``'mse'``: : A function with the following signature is returned

          .. code-block:: python

              nll(Y, T, data, allowed_outputs, empirical_fisher,
                  ts_factors=None, beta=None)

        where the positional arguments ``Y``, ``T``, ``data``,
        ``allowed_outputs`` and ``empirical_fisher`` are defined in the
        docstring argument ``custom_nll`` of function
        :func:`utils.ewc_regularizer.compute_fisher`.

        Note:
            Just like function :func:`utils.ewc_regularizer.compute_fisher`
            prescribes, we assume that ``allowed_outputs`` has already been
            applied to ``Y``.

        Note:
            The current implementation of the loss functions does not make use
            of arguments ``data``, ``allowed_outputs`` and ``empirical_fisher``,
            which therefore can be passed as ``None``.

        The keyword arguments have the following meaning:

        - **ts_factors** (torch.Tensor, optional): A list of factors,
          one for each timestep
          :math:`\log p(y_n^{(t)} \mid y_n^{(1)}, \dots, y_n^{(t-1)}, W)`.
          These factors are multiplied before the timesteps are summed together.
          The tensor should be of shape ``[num_timesteps, 1]`` or
          ``[num_timesteps, batch_size]``.

          Note:
              Setting ``ts_factors`` to ``0`` should be identical to setting
              ``mask`` or ``beta`` values to zero (at least the gradient
              computation through the loss is identical, the actuall NLL value
              might be different. Keep in mind, that for the MSE loss the NLL is
              anyway only computed up to additive constants.
        - **mask** (torch.Tensor, optional): The mask :math:`m^{(t,c)}` that can
          be applied per timestep/channel. The shape of the mask must either
          be identical to the one of ``Y`` or broadcastable with respect to
          ``Y``.
        - **beta** (torch.Tensor, optional): Contains the inverse temperatures
          :math:`\beta^{(t)}` per timestep for the ``'ce'`` loss and the
          inverse temperatures per timestep/channel for the ``'bce'`` loss.
          The shape of ``beta`` must be broadcastable with respect to ``Y``.
          In addition, ``beta`` is expected to fulfill ``beta.shape[2] == 1``
          in case of the ``'ce'`` loss.
    """
    assert reduction in ['sum', 'mean']
    assert loss_type in ['ce', 'bce', 'mse']

    def custom_nll_mse(Y, T, data, allowed_outputs, empirical_fisher,
                       ts_factors=None, mask=None):
        assert np.all(np.equal(list(Y.shape), list(T.shape)))

        nll = (Y - T)**2
        if mask is not None:
            nll = mask * nll

        nll = nll.mean(dim=2)

        if ts_factors is not None:
            assert len(ts_factors.shape) == 2
            nll = ts_factors * nll

        if reduction == 'mean':
            nll = nll.mean()
        else:
            nll = nll.mean(dim=0).sum()

        return nll

    def custom_nll_ce(Y, T, data, allowed_outputs, empirical_fisher,
                      ts_factors=None, beta=None):
        # We expect targets to be either given as labels or as 1-hot encodings.
        assert len(Y.shape) == 3 and \
            np.all(np.equal(list(Y.shape[:2]), list(T.shape[:2]))) and \
            (len(T.shape) == 2 or Y.shape[2] == T.shape[2])

        if len(T.shape) == 2:
            labels = T
        else:
            labels = torch.argmax(T, 2)

        if beta is None:
            log_sm = F.log_softmax(Y, dim=2)
        else:
            log_sm = F.log_softmax(Y * beta, dim=2)
        # We need to swap dimensions from [T, B, C] to [T, C, B].
        # See documentation of method `F.nll_loss`.
        log_sm = log_sm.permute(0, 2, 1)
        nll = F.nll_loss(log_sm, labels, reduction='none')
        assert len(nll.shape) == 2

        if ts_factors is not None:
            assert len(ts_factors.shape) == 2
            nll = ts_factors * nll

        # Sum across time-series dimension.
        nll = nll.sum(dim=0)

        if reduction == 'mean':
            nll = nll.mean()
        else:
            nll = nll.sum()

        return nll

    def custom_nll_bce(Y, T, data, allowed_outputs, empirical_fisher,
                       ts_factors=None, beta=None):
        # T is expected to be binary vector.
        assert np.all(np.equal(list(Y.shape), list(T.shape)))

        if beta is None:
            nll = F.binary_cross_entropy_with_logits(Y, T, reduction='none')
        else:
            nll = F.binary_cross_entropy_with_logits(beta * Y, T,
                                                     reduction='none')
        assert len(nll.shape) == 3

        # Sum accross channel dimension.
        nll = nll.sum(dim=2)

        if ts_factors is not None:
            assert len(ts_factors.shape) == 2
            nll = ts_factors * nll

        # Sum across time-series dimension.
        nll = nll.sum(dim=0)

        if reduction == 'mean':
            nll = nll.mean()
        else:
            nll = nll.sum()

        return nll

    if loss_type == 'ce':
        return custom_nll_ce
    elif loss_type == 'bce':
        return custom_nll_bce
    else:
        assert loss_type == 'mse'
        return custom_nll_mse

def generate_binary_masks(config, device, tnet):
    """Generate a set of binary masks for individually modulating the main
    network for each task.

    Args:
        config: Command-line arguments.
        device: PyTorch device.
        tnet: The target network.

    Returns:
        (list): A list gains and shifts (one per task) that implement
        binary masks.
    """
    assert tnet.context_mod_layers is not None and \
        len(tnet.context_mod_layers) > 0

    # For simplicity, we assume the following is true, otherwise, the creation
    # of masks would be unnecessarily complicated.
    if isinstance(tnet, SimpleRNN):
        # FIXME We should also ensure that this is the case for BiRNNs!
        assert tnet._context_mod_num_ts == -1 or \
            tnet._context_mod_separate_layers_per_ts

    # Note, we don't want to silence inputs or permanently shut-off output
    # neurons.
    assert not config.context_mod_inputs and config.no_context_mod_outputs

    ctx_masks = []

    for t in range(config.num_tasks):
        gain_shifts = []
        for cm_layer in tnet.context_mod_layers:
            assert not cm_layer.gain_softplus_applied
            assert len(cm_layer._num_features) == 1
            assert not cm_layer._no_gains and not cm_layer._no_shifts
            n_units = cm_layer._num_features[0]

            idx = np.arange(n_units)
            np.random.shuffle(idx)
            n_masked = int(n_units * config.mask_fraction)
            mask_idx = idx[0:n_masked]

            mask = torch.ones(n_units)
            mask[mask_idx] = 0

            if cm_layer.gain_offset_applied:
                mask -= 1

            # Since we use the context_mod infrastructure, we have to specify
            # the binary masks as gains and shifts. To do this we only set the
            # gains of the hidden units.
            gain_shifts.append(mask.to(device))
            gain_shifts.append(torch.zeros(n_units).to(device))

        ctx_masks.append(gain_shifts)

    return ctx_masks

def extract_hh_weights(mnet, hnet_out=None):
    """Extract hidden-to-hidden weights.

    This function extracts the hidden-to-hidden weights that are either
    internally maintained or produced by a hypernetwork.

    Note, if the main network uses LSTM weights, then this function
    automatically decomposes them since they are stored in a concatenated form.

    Args:
        mnet (mnets.simple_rnn.SimpleRNN): The main network.
        hnet_out (list, optional): If applicable, the hypernetwork output.

    Returns:
        (list): A list with the extracted weight tensors.
    """
    ret = []

    if mnet.internal_params is not None:
        for meta in mnet.param_shapes_meta:
            if meta['name'] == 'weight' and 'info' in meta.keys() and \
                    meta['info'] == 'hh' and meta['index'] != -1:
                ret.append(mnet.internal_params[meta['index']])

    if hnet_out is not None:
        assert len(hnet_out) == len(mnet.hyper_shapes_learned)

        for i, sind in enumerate(mnet.hyper_shapes_learned_ref):
            meta  = mnet.param_shapes_meta[sind]
            if meta['name'] == 'weight' and 'info' in meta.keys() and \
                    meta['info'] == 'hh':
                ret.append(hnet_out[i])

    assert len(ret) == mnet.num_rec_layers

    # LSTM weight matrices are stored such that the hidden-to-hidden matrices
    # for the 4 gates are concatenated.
    if mnet.use_lstm:
        tmp_ret = ret
        ret = []
        for mat in tmp_ret:
            out_dim, _ = mat.shape
            # FIXME Would be nicer if we have direct access to the feature
            # sizes.
            assert out_dim % 4 == 0
            fs = out_dim // 4

            ret.append(mat[:fs, :])
            ret.append(mat[fs:2*fs, :])
            ret.append(mat[2*fs:3*fs, :])
            ret.append(mat[3*fs:, :])

    return ret

def orthogonal_regularizer(config, mnet, hnet_out=None):
    r"""Compute orthogonal regularizer.

    .. math::

        \lambda \sum_i \lVert W_i^T W_i - I \rVert^2

    This function computes the orthogonal regularizer for all hidden-to-hidden
    weights as returned by function :func:`extract_hh_weights`.

    Args:
        config (argparse.Namespace): The user configuration.
        mnet (mnets.simple_rnn.SimpleRNN): The main network.
        hnet_out (list, optional): If applicable, the hypernetwork output.

    Returns:
        (torch.Tensor): A scalar tensor representing the orthogonal regularizer.
    """
    assert config.orthogonal_hh_reg > 0

    # Weights to be regularized.
    weights = extract_hh_weights(mnet, hnet_out=hnet_out)

    reg = 0
    for W in weights:
        # Compute Frobenius norm of W^T W - I
        reg += torch.norm(torch.matmul(W, W.transpose(0,1)) - \
                          torch.eye(W.shape[0], device=W.device))**2

    return config.orthogonal_hh_reg * reg

def ckpt_filenames(out_dir, task_id=None, train_iter=None):
    """Filenames for model checkpoints.

    Helper function of functions :func:`save_models` and :func:`load_models`.

    Args:
        (....): See docstring of function :func:`save_models`.

    Returns:
        (tuple): Tuple containing:

        - **ckpt_mnet_fn** (str): Checkpoint filename for main network
        - **ckpt_hnet_fn** (str): Checkpoint filename for hypernetwork
        - **ckpt_dnet_fn** (str): Checkpoint filename for decoder network.
        - **ckpt_wembs_fn** (str): Checkpoint filename for word embeddings.
    """
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    if task_id is None and train_iter is None:
        ckpt_mnet_fn = os.path.join(ckpt_dir, 'final_mnet')
        ckpt_hnet_fn = os.path.join(ckpt_dir, 'final_hnet')
        ckpt_dnet_fn = os.path.join(ckpt_dir, 'final_dnet')
        ckpt_wembs_fn = os.path.join(ckpt_dir, 'final_wembs')
    elif train_iter is None:
        ckpt_mnet_fn = os.path.join(ckpt_dir, 'mnet_task_%d' % task_id)
        ckpt_hnet_fn = os.path.join(ckpt_dir, 'hnet_task_%d' % task_id)
        ckpt_dnet_fn = os.path.join(ckpt_dir, 'dnet_task_%d' % task_id)
        ckpt_wembs_fn = os.path.join(ckpt_dir, 'wembs_task_%d' % task_id)
    else:
        ext = '' if task_id is None else '_%d' % task_id
        ckpt_mnet_fn = os.path.join(ckpt_dir, 'current_mnet%s' % ext)
        ckpt_hnet_fn = os.path.join(ckpt_dir, 'current_hnet%s' % ext)
        ckpt_dnet_fn = os.path.join(ckpt_dir, 'current_dnet%s' % ext)
        ckpt_wembs_fn = os.path.join(ckpt_dir, 'current_wembs%s' % ext)

    return ckpt_mnet_fn, ckpt_hnet_fn, ckpt_dnet_fn, ckpt_wembs_fn

def save_models(out_dir, logger, performance, mnet, hnet=None, dnet=None,
                wembs=None, task_id=None, train_iter=None, **kwargs):
    """Checkpoint the main and (if existing) the current hypernet.

    Args:
        out_dir (str): The simulation output directory.
        logger: Console (and file) logger.
        performance (float): The network its test performance. If ``task_id``
            is provided, then this quantity is assumed to represent the during
            accuracy of the corresponding task. Otherwise, it is assumed to be
            the average final accuracy.
        mnet: The main network.
        hnet (optional): The hypernetwork.
        dnet (optional): The decoder (replay) network.
        wembs (list, optional): List of instances of class
            :class:`sequential.embedding_utils.WordEmbLookup`.
        task_id (int, optional): Task ID of the last task the models have been
            trained on. Will be integrated into the filenames.
        train_iter (int, optional): The current training iteration. If provided,
            ``performance`` is expected to be the current validation
            performance. If ``task_id`` is provided, it will still be integrated
            into the filename.
        **kwargs: Keyword arguments passed to function
            :func:`utils.torch_ckpts.save_checkpoint`.
    """
    ckpt_mnet_fn, ckpt_hnet_fn, ckpt_dnet_fn, ckpt_wembs_fn = ckpt_filenames( \
        out_dir, task_id=task_id, train_iter=train_iter)

    if task_id is None and train_iter is None:
        perf_key = 'avg_final_acc'
    elif train_iter is None:
        perf_key = 'during_acc_task_%d' % task_id
    else:
        perf_key = 'val_performance'

    perf_dict = {perf_key: performance}
    if train_iter is not None:
        perf_dict['train_iter'] = train_iter

    ts = time()

    logger.debug('Checkpointing main network in %s ...' % ckpt_mnet_fn)
    tckpt.save_checkpoint({'state_dict': mnet.state_dict(),
                           **perf_dict},
        ckpt_mnet_fn, performance, train_iter=train_iter,
        timestamp=ts, **kwargs)
    if hnet is not None:
        logger.debug('Checkpointing hypernetwork in %s ...' % ckpt_hnet_fn)
        tckpt.save_checkpoint({'state_dict': hnet.state_dict(),
                               **perf_dict},
            ckpt_hnet_fn, performance, train_iter=train_iter,
            timestamp=ts, **kwargs)
    if dnet is not None:
        logger.debug('Checkpointing decoder network in %s ...' % ckpt_dnet_fn)
        tckpt.save_checkpoint({'state_dict': dnet.state_dict(),
                               **perf_dict},
            ckpt_dnet_fn, performance, train_iter=train_iter,
            timestamp=ts, **kwargs)
    if wembs is not None:
        # FIXME no need to always checkpoint all word embeddings, as only the
        # ones from the current task have been modified.
        tmp_wembs = torch.nn.ModuleList(wembs)
        logger.debug('Checkpointing word embeddings in %s ...' % ckpt_wembs_fn)
        tckpt.save_checkpoint({'state_dict': tmp_wembs.state_dict(),
                               **perf_dict},
            ckpt_wembs_fn, performance, train_iter=train_iter,
            timestamp=ts, **kwargs)


def load_models(out_dir, device, logger, mnet, hnet=None, dnet=None, wembs=None,
                task_id=None, train_iter=None, return_models=False):
    """Load checkpointed networks.

    Note, if ``train_iter`` is ``-1``, the best checkpoint is loaded.

    Args:
        (....): See docstring of function :func:`save_models`.
        device: PyTorch device.
        return_models (boolean, optional): Whether to return the models.

    Returns:
        (float or tuple): Where the tuple is containing:

        - **score** (float): The performance score of the checkpoint.
        - **mnet**: If ``return_models`` is ``True``, then the main network is 
          returned. If ``return_models`` is ``False``, then only ``score`` is 
          returned.
        - **hnet**: If ``return_models`` is ``True``, then the hypernetwork is 
          returned. If ``return_models`` is ``False``, then only ``score`` is 
          returned.
        - **dnet**: If ``return_models`` is ``True``, then the replay decoder is 
          returned. If ``return_models`` is ``False``, then only ``score`` is 
          returned.
    """
    ckpt_mnet_fn, ckpt_hnet_fn, ckpt_dnet_fn, ckpt_wembs_fn = ckpt_filenames( \
        out_dir, task_id=task_id, train_iter=train_iter)
    if train_iter is not None:
        if train_iter == -1:
            ckpt_mnet_fn = tckpt.get_best_ckpt_path(ckpt_mnet_fn)
            ckpt_hnet_fn = tckpt.get_best_ckpt_path(ckpt_hnet_fn)
            ckpt_dnet_fn = tckpt.get_best_ckpt_path(ckpt_dnet_fn)
            ckpt_wembs_fn = tckpt.get_best_ckpt_path(ckpt_wembs_fn)
        else:
            raise NotImplementedError()

    ckpt_dict, score = tckpt.load_checkpoint(ckpt_mnet_fn, mnet, device=device,
                                             ret_performance_score=True)
    if hnet is not None:
        ckpt_dict_hnet, _ = tckpt.load_checkpoint(ckpt_hnet_fn, hnet,
            device=device, ret_performance_score=True)
    if dnet is not None:
        ckpt_dict_dnet, _ = tckpt.load_checkpoint(ckpt_dnet_fn, dnet,
            device=device, ret_performance_score=True)
    if wembs is not None:
        tmp_wembs = torch.nn.ModuleList(wembs)
        ckpt_dict_wembs, _ = tckpt.load_checkpoint(ckpt_wembs_fn, tmp_wembs,
            device=device, ret_performance_score=True)

    if task_id is None and train_iter is None:
        logger.debug('Loaded network(s) with an average final accuracy of ' +
                     '%f%%.' % score)
    elif train_iter is None:
        logger.debug('Loaded network(s) for task %d from checkpoint,' \
                     % task_id + 'that has a during accuracy of %f%%.' % score)
    else:
        train_iter = ckpt_dict['train_iter']
        if hnet is not None:
            assert train_iter == ckpt_dict_hnet['train_iter']
        if dnet is not None:
            assert train_iter == ckpt_dict_dnet['train_iter']
        if wembs is not None:
            assert train_iter == ckpt_dict_wembs['train_iter']

        if task_id is None:
            logger.info('Restored network(s) from training iteration %d ' \
                        % train_iter + 'with a validation accuracy of %f%%.' \
                        % score)
        else:
            logger.info('Restored network(s) for task %d from ' % task_id +
                        'training iteration %d ' % train_iter +
                         'with a validation accuracy of %f%%.' % score)

    # FIXME Why would we wanna return the models? They have been modified
    # in-place.
    if return_models:
        warn('Option "return_models" is deprecated. Passed models are ' +
             'modified in place.' , DeprecationWarning)
        return score, mnet, hnet, dnet
    else:
        return score

def out_units_of_task(config, data, task_id, dhandlers=None,
                      trained_task_id=None):
    """The output units that belong to task `task_id``.

    Based on the user config, compute the indices of all output neurons of the
    main network that belong to the given task.

    Note:
        This function assumes that starting at index 0 all output units of the
        main network are considered as actual outputs wrt the dataset.
        There might be more output units for other usage (e.g., the replay
        latent space), that are ignored by this function.

    Args:
        config (argparse.Namespace): Command-line arguments.
        data (data.dataset.Dataset): A dataset handler.

            Note:
                It is assumed that all handlers have the same number of classes!
        task_id (int): The ID of the task for which the output units should be
            computed (note, first task has ID ``0``).
        dhandlers (list, optional): List of all datahandlers. This option is
            not required as long as all tasks have the same output
            dimensionality. It will be used to perform sanity checks.
        trained_task_id (int, optional): If provided, a growing softmax is
            assumed, i.e., if ``config.all_task_softmax`` is ``True``, then
            ``trained_task_id`` is assumed to be the last task that has been
            trained on and the output unit IDs only up to this task are
            returned (hence, the output head is growing with
            ``trained_task_id``).

    Returns:
        (list): A list of integers, each denoting the index of an output neuron
        of the main network belonging to this task (the list contains
        consecutive indices).
    """
    if data.classification:
        # Softmax output.
        num_outs = data.num_classes
    else:
        assert len(data.out_shape) == 1
        num_outs = data.out_shape[0]

    num_outs_per_task = None
    if dhandlers is not None: # Sanity checks.
        assert data is dhandlers[task_id]
        assert len(dhandlers) == config.num_tasks
        num_outs_per_task = []
        for dh in dhandlers:
            if dh.classification:
                num_outs_per_task.append(dh.num_classes)
            else:
                assert len(dh.out_shape) == 1
                num_outs_per_task.append(dh.out_shape[0])
        num_outs_per_task = np.array(num_outs_per_task)
        if not np.all(np.equal(num_outs_per_task, num_outs)):
            # FIXME "dhandlers" should not be optional if we allow this.
            # Also, single-head would not be possible anymore.
            raise ValueError('This code was not designed to work with tasks ' +
                             'that have varying output sizes.')

    num_prev = num_outs * task_id
    num_total = num_outs * config.num_tasks
    if num_outs_per_task is not None:
        num_prev = np.sum(num_outs_per_task[:task_id])
        num_total = num_outs_per_task.sum()

    assert hasattr(config, 'multi_head')
    if hasattr(config, 'all_task_softmax') and config.all_task_softmax:
        # A growing softmax doesn't make sense otherwise.
        assert data.classification
        # DELETEME To avoid programming bugs, I add this assertion since I know
        # I only want to consider the growing softmax case for now.
        # FIXME Has to be modified if allowing also multitask learning with
        # `all_task_softmax` (since the softmax is not growing).
        assert trained_task_id is not None
        assert not config.multi_head and config.use_replay

        if trained_task_id is not None:
            num_curr_total = num_outs * (trained_task_id+1)
            if num_outs_per_task is not None:
                num_curr_total = np.sum(num_outs_per_task[:trained_task_id+1])

            ret = range(num_curr_total)
        else:
            # All output neurons have to be considered.
            ret = range(num_total)
    elif config.multi_head:
        ret = range(num_prev, num_prev + num_outs)
    else: # Single head
        ret = range(num_outs)

    return list(ret)

def adjust_targets_to_head(config, data, targets, task_id, dhandlers=None,
                           trained_task_id=None, is_one_hot=True):
    """Adjust target outputs to the actual size of the output head.

    Note, this function won't change the targets in a single- or multi-head
    setting. Only if a softmax configuration is chosen, that spans more
    than one head, the targets have to be zero-padded.

    Args:
        (....): See docstring of function :func:`out_units_of_task`.
        targets (torch.Tensor): The targets outputs, assuming a single head
            would be used.
        is_one_hot (bool): Whether the target outputs are 1-hot encoded.

    Returns:
        (torch.Tensor): The zero-padded ``targets``, if necessary.
    """
    if not hasattr(config, 'all_task_softmax') or not config.all_task_softmax:
        return targets

    assert data.classification
    assert len(targets.shape) == 3

    num_outs = data.num_classes

    #num_outs_per_task = None
    #if dhandlers is not None: # Sanity checks.
    #    assert data is dhandlers[task_id]
    #    assert len(dhandlers) == config.num_tasks
    #    num_outs_per_task = []
    #    for dh in dhandlers:
    #        assert dh.classification
    #        num_outs_per_task.append(dh.num_classes)
    #    num_outs_per_task = np.array(num_outs_per_task)

    num_total = num_outs * config.num_tasks
    num_prev = num_outs * task_id
    #if num_outs_per_task is not None:
    #    num_prev = np.sum(num_outs_per_task[:task_id])
    #    num_total = num_outs_per_task.sum()

    # DELETEME To avoid programming bugs, I add this assertion since I know
    # I only want to consider the growing softmax case for now.
    # FIXME Has to be modified if allowing also multitask learning with
    # `all_task_softmax` (since the softmax is not growing).
    assert trained_task_id is not None

    if trained_task_id is not None:
        if trained_task_id < task_id:
            raise RuntimeError('In a growing softmax, outputs of future ' +
                               'tasks, cannot be retrieved.')

        num_total = num_outs * (trained_task_id+1)
        #if num_outs_per_task is not None:
        #    num_total = np.sum(num_outs_per_task[:trained_task_id+1])


    if is_one_hot:
        assert targets.shape[2] == data.num_classes

        targets = F.pad(targets, (num_prev, num_total-(num_prev+num_outs)),
                        mode='constant', value=0)

    else:
        assert targets.shape[2] == 1

        label_offset = num_prev
        targets += label_offset

    return targets

def preprocess_inputs(config, shared, inputs, task_id):
    """Preprocess a batch of inputs from the datahandler before inputting it
    into the classifier.

    Things that are done in this function:

    - If word embeddings are used (attribute ``shared.word_emb_lookups``), then
      inputs are first translated using the word embeddings corresponding to
      ``task_id``.
    - If ``config.input_task_identity`` is set, then a 1-hot encoding is
      appended to the feature dimension, where the ``task_id``-th entry is set
      to ``1``.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Miscellaneous information shared across
            functions.
        inputs (torch.Tensor): Batch of input samples.
        task_id (int): ID of the task from which that inputs stem.

    Returns:
        (torch.Tensor): Preprocessed ``inputs``.
    """
    if hasattr(shared, 'word_emb_lookups'):
        inputs = shared.word_emb_lookups[task_id].forward(inputs)

    if config.input_task_identity:
        assert len(inputs.shape) == 3
        # Add a one-hot-encoding of task identity.
        inputs = F.pad(inputs, (0, config.num_tasks), mode='constant', value=0)
        inputs[:,:,-config.num_tasks+task_id] = 1.

    return inputs

def hnet_forward(config, hnet, task_id):
    """Create weights for task ``task_id`` using the hypernet ``hnet``.

    This function simply ensures the correct use of the hypernetwork interface,
    based on the type of hypernetwork (old or new).

    Args:
        config (argparse.Namespace): Command-line arguments.
        hnet: The hypernetwork.
        task_id (int): Task ID for which weights should be generated.

    Returns:
        (list): Result of ``forward`` computation.
    """
    if config.use_new_hnet:
        return hnet.forward(cond_id=task_id)
    else:
        return hnet.forward(task_id=task_id)

def update_coresets(config, shared, task_id, data):
    """Create a new coreset for task ``task_id``.

    This function extracts a random subset of the training set from the data
    handler ``data`` and stores it as a coreset. The coresets are stored in
    the container ``shared``.

    In addition, the sample IDs of the corresponding coreset inputs are stored
    in ``shared.coreset_sample_ids``.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Miscellaneous information shared across
            functions.
        task_id (int): The task ID associated to the corset samples
        data: Dataset loader. New data will be added to the coreset from the
            training set of this data loader.
    """
    assert hasattr(config, 'coreset_size')
    if config.coreset_size == -1:
        return

    if task_id > 0:
        assert hasattr(shared, 'coresets') and len(shared.coresets) == task_id
    else:
        shared.coresets = []
        shared.coreset_sample_ids = []

    # Pick random samples from the training set as new coreset.
    batch = data.next_train_batch(config.coreset_size, return_ids=True)
    # We don't transform them to tensors yet, as random data augmentation might
    # be applied in this step.
    #coreset = data.input_to_torch_tensor(batch[0], device, mode='train')
    coreset = batch[0]

    shared.coresets.append(coreset)
    shared.coreset_sample_ids.append(batch[2])

def get_target_net_weight_masks(target_net, weight_type='rec', device=None):
    """Return a mask for the recurrent main network weights that selects
    either recurrent or feedforward weights.

    Depending on the type of weight requested (`rec` or `ff`),
    this function returns a list of tensors akin to ``target_net.param_shapes``
    with values equal to 1 for weights that are of the requested type (
    recurrent or feedforward respectively), and 0 for the other weights.

    Args:
        target_net: The target recurrent network.
        weight_type (optional, str): The type of weight of interest. Options
            are `recurrent` or `forward`.
        device (optional): PyTorch device of returned mask.

    Return:
        (list): The corresponding mask for each tensor in the main network.
    """
    weight_masks = []
    for i, weights_shape in enumerate(target_net.param_shapes):

        weight_is_recurrent = 'info' in target_net.param_shapes_meta[i].keys() \
                        and target_net.param_shapes_meta[i]['info'] == 'hh'

        # Masks are zeros unless the current weights are of the requested type.
        mask = torch.zeros(weights_shape)
        if weight_type == 'rec' and weight_is_recurrent:
            mask = torch.ones(weights_shape)
        elif weight_type == 'ff' and not weight_is_recurrent:
            mask = torch.ones(weights_shape)
        if device is not None:
            mask = mask.to(device)
        weight_masks.append(mask)

    return weight_masks
