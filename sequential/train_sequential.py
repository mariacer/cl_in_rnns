#!/usr/bin/env python3
# Copyright 2019 Maria Cervera, Benjamin Ehret

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
# @title           :sequential/train_sequential.py
# @author          :mc, be
# @contact         :mariacer@ethz.ch
# @created         :30/10/2019
# @version         :1.0
# @python_version  :3.6.8
"""
Training functions for Continual Learning methods on Sequential Data
--------------------------------------------------------------------

Set of functions to train an RNN (with or without hypernetwork) on a set of
sequential tasks.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn.functional as F

from data.timeseries.copy_data import CopyTask
from data.timeseries.mud_data import MUDData
from data.timeseries.smnist_data import SMNISTData
from mnets.bi_rnn import BiRNN
from mnets.simple_rnn import SimpleRNN
import sequential.replay_utils as rtu
import sequential.plotting_sequential as plc
import sequential.train_utils_sequential as stu
import sequential.copy.train_utils_copy as tu_copy
from sequential.pos_tagging.train_utils_pos import compute_f_score
import sequential.smnist.train_utils_smnist as tu_smnist
import utils.hnet_regularizer as hreg
import utils.ewc_regularizer as ewc
import utils.optim_step as opstep
import utils.si_regularizer as si
from utils.torch_utils import get_optimizer
from utils import misc

def test(dhandlers, device, config, shared, logger, writer, target_net, hnet,
         ctx_masks=None, store_activations=False, plot_output=False,
         accuracy_func=None, task_loss_func=None, num_trained=None,
         return_acc_per_ts=False):
    """Test networks on all tasks and return list of accuracies.

    Args:
        (....): See docstring of function :func:`train_tasks`.
        store_activations (bool): Whether hidden activations should be stored.
        plot_output (bool, optional): Plot the outputs of the network for all
            tasks.
        task_loss_func (fct_handler): The function to compute task loss.
        accuracy_func (fct_handler): The function to compute the accuracy, only
            necessary for classification tasks.
        num_trained (int, optional): The number of tasks that have been trained
            on so far. If a growing softmax is used, then it is important to
            provide this option.
        return_acc_per_ts (boolean, optional): If True, the accuracy per 
            timestep will also be computed. Only implemented for copy task
            currently.

    Returns:
        (tuple): Tuple containing:

        - **losses** (list): The test losses on each task.
        - **accuracies** (list): The test accuracies on each task for
          classification tasks only.
        - **accuracies_per_ts** (list): The test accuracies per timestep on each 
          task for classification tasks only. Note that this element is only
          returned if the option "return_acc_per_ts" is enabled. Besides, this
          calculation only makes sense for tasks that have sequences with
          identical lengths.
    """
    target_net.eval()
    if hnet is not None:
        hnet.eval()

    if hasattr(config, 'all_task_softmax') and config.all_task_softmax:
        # For replay methods, we assume a growing softmax.
        assert num_trained is not None

        if config.multitask:
            # No growing softmax in this case.
            assert num_trained == config.num_tasks

    # Create list to contain activation values.
    hidden_activations = []
    hidden_activations_int = []
    targets = []
    modulations= []

    if plot_output:
        outputs = []

    with torch.no_grad():
        losses = []
        accuracies = None
        accuracies_per_ts = None
        if config.classification:
            accuracies = []
            accuracies_per_ts = []
        for t in range(config.num_tasks):
            if hasattr(config, 'all_task_softmax') and config.all_task_softmax:
                if t >= num_trained:
                    # When using a growing softmax, we cannot test on future
                    # tasks.
                    losses.append(-1.)
                    if config.classification:
                        accuracies.append(-1.)

                    logger.debug('Skipping testing of task %d.' % t)
                    if store_activations:
                        logger.warn('No activations for task %d will be ' % t +
                                    'stored!')
                    continue

            # We need to tell the target network, which context-mod weights to
            # use, in case we checkpoint them for every task.
            tnet_kwargs = {}
            if config.use_context_mod and config.checkpoint_context_mod:
                # Note, there are no checkpointed weights for future tasks, so
                # we just use the current ones.
                if num_trained is None:
                    # FIXME A bit hacky - on how many tasks did we train
                    # already?
                    num_trained = \
                        int(target_net.context_mod_layers[0].num_ckpts)
                if t >= num_trained:
                    # Use current, untrained context-mod weights.
                    tnet_kwargs['condition'] = num_trained
                else:
                    # Load checkpoint.
                    tnet_kwargs['condition'] = t

            # reset data
            dhandler = dhandlers[t]
            dhandler.reset_batch_generator()

            # get all test data for this task
            sample_ids = dhandler.get_test_ids()
            X = dhandler.input_to_torch_tensor( \
                dhandler.get_test_inputs(), device, mode='inference',
                sample_ids=sample_ids)
            X = stu.preprocess_inputs(config, shared, X, t)

            T = dhandler.output_to_torch_tensor( \
                dhandler.get_test_outputs(), device, mode='inference',
                sample_ids=sample_ids)
            T = stu.adjust_targets_to_head(config, dhandler, T, t,
                dhandlers=dhandlers, trained_task_id=num_trained-1,
                is_one_hot=True)

            use_replay = hasattr(config, 'use_replay') and config.use_replay
            if hnet is not None and not use_replay:
                # TODO pass `dnet` and check:
                #assert dnet is None
                ext_tnet_weights = stu.hnet_forward(config, hnet, t)
            else:
                ext_tnet_weights = None
            if config.use_masks:
                ext_tnet_weights = ctx_masks[t]

            allowed_outputs = stu.out_units_of_task(config, dhandler, t,
                dhandlers=dhandlers, trained_task_id=num_trained-1)

            # get predictions
            if isinstance(target_net, SimpleRNN):
                Y_logits, hidden, hidden_int = target_net.forward(X, \
                    weights=ext_tnet_weights, return_hidden=True,
                    return_hidden_int=True, **tnet_kwargs)
            else:
                tnet_kwargs['seq_lengths'] = \
                    dhandler.get_in_seq_lengths(sample_ids)
                if store_activations:
                    raise NotImplementedError()
                Y_logits = target_net.forward(X, \
                    weights=ext_tnet_weights, **tnet_kwargs)
            Y_logits = Y_logits[:, :, allowed_outputs]

            if plot_output:
                outputs.append(Y_logits.cpu())
            targets.append(T.cpu())

            # append activations
            if store_activations:
                assert len(hidden) == len(hidden_int)

                # Concatenate hidden activities of all layers into single tensor
                all_hidden = torch.tensor(())
                all_hidden_int = torch.tensor(())
                for h, h_int in zip(hidden, hidden_int):
                    all_hidden = torch.cat((all_hidden, h.cpu()), 2)
                    all_hidden_int = torch.cat((all_hidden_int, h_int.cpu()), 2)

                hidden_activations.append(all_hidden.cpu())
                hidden_activations_int.append(all_hidden_int.cpu())
                targets.append(T.cpu())
                if config.use_context_mod:
                    if hnet is not None:
                        gain_shifts = []
                        for ii, p in enumerate(ext_tnet_weights):
                            i_ps = target_net.hyper_shapes_learned_ref[ii]
                            meta = target_net.param_shapes_meta[i_ps]
                            if meta['name'] == 'cm_scale' or \
                                    meta['name'] == 'cm_shift':
                                gain_shifts.append(p)
                    else:
                        # Note, checkpointed context-mod weights have to be
                        # retrieved, not necessarily the current ones.
                        raise NotImplementedError()

                    gs_cpu = [gs.cpu() for gs in gain_shifts]
                    modulations.append(gs_cpu)

            losses.append(task_loss_func(Y_logits, T, dhandler, None, None,
                sample_ids) / dhandler.num_test_samples)
            if config.classification:
                classifier_accuracy, classifier_accuracy_per_ts = \
                    accuracy_func(Y_logits, T, dhandler, sample_ids)
                accuracies.append(classifier_accuracy)
                accuracies_per_ts.append(classifier_accuracy_per_ts)
            if isinstance(dhandler, MUDData) and num_trained is not None:
                f_score = compute_f_score(Y_logits, T, dhandler, sample_ids)
                if shared.f_scores is None and t == 0:
                    shared.f_scores = np.zeros((1, config.num_tasks))
                elif t == 0:
                    shared.f_scores = np.concatenate((shared.f_scores,
                        np.zeros((1, config.num_tasks))), axis=0)
                shared.f_scores[num_trained-1, t] = f_score
                if t < num_trained:
                    logger.info('F-score of task %d after learning task %d: ' \
                                % (t, num_trained-1) + '%.4f.' % f_score)

    if store_activations:
        modulations=None if len(modulations) == 0 else modulations
        stu.save_activations(config, hidden_activations, targets=targets,
                         modulations=modulations)
        stu.save_activations(config, hidden_activations_int, name_prefix='int')

    if plot_output:
        plc.plot_outputs(outputs, targets, config)

    if return_acc_per_ts:
        return losses, accuracies, accuracies_per_ts
    else:
        return losses, accuracies


def evaluate(config, shared, logger, writer, device, task_id, data, mnet, hnet,
             dnet, train_iter, mnet_kwargs, ctx_masks=None, accuracy_func=None,
             task_loss_func=None, num_trained=None):
    """Evaluate training progress on validation set.

    Args:
        (....): See docstring of function :func:`train_tasks` and :func:`test`.
        task_id (int): Id of current task (that's being traing on).
        data (data.dataset.Dataset): Data handler of task ``task_id``.
        mnet: Main network.
        train_iter (int): Current training iteration.
        mnet_kwargs (dict): Additional keyword arguments that should be passed
            to the main network.

    Returns:
        (tuple): Tuple containing:

        - **val_loss** (torch.Tensor): Scalar tensor representing the validation
          task loss.
        - **val_acc** (torch.Tensor, optional): Scalar tensor representing the
          validation accuracy. ``None`` if ``accuracy_func`` not provided.
    """
    if data.num_val_samples == 0:
        logger.warn('No validation set for task %d. Training ' % task_id +
                    'evaluation skipped.')
        return None, None

    mnet.eval()
    if hnet is not None:
        hnet.eval()
    if dnet is not None:
        dnet.eval()

    if hasattr(config, 'all_task_softmax') and config.all_task_softmax:
        # For replay methods, we assume a growing softmax.
        assert num_trained is not None

        if config.multitask:
            # No growing softmax in this case.
            assert num_trained == config.num_tasks

    with torch.no_grad():
        sample_ids = data.get_val_ids()
        X = data.input_to_torch_tensor(data.get_val_inputs(), device,
                                       mode='inference', sample_ids=sample_ids)
        X = stu.preprocess_inputs(config, shared, X, task_id)

        T = data.output_to_torch_tensor(data.get_val_outputs(), device,
                                        mode='inference', sample_ids=sample_ids)
        T = stu.adjust_targets_to_head(config, data, T, task_id,
            trained_task_id=num_trained-1, is_one_hot=True)

        if hnet is not None and dnet is None:
            ext_tnet_weights = stu.hnet_forward(config, hnet, task_id)
        elif config.use_masks:
            ext_tnet_weights = ctx_masks[task_id]
        else:
            ext_tnet_weights = None

        allowed_outputs = stu.out_units_of_task(config, data, task_id,
                                                trained_task_id=num_trained-1)

        if isinstance(mnet, BiRNN):
            mnet_kwargs['seq_lengths'] = data.get_in_seq_lengths(sample_ids)

        Y_logits = mnet.forward(X, weights=ext_tnet_weights, **mnet_kwargs)
        Y_logits = Y_logits[:, :, allowed_outputs]

        val_loss = task_loss_func(Y_logits, T, data, None, None,
                                  sample_ids)
        val_loss /= data.num_val_samples
        writer.add_scalar('val/task_%d/task_loss' % task_id, val_loss,
                          train_iter)
        if config.multitask:
            val_msg = 'Task %i '%task_id + 'loss on validation set %f%s.'
        else:
            val_msg = 'Current task loss on validation set %f%s.'

        val_accuracy = None
        if accuracy_func is not None:
            val_accuracy, _ = accuracy_func(Y_logits, T, data, \
                data.get_val_ids())
            writer.add_scalar('val/task_%d/acc' % task_id, val_accuracy,
                              train_iter)
            val_msg = val_msg % (val_loss,
                                 ' (validation accuracy %.2f%%)' % val_accuracy)
        else:
            val_msg = val_msg % (val_loss, '')
        logger.info(val_msg)

    return val_loss, val_accuracy

def train_one_task(dhandlers, target_net, hnet, dnet, device, config, shared,
    logger, writer, curr_task_id, ctx_masks=None, smoothing_factor=0.5,
    task_loss_func=None, accuracy_func=None, ewc_loss_func=None,
    replay_fcts=None):
    """Train continual learning experiment.

    In the default case, this trains the networks in the current task with ID
    ``curr_task_id``. In the multitask scenario, data from all tasks is used
    simultaneously to train the networks.

    Args:
        (....): See docstring of :func:`train_tasks`.
        curr_task_id: Task id. Unused if we are in a multitask setting.
        smoothing_factor (float, optional): The smoothing factor for computing
            the exponentially smoothed training accuracy between evaluation
            steps (see :func:`evaluate`), to approximate the generalization gap.

    Returns:
        (dict): The task, ewc, and hnet components of the training loss. It has
        three keys, and the corresponding values are lists.
    """
    use_replay = hasattr(config, 'use_replay') and config.use_replay
    if use_replay:
        assert config.use_replay
        if config.multitask:
            raise ValueError('Replay not compatible with multitask training.')
        if dnet is not None:
            dnet.train()
        # Things that are done wrt. replay in this function:
        # * The decoder is jointly trained with the target network (classifier)
        #   to reconstruct the inputs based on a meaningful latent space
        #   (enforced via prior-matching).
        # * During that time, a checkpointed decoder is used from the second
        #   task onwards to create input samples for old tasks (by sampling
        #   from the latent space). Those samples are classified by a
        #   checkpointed target network, such that we retrieve "soft targets".
        # * These replayed inputs and soft targets are used to train the
        #   classifier via a distillation loss (which happens jointly with the
        #   training on the current task its data).
        # * Note, if the decoder is hypernet protected, only data from the
        #   current task is used to train the decoder. Otherwise, all data
        #   (current and replayed) is used.
        # * Note, if there is no hypernet (the hnet is task conditioned due to
        #   its embedding) and a multihead setup is used, then we also need to
        #   provide the task identity as input to the decoder. Otherwise, we
        #   wouldn't know what output head of the classifier to look at when
        #   selecting soft targets.
        assert replay_fcts is not None
        rec_loss_fct = replay_fcts['rec_loss']
        distill_loss_fct = replay_fcts['distill_loss']
        soft_trgt_acc_fct = replay_fcts['soft_trgt_acc']

        ### Checkpoint previous classifier/decoder/hypernet ###
        # We need the checkpointed replay model to produce input samples from
        # previous tasks. We also need the checkpointed classifier to compute
        # soft targets for the replayed inputs.

        # FIXME We trivially assume that all the computation depends on the
        # weights (and there is no other information in the network models that
        # is changed during training that might influence the forward
        # computation (e.g., batchnorm stats)). This is currently the case,
        # but might change in the future. Alternatively, we could make a deep
        # copy of the networks themselves.
        ckpt_tnet_weights = [w.detach().clone() \
                             for w in target_net.internal_params]
        ckpt_dnet_weights = None
        ckpt_hnet_theta = None
        ckpt_hnet_tembs = None
        if dnet is None:
            # Doesn't make sense to have a hypernet of the `target_net` whne
            # using replay.
            assert hnet is None
        else:
            if hnet is None:
                ckpt_dnet_weights = [w.detach().clone() \
                                     for w in dnet.internal_params]
            else:
                assert dnet.internal_params is None or \
                    len(dnet.internal_params) == 0
                if hasattr(config, 'use_new_hnet') and config.use_new_hnet:
                    ckpt_hnet_theta = [w.detach().clone() \
                                       for w in hnet.unconditional_params]
                    ckpt_hnet_tembs = [e.detach().clone() \
                                       for e in hnet.conditional_params]
                else:
                    ckpt_hnet_theta = [w.detach().clone() for w in hnet.theta]
                    ckpt_hnet_tembs = [e.detach().clone() \
                                       for e in hnet.get_task_embs()]

    if hasattr(config, 'all_task_softmax') and config.all_task_softmax:
        # Note, function `out_units_of_task` uses a growing softmax, that is
        # not compatible with multitask learning.
        assert not config.multitask

    if config.use_masks:
        assert ctx_masks is not None
    else:
        assert ctx_masks is None

    #num_total_tasks = len(dhandlers)
    all_dhandlers = dhandlers
    if not config.multitask:
        logger.info('\nStart training on task id: %i' % curr_task_id)
        # Modify the list of data handlers such that it only contains task t.
        dhandlers = [dhandlers[curr_task_id]]
    else:
        logger.info('\nStart multitask training with %d tasks' % len(dhandlers))
    num_tasks = len(dhandlers)

    # Note, during training, we do not tell the target network to load any
    # context-mod checkpoint, as we want to train the internal context-mod
    # weights and checkpoint them after training.
    tnet_kwargs = {}

    # Collect weights that should be regularized by EWC or SI. Note, if
    # checkpointing is used, the context-mod weights will be part of the main
    # network rather than provided externally.
    if config.use_ewc or config.use_si:
        assert not config.use_ewc or not config.use_si
        regged_tnet_weights = target_net.get_non_cm_weights()

    # Define loss function and optimizer
    # We use Adam as it is used in the related work.
    target_net.train()
    params = [] # Parameters to be optimized

    # Extract output head weights from target network.
    po_inds = [ii for ii, m in enumerate(target_net.get_output_weight_mask())
               if m is not None]
    tnet_head_params = []
    for poi in po_inds:
        if target_net.param_shapes_meta[poi]['index'] != -1:
            tnet_head_params.append(target_net.internal_params[ \
                target_net.param_shapes_meta[poi]['index']])
        else:
            assert not config.train_only_heads_after_first

    # Extract non-output head weights from target network.
    nonpo_inds = [ii for ii, m in enumerate(target_net.get_output_weight_mask())
                  if m is None]
    tnet_nonhead_params = []
    for poi in nonpo_inds:
        if target_net.param_shapes_meta[poi]['index'] != -1:
            tnet_nonhead_params.append(target_net.internal_params[ \
                target_net.param_shapes_meta[poi]['index']])

    if dnet is not None: # Generative Replay
        # Classifier is always trained when using replay.
        if config.train_only_heads_after_first and curr_task_id > 0:
            params = list(tnet_head_params)
            logger.debug('Only training output head weights of encoder.')
        elif config.dont_train_heads:
            params = list(tnet_nonhead_params)
            logger.debug('Not training output head weights of encoder.')
        else:
            params = list(target_net.parameters())
        if config.hnet_all:
            # Note, hypernet parameters are appended to `params` below.
            assert hnet is not None and (dnet.internal_params is None or \
                                         len(dnet.internal_params) == 0)
        else:
            params += list(dnet.parameters())
    elif not config.hnet_all:
        if curr_task_id == 0:
            if config.dont_train_heads:
                params = list(tnet_nonhead_params)
                logger.debug('Not training output head weights of main net.')
            else:
                params = list(target_net.parameters())
        elif config.train_tnet_once or config.train_only_heads_after_first:
            if config.train_only_heads_after_first:
                params += list(tnet_head_params)
                logger.debug('Internal target network weights except output ' +
                    'heads are not learned when training on task ' +
                    '%d.' % curr_task_id)
            else:
                logger.debug('Internal target network weights are not ' +
                    'learned when training on task %d.' % curr_task_id)
            if config.use_context_mod:
                # Context-mod weights are not considered internal target net
                # weights.
                params = []
                if config.checkpoint_context_mod:
                    params += target_net.get_cm_weights()
                else:
                    assert hnet is not None
        else:
            if config.dont_train_heads:
                params = list(tnet_nonhead_params)
                logger.debug('Not training output head weights of main net.')
            else: 
                params = list(target_net.parameters())
    else:
        assert config.hnet_all and len(list(target_net.parameters())) == 0

    if hnet is not None:
        hnet.train()
        # Note, this means that we include all task embeddings as trainable
        # parameters (hence, the regularizer may change old task embeddings
        # as well).
        params += hnet.parameters()

    # Add word embeddings to training parameters, if any.
    if hasattr(shared, 'word_emb_lookups') and not config.dont_learn_wembs:
        if config.multitask:
            logger.debug('Adding word embeddings of all tasks to optimizer.')
            for wemb in shared.word_emb_lookups:
                params.extend(wemb.parameters())
        else:
            logger.debug('Adding word embeddings of task %d to optimizer.' \
                         % curr_task_id)
            params.extend( \
                list(shared.word_emb_lookups[curr_task_id].parameters()))

    optimizer = get_optimizer(params, lr=config.lr,
        adam_beta1=config.adam_beta1, use_adam=True,
        weight_decay=config.weight_decay)

    # Note, the optimizer contains all trainable weights (including context-mod
    # or hnet weights) and the order might be arbitrary. However, for SI we only
    # need the predicted parameter change for those parameters included
    # in `regged_tnet_weights`. Therefore, we need to know how to extract the
    # parameter changes prescribed by the optimizer for these weights.
    if config.use_si:
        assert len(optimizer.param_groups) == 1
        regged_to_optim_params = {}
        for i1, p1 in enumerate(regged_tnet_weights):
            for i2, p2 in \
                    enumerate(optimizer.param_groups[0]['params']):
                if p1 is p2:
                    regged_to_optim_params[i1] = i2

    # In order to have the same total number of training samples when doing 
    # multitask learning, we multiply the number of iterations by the number 
    # of tasks considered (as we don't loop over tasks outside this function),
    # but the number of samples per iteration stays the same.
    num_train_iter = config.n_iter * len(dhandlers)

    for dhandler in dhandlers:
        dhandler.reset_batch_generator()

    ### Prepare hnet regularizer ###
    calc_reg = config.beta > 0 and curr_task_id > 0  and \
        hnet is not None and not config.train_from_scratch and \
        not config.multitask

    if calc_reg:
        # Usually, our regularizer acts on all weights of the main network.
        # Though, some weights might be connected to unused output neurons,
        # which is why we can ignore them.
        if config.multi_head:
            inds_of_prev_out_heads = [stu.out_units_of_task(config,
                all_dhandlers[tt], tt) for tt in range(curr_task_id)]
        else:
            inds_of_prev_out_heads = None

        if config.calc_hnet_reg_targets_online:
            # Compute targets for the regularizer whenever they are needed.
            # -> Computationally expensive.
            hnet_targets = None
            if dnet is not None:
                assert ckpt_hnet_theta is not None and \
                    ckpt_hnet_tembs is not None
                prev_hnet_theta = ckpt_hnet_theta
                prev_hnet_tembs = ckpt_hnet_tembs
            else:
                if hasattr(config, 'use_new_hnet') and config.use_new_hnet:
                    prev_hnet_theta = [w.detach().clone() \
                                       for w in hnet.unconditional_params]
                    prev_hnet_tembs = [e.detach().clone() \
                                       for e in hnet.conditional_params]
                else:
                    prev_hnet_theta = [w.detach().clone() for w in hnet.theta]
                    prev_hnet_tembs = [e.detach().clone() \
                                       for e in hnet.get_task_embs()]
        else:
            # Compute all targets before training.
            # -> Memory expensive; as many targets as previous tasks.
            hnet_targets = hreg.get_current_targets(curr_task_id, hnet)
            prev_hnet_theta = None
            prev_hnet_tembs = None

    ### Miscellaneous training preparations ###
    # Whether some kind of regularization is added to the task-specific loss.
    has_cl_reg = calc_reg  or curr_task_id > 0 and \
        (config.use_ewc or config.use_si) or use_replay
    # Do add any regularizer to the task loss?
    has_any_reg = has_cl_reg or \
        (config.use_context_mod and config.sparsify_context_mod) or \
        config.orthogonal_hh_reg > 0

    # Create training loss containers:
    # - 'rec': VAE reconstruction loss.
    # - 'pm': VAE prior-matching loss.
    # - 'distill': Distillation loss when training with softtargets during
    #   replay.
    loss_dict = {'task':[], 'ewc':[], 'si':[], 'hnet':[], 'sparse':[],
                 'orthogonal':[], 'rec': [], 'pm': [], 'distill': []}

    ######################
    ### Start Training ###
    ######################
    saved_models = False
    best_val_loss = None
    best_val_acc = None
    es_val_history = []
    es_val_iters = []
    es_val_weights = []
    smoothed_accuracy = None
    total_samples = 0
    for curr_iter in range(num_train_iter):

        # set optimizer to zero
        optimizer.zero_grad()

        # Checkpoint current weights for SI.
        if (config.num_tasks == 1 or curr_task_id < config.num_tasks - 1) and \
                config.use_si:
            si.si_pre_optim_step(target_net, regged_tnet_weights,
                                 no_pre_step_ckpt=config.si_task_loss_only)

        # Choose number of samples for each task:
        batch_tasks, batch_sizes_tmp = np.unique(np.random.randint(0,
            high=num_tasks, size=config.batch_size), return_counts=True)
        assert config.multitask or num_tasks == 1
        batch_sizes_tmp = batch_sizes_tmp.tolist()
        batch_sizes = []
        for ii in range(num_tasks):
            if ii in batch_tasks:
                batch_sizes.append(batch_sizes_tmp.pop(0))
            else:
                batch_sizes.append(0)
        #assert np.sum(batch_sizes) == config.batch_size

        # Iterate across tasks, taking batches from each task and accumulating 
        # the loss into a single value.
        loss_task = 0
        if calc_reg:
            loss_hnet = 0
        if curr_task_id > 0 and config.use_ewc:
            loss_ewc = 0
        if curr_task_id > 0 and config.use_si:
            loss_si = 0
        if config.use_context_mod and config.sparsify_context_mod:
            loss_sparse = 0
        if config.orthogonal_hh_reg > 0:
            loss_orth = 0
        if config.classification:
            accuracy = 0
        if use_replay:
            loss_distill = 0
            if soft_trgt_acc_fct is not None:
                soft_trgt_acc = 0
        if dnet is not None:
            loss_rec = 0
            loss_pm = 0
        val_loss = 0
        val_acc = 0
        val_num_samples = 0

        num_samples = 0

        # Here we redefine the list of data handlers for safety. This is needed
        # for permuted datasets, where the dhandlers might not be an actual
        # list of data-handlers but a pseudo-list that only holds a single 
        # time the original dataset, and then performs permutations on the fly
        # as specific datahandlers are requested. Note that the step below is
        # only really useful if `use_replay` is activated, as `all_dhandlers`
        # object is called within the replay functions, and might cause the
        # current `dhandler` to change.
        if not config.multitask:
            dhandlers = [all_dhandlers[curr_task_id]]
        else:
            dhandlers = all_dhandlers

        # This loop is only relevant for the multitask scenario.
        for batch_size, dhandler in zip(batch_sizes, dhandlers):

            if batch_size == 0:
                # FIXME In multitask learning, this means that also the
                # validation isn't run.
                continue

            if not config.last_task_only:
                task_id = dhandler._data['task_id']
            else:
                task_id = 0
            if not config.multitask and not config.last_task_only:
                assert(task_id == curr_task_id)

            # Output units of the current task.
            allowed_outputs = stu.out_units_of_task(config, dhandler, task_id,
                dhandlers=None, trained_task_id=task_id)

            ##################################
            ### Evaluate training progress ###
            ##################################
            # We test the network before we run the training iteration.
            # That way, we can see the initial performance of the untrained
            # network.
            curr_val_loss = None
            curr_val_acc = None
            # Note, we also compute the validation accuracy before the last
            # training iteration to know the performance at the end of
            # training.
            if (curr_iter % config.val_iter == 0 or \
                    curr_iter == num_train_iter-1) and \
                    dhandler.num_val_samples > 0:
                curr_val_loss, curr_val_acc = evaluate(config, shared, logger,
                    writer, device, task_id, dhandler, target_net, hnet, dnet,
                    curr_iter, tnet_kwargs, ctx_masks=ctx_masks,
                    accuracy_func=accuracy_func, task_loss_func=task_loss_func,
                    num_trained=config.num_tasks if config.multitask \
                        else task_id+1)
                val_loss += curr_val_loss * dhandler.num_val_samples
                if curr_val_acc is not None:
                    val_acc += curr_val_acc * dhandler.num_val_samples
                val_num_samples += dhandler.num_val_samples

                target_net.train()
                if hnet is not None:
                    hnet.train()
                if dnet is not None:
                    dnet.train()
            else:
                val_loss = None
                val_acc = None

            #######################################
            ### Task Loss on Current Mini-batch ###
            #######################################
            batch = dhandler.next_train_batch(batch_size, return_ids=True)
            X = dhandler.input_to_torch_tensor(batch[0], device, mode='train',
                                               sample_ids=batch[2])
            X_before_preprocessing = X
            X = stu.preprocess_inputs(config, shared, X, task_id)

            T = dhandler.output_to_torch_tensor(batch[1], device, mode='train',
                                                sample_ids=batch[2])
            T = stu.adjust_targets_to_head(config, dhandler, T, task_id,
                trained_task_id=task_id, is_one_hot=True)

            num_samples += X.shape[1]
            total_samples += X.shape[0]

            # compute loss of current data
            if hnet is not None and dnet is None:
                ext_tnet_weights = stu.hnet_forward(config, hnet, task_id)
            else:
                ext_tnet_weights = None
            if config.use_masks:
                ext_tnet_weights = ctx_masks[task_id]

            if isinstance(target_net, BiRNN):
                tnet_kwargs['seq_lengths'] = \
                    dhandler.get_out_seq_lengths(batch[2])

            Y_logits = target_net.forward(X, weights=ext_tnet_weights,
                                          **tnet_kwargs)
            Y_logits_full = Y_logits
            Y_logits = Y_logits[:, :, allowed_outputs]

            if curr_iter % 500 == 0 and config.show_plots:
                plc.plot_outputs([Y_logits], [T], config, mode='train', \
                    curr_iter=curr_iter, task_id=task_id)

            # Note, some arguments are currently unused in the loss function
            # interface.
            loss_task += task_loss_func(Y_logits, T, dhandler, None, None,
                                        batch[2])
            # Note, since we only estimate the NLL on a minibatch, we have to
            # rescale it's value by a factor N/B, where N is the dataset size
            # and B is the batch size.
            if not config.multitask:
                # Note, in the multitask case we do rescaling below. But for the
                # CL case we do it here as `loss_task` is reused within the
                # for loop.
                loss_task /= batch_size
            # FIXME Scaling requires high reg strengths, so we omit it.
            #loss_task *= dhandler.num_train_samples / batch_size
            # NOTE The distillation losses are currently meaned across the
            # batch.

            # If we compute the gradients wrt to the task-specific loss already
            # now, then we don't need to compute them later anymore.
            calc_grad_loss_task = True
            if (config.num_tasks == 1 or curr_task_id < config.num_tasks-1) \
                    and config.use_si and config.si_task_loss_only:
                calc_grad_loss_task = False

                # Compute gradients of task-specific loss.
                loss_task.backward(retain_graph=has_any_reg)

                # Use the task-specific gradients to determine which update the
                # optimizer would perform, without actually performing it.
                param_step = opstep.calc_delta_theta(optimizer, False,
                    lr=config.lr, detach_dt=True)

                delta_params = [None] * len(regged_tnet_weights)
                for kk, vv in regged_to_optim_params.items():
                    delta_params[kk] = param_step[vv]

                assert len(delta_params) == len(regged_tnet_weights)

                si.si_post_optim_step(target_net, regged_tnet_weights,
                                      delta_params=delta_params)

            # Compute accuracies and add to overall training step with the
            # appropriate weight depending on the number of samples used.
            if config.classification:
                classifier_accuracy, _ = accuracy_func(Y_logits, T, dhandler,
                                                    batch[2])
                accuracy += classifier_accuracy * batch_size

            ############################
            ### Hypernet Regularizer ###
            ############################
            # compute hypernet loss and compute hnet reg
            if calc_reg:
                # Note, we do not calculate "dTheta" on purpose, as it usually
                # doesn't give much and would require extra computation and the
                # declaration of multiple optimizers.
                loss_hnet += config.beta * hreg.calc_fix_target_reg(hnet,
                     task_id, targets=hnet_targets,
                     mnet=target_net if dnet is None else None,
                     dTheta=None, dTembs=None, prev_theta=prev_hnet_theta,
                     prev_task_embs=prev_hnet_tembs,
                     inds_of_out_heads=inds_of_prev_out_heads \
                         if dnet is None else None,
                     batch_size=config.hnet_reg_batch_size)

            #######################
            ### EWC Regularizer ###
            #######################
            # compute (online) ewc reg.
            if task_id > 0 and config.use_ewc:
                loss_ewc += config.ewc_lambda * ewc.ewc_regularizer(task_id,
                    regged_tnet_weights, target_net,
                    online=True, gamma=config.ewc_gamma)

            ######################
            ### SI Regularizer ###
            ######################
            # Synaptic Intelligence reg.
            if task_id > 0 and config.use_si:
                loss_si += config.si_lambda * si.si_regularizer(target_net,
                    regged_tnet_weights)

            #############################
            ### Sparsity  Regularizer ###
            #############################
            # Add sparsification regularization for gains
            loss_sparse_curr = 0

            if config.use_context_mod and config.sparsify_context_mod:
                gains = []
                if ext_tnet_weights is not None:
                    for ii, p in enumerate(ext_tnet_weights):
                        i_ps = target_net.hyper_shapes_learned_ref[ii]
                        meta = target_net.param_shapes_meta[i_ps]
                        if meta['name'] == 'cm_scale':
                            gains.append(p)
                else:
                    assert target_net.context_mod_layers is not None
                    for cm_layer in target_net.context_mod_layers:
                        gains.append(cm_layer.gain)

                n_gains = len(gains)
                for ii, gain in enumerate(gains):
                    gain = target_net.context_mod_layers[ii]. \
                        preprocess_gain(gain)

                    if config.sparsification_reg_type == 'l1':
                        loss_sparse_curr += gain.abs().mean()
                    else:
                        assert config.sparsification_reg_type == 'log'
                        # `eps` simply ensures that gains that are already
                        # pretty close to zero are not pushed any further to
                        # zero. Otherwise this reg would explode.
                        eps = 1e-5
                        # Note, the `abs()` is not necessary if gains are
                        # positive by default (e.g., in case of softplus-gains).
                        loss_sparse_curr += torch.log(gain.abs() + eps).mean()

                loss_sparse_curr *= config.sparsification_reg_strength / n_gains
                loss_sparse += loss_sparse_curr

            ##############################
            ### Orthogonal Regularizer ###
            ##############################
            # Add orthogonal regularization to hidden-to-hidden weights.
            if config.orthogonal_hh_reg > 0:
                loss_orth += stu.orthogonal_regularizer(config, target_net,
                    hnet_out=ext_tnet_weights)
                if dnet is not None:
                    assert ext_tnet_weights is None
                    ext_dnet_weights = None
                    if hnet is not None:
                        ext_dnet_weights = stu.hnet_forward(config, hnet,
                                                            task_id)
                    loss_orth += stu.orthogonal_regularizer(config, dnet,
                        hnet_out=ext_dnet_weights)

            #######################################
            ### Replay Data from Previous Tasks ###
            #######################################
            if use_replay and task_id > 0:
                # Replay a batch of samples containing data from all previous
                # tasks.
                # TODO Should we make the batch size user configurable?
                # At the moment, all old tasks together have as many samples as
                # the current task!
                rep_batch_size = batch_size
                X_rep, X_rep_ids, X_rep_tids, X_rep_lens = rtu.replay_samples( \
                    config, shared, device, all_dhandlers, list(range(task_id)),
                    rep_batch_size, dnet, hnet=hnet,
                    dnet_weights=ckpt_dnet_weights,
                    hnet_weights=ckpt_hnet_theta, hnet_tembs=ckpt_hnet_tembs,
                    split_by_id=True,
                    replay_all_data=config.replay_true_data,
                    coresets=shared.coresets if config.coreset_size != -1 \
                                             else None,
                    ret_seq_lens=True)

                # Compute soft-targets for the replayed samples, such that we
                # have generated "training data for old tasks".
                T_rep_logits = []
                for i_rep in range(len(X_rep)):
                    # Note, we use the checkpointed classifier, that was only
                    # trained up to `task_id-1`.
                    # Hence, if a growing softmax is used, we have to append
                    # zeros at some point before using the targets. However,
                    # we can't do this right away as we have logits and not
                    # softmax targets.
                    T_rep_logits.append(rtu.get_soft_targets(config,
                        all_dhandlers, X_rep[i_rep], X_rep_ids[i_rep],
                        target_net, ckpt_tnet_weights,
                        trained_task_id=task_id-1,
                        input_lens=None if X_rep_lens is None \
                                        else X_rep_lens[i_rep]))

            ###############################################
            ### "Distill" Replayed Data into Classifier ###
            ###############################################
            if use_replay and task_id > 0:
                # We currently don't allow a hypernet for the classifier when
                # using replay!
                assert ext_tnet_weights is None

                Y_rep_logits_full = []
                for i_rep in range(len(X_rep)):
                    if isinstance(target_net, BiRNN):
                        # TODO Think how to pass sensible sequence lengths to
                        # bidirectional RNN.
                        if X_rep_lens is None:
                            raise NotImplementedError()
                        Y_rep_logits_i = target_net.forward(X_rep[i_rep],
                            weights=ext_tnet_weights,
                            seq_lengths=X_rep_lens[i_rep])
                    else:
                        Y_rep_logits_i = target_net.forward(X_rep[i_rep],
                            weights=ext_tnet_weights)
                    Y_rep_logits_full.append(Y_rep_logits_i)
                    allowed_outputs_rep_i = stu.out_units_of_task(config,
                        all_dhandlers[X_rep_tids[i_rep]], X_rep_tids[i_rep],
                        trained_task_id=task_id)
                    Y_rep_logits_i = Y_rep_logits_i[:, :, allowed_outputs_rep_i]

                    # The input is provided to the distillation loss, as it
                    # might contain information that the loss function might
                    # need, e.g., the unpadded sequence length by looking at
                    # sequence stop bits.

                    # Usually, we assume that the distillation loss function
                    # can infer the sequence length form the replayed data
                    # (as the ground truth is typically not known). However,
                    # if true data is replayed, the sequence length is known
                    # and could be passed.
                    extra_rep_args = {}
                    if isinstance(dhandler, MUDData):
                        extra_rep_args['in_seq_lens'] = X_rep_lens[i_rep]
                    loss_distill += distill_loss_fct(config, X_rep[i_rep],
                        Y_rep_logits_i, T_rep_logits[i_rep],
                        all_dhandlers[X_rep_tids[i_rep]], **extra_rep_args)

                    if soft_trgt_acc_fct is not None:
                        soft_trgt_acc += soft_trgt_acc_fct(config,
                                X_rep[i_rep], Y_rep_logits_i,
                                T_rep_logits[i_rep],
                                all_dhandlers[X_rep_tids[i_rep]],
                                **extra_rep_args) * \
                            T_rep_logits[i_rep].shape[1]
                if soft_trgt_acc_fct is not None:
                    soft_trgt_acc /= rep_batch_size

                loss_distill *= config.replay_distill_reg

            ########################
            ### Train Replay VAE ###
            ########################
            # Compute reconstruction and prior-matching loss.
            if dnet is not None:
                enc_inputs = []
                enc_logits = []
                enc_task_ids = []
                # Note, if a hypernetwork is used, then the decoder is
                # always only trained on current task data. Otherwise, it is
                # also trained on its replayed data.
                if task_id > 0 and hnet is None:
                    enc_inputs.extend(X_rep)
                    enc_logits.extend(Y_rep_logits_full)
                    enc_task_ids.extend(X_rep_tids)
                enc_inputs.append(X)
                enc_logits.append(Y_logits_full)
                enc_task_ids.append(task_id)

                loss_rec, loss_pm, dnet_logits = rtu.compute_vae_loss(config,
                    enc_inputs, enc_logits, enc_task_ids, rec_loss_fct, dnet,
                    hnet=hnet)
                loss_rec *= config.replay_rec_strength
                loss_pm *= config.replay_pm_strength

        assert(num_samples == config.batch_size)
        if config.multitask:
            # Check comment above for why rescaling is necessary.
            loss_task /= config.batch_size

        #############################
        ### Minimize overall loss ###
        #############################

        if config.classification:
            # Normalize accumulated accuracy by number of samples used.
            accuracy /= config.batch_size
        if val_loss is not None:
            val_loss /= val_num_samples
        if val_acc is not None:
            val_acc /= val_num_samples

        # Compute backward pass and update parameters based on the accumulated
        # loss from all tasks.
        # Note, we are cloning the tensor `loss_task` here as we are later using
        # it to log the task loss to tensorboard. Otherwise, it would be
        # modified everytime `loss` is modified.
        loss = loss_task.clone()
        loss_cl = 0

        if calc_reg:
            loss_cl += loss_hnet
        if curr_task_id > 0 and config.use_ewc:
            loss_cl += loss_ewc
        if curr_task_id > 0 and config.use_si:
            loss_cl += loss_si
        if use_replay:
            loss_cl += loss_distill
            if dnet is not None:
                # FIXME VAE losses are not really CL losses.
                loss_cl += loss_pm + loss_rec
        assert has_cl_reg or loss_cl == 0

        # Note: loss == loss_task + loss_reg
        loss_reg = loss_cl
        if config.use_context_mod and config.sparsify_context_mod:
            loss_reg += loss_sparse
        if config.orthogonal_hh_reg > 0:
            loss_reg += loss_orth

        loss += loss_reg

        if np.isnan(loss.item()):
            task_id_msg = ''
            if not config.multitask:
                task_id_msg = 'task %d, '% curr_task_id
            raise Exception('NaN observed in training loss (' +
                            task_id_msg + 'step %d).'% curr_iter)

        if calc_grad_loss_task:
            loss.backward()
        elif has_any_reg:
            # Gradient of current task loss already calculated.
            loss_reg.backward()
        else:
            assert loss_reg == 0

        if config.clip_grad_norm != -1:
            torch.nn.utils.clip_grad_norm_(\
                optimizer.param_groups[0]['params'], config.clip_grad_norm)

        optimizer.step()

        if config.use_si and curr_task_id < config.num_tasks-1 and \
                not config.si_task_loss_only:
            si.si_post_optim_step(target_net, regged_tnet_weights)

        #############################
        ### Log training progress ###
        #############################
        if curr_iter % 50 == 0:

            if config.multitask:
                subfolder_name = 'all_tasks'
            else:
                subfolder_name = 'task_%d' % curr_task_id

            # add info to writer
            if config.classification:
                writer.add_scalar('train/%s/class_accuracy' % subfolder_name,
                                  accuracy, curr_iter)
            writer.add_scalar('train/%s/loss_task' % subfolder_name,
                              loss_task, curr_iter)
            if calc_reg:
                writer.add_scalar('train/%s/hnet_loss_task' % subfolder_name,
                                  loss_hnet, curr_iter)
            if curr_task_id > 0 and config.use_ewc:
                writer.add_scalar('train/%s/loss_ewc_task' % subfolder_name,
                                  loss_ewc, curr_iter)
            if curr_task_id > 0 and config.use_si:
                writer.add_scalar('train/%s/loss_si' % subfolder_name,
                                  loss_si, curr_iter)
            if config.sparsify_context_mod:
                writer.add_scalar('train/%s/loss_sparse' % subfolder_name,
                                  loss_sparse, curr_iter)
            if config.orthogonal_hh_reg > 0:
                writer.add_scalar('train/%s/loss_orthogonal' % subfolder_name,
                                  loss_orth, curr_iter)
            if use_replay and curr_task_id > 0:
                writer.add_scalar('train/%s/loss_distill' % subfolder_name,
                                      loss_distill, curr_iter)
                if soft_trgt_acc_fct is not None:
                    # Note, the accuracy is based on fake (replayed) targets and
                    # not ground-truth!
                    writer.add_scalar('train/%s/acc_replayed' % subfolder_name,
                                      soft_trgt_acc, curr_iter)
            if dnet is not None:
                writer.add_scalar('train/%s/loss_vae_pm' % subfolder_name,
                                  loss_pm, curr_iter)
                writer.add_scalar('train/%s/loss_vae_rec' % subfolder_name,
                                  loss_rec, curr_iter)
            # print info to command line
            msg = 'Training step {}: '
            if config.classification:
                msg += 'Accuracy on current batch: {:.3f}, ' 
            msg += 'Task loss: {:.3f}.'

            # optional message vor ewc
            if curr_task_id > 0 and config.use_ewc:
                msg_ewc = ', EWC loss: {:.3f}'
                msg_ewc = msg_ewc.format(loss_ewc)
            else:
                msg_ewc = ''
            # optional message vor si
            if curr_task_id > 0 and config.use_si:
                msg_si = ', SI loss: {:.3f}'
                msg_si = msg_si.format(loss_si)
            else:
                msg_si = ''
            # optional message vor hnet
            if calc_reg:
                msg_hnet = ', hnet loss: {:.3f}'
                msg_hnet = msg_hnet.format(loss_hnet)
            else:
                msg_hnet = ''
            if config.sparsify_context_mod:
                msg_sparse = ', sparse loss: {:.3f}'
                msg_sparse = msg_sparse.format(loss_sparse)
            else:
                msg_sparse = ''
            if config.orthogonal_hh_reg > 0:
                msg_orth = ', orthogonal loss: {:.3f}'
                msg_orth = msg_orth.format(loss_orth)
            else:
                msg_orth = ''
            if use_replay:
                msg_repl = ''
                if dnet is not None:
                    msg_repl = ', reconstruction loss: {:.3f}, ' +\
                        'prior-matching loss {:.3f}' 
                    msg_repl = msg_repl.format(loss_rec, loss_pm)
                if curr_task_id > 0:
                    msg_repl += ', distillation loss: {:.3f}'
                    msg_repl = msg_repl.format(loss_distill)
            else:
                msg_repl = ''

            if config.classification:
                logger.info(msg.format(curr_iter, accuracy, loss_task) +
                    msg_ewc + msg_si + msg_hnet + msg_sparse + msg_orth +
                    msg_repl)
            else:
                logger.info(msg.format(curr_iter, loss_task) +
                    msg_ewc + msg_si + msg_hnet + msg_sparse + msg_orth +
                    msg_repl)

            # collect info in lists for later analysis 
            loss_dict['task'].append(loss_task.detach().cpu().numpy())
            if curr_task_id > 0 and config.use_ewc:
                loss_dict['ewc'].append(loss_ewc.detach().cpu().numpy())
            if curr_task_id > 0 and config.use_si:
                loss_dict['si'].append(loss_si.detach().cpu().numpy())
            if calc_reg:
                loss_dict['hnet'].append(loss_hnet.detach().cpu().numpy())
            if config.sparsify_context_mod:
                loss_dict['sparse'].append(loss_sparse.detach().cpu().numpy())
            if config.orthogonal_hh_reg > 0:
                loss_dict['orthogonal'].append(loss_orth.detach().cpu().numpy())
            if dnet is not None:
                loss_dict['rec'].append(loss_rec.detach().cpu().numpy())
                loss_dict['pm'].append(loss_pm.detach().cpu().numpy())
            if use_replay and  curr_task_id > 0:
                loss_dict['distill'].append( \
                    loss_distill.detach().cpu().numpy())

            if val_loss is not None and config.multitask:
                # We only track the validation loss per task inside function
                # `evaluate`.
                writer.add_scalar('val/%s/loss' % subfolder_name,
                                  val_loss, curr_iter)
            if config.classification:
                if smoothed_accuracy is None:
                    smoothed_accuracy = accuracy
                else:
                    smoothed_accuracy = smoothing_factor * accuracy + \
                        smoothed_accuracy * (1-smoothing_factor)

                if val_acc is not None:
                    if config.multitask:
                        # We only track the validation accuracy per task inside
                        # function `evaluate`.
                        writer.add_scalar('val/%s/acc' % subfolder_name,
                                          val_acc, curr_iter)
                    # Note, this is of course a very crude approximation of the
                    # generalization gap, depending heavily on the training
                    # batch size. Though, it gives us an easy way to detect
                    # overfitting.
                    generalization_gap = np.abs(smoothed_accuracy - val_acc)
                    writer.add_scalar('val/%s/train_acc' % subfolder_name,
                                      smoothed_accuracy, curr_iter)
                    writer.add_scalar('val/%s/generalization_gap' \
                        % subfolder_name, generalization_gap, curr_iter)

                    smoothed_accuracy = None

            ############################################
            ### Log norm of hidden-to-hidden weights ###
            ############################################
            # FIXME Also log for decoder.
            weights_hh = stu.extract_hh_weights(target_net,
                                                hnet_out=ext_tnet_weights)
            weights_hh = torch.cat([w.flatten() for w in weights_hh])
            weights_hh_norm = torch.norm(weights_hh, 2)
            writer.add_scalar('train/%s/tnet_weights_hh_norm' % subfolder_name,
                              weights_hh_norm, curr_iter)

            ###############################
            ### Log gradient magnitudes ###
            ###############################
            if not config.hnet_all:
                tnet_grad = torch.cat([g.grad.flatten() for g in \
                       target_net.get_non_cm_weights() if g.grad is not None])

                tnet_grad_norm = torch.norm(tnet_grad, 2)
                writer.add_scalar('train/%s/tnet_grad_norm' % subfolder_name,
                                  tnet_grad_norm, curr_iter)

            if config.use_context_mod and hnet is None:
                cm_grad = torch.cat([g.grad.flatten() for g in \
                                     target_net.get_cm_weights()])
                cm_grad_norm = torch.norm(cm_grad, 2)
                writer.add_scalar('train/%s/cm_grad_norm' % subfolder_name,
                                  cm_grad_norm, curr_iter)
            if hnet is not None:
                hnet_grad = []
                for g in hnet.parameters():
                    # Note, future task embeddings haven't been used and have no
                    # grads therefore.
                    if g.grad is None:
                        continue
                    hnet_grad.append(g.grad.flatten())
                hnet_grad = torch.cat(hnet_grad)
                hnet_grad_norm = torch.norm(hnet_grad, 2)
                writer.add_scalar('train/%s/hnet_grad_norm' % subfolder_name,
                                  hnet_grad_norm, curr_iter)

            if dnet is not None and hnet is None:
                dnet_grad = torch.cat([g.grad.flatten() \
                                       for g in dnet.parameters()])
                dnet_grad_norm = torch.norm(dnet_grad, 2)
                writer.add_scalar('train/%s/dnet_grad_norm' % subfolder_name,
                                  dnet_grad_norm, curr_iter)

            #############################
            ### Dataset-specific logs ###
            #############################
            if isinstance(dhandler, CopyTask) and not config.multitask:
                X_np = X.detach().cpu().numpy()
                if config.input_task_identity:
                    X_np = X_np[:,:,:-config.num_tasks]
                Y_np = (Y_logits >= 0).detach().cpu().numpy()
                T_np = T.detach().cpu().numpy()

                # FIXME We shouldn't use private methods.
                X_np = dhandler._flatten_array(X_np, ts_dim_first=True)
                Y_np = dhandler._flatten_array(Y_np, ts_dim_first=True)
                T_np = dhandler._flatten_array(T_np, ts_dim_first=True)

                dhandler.plot_samples('Example (Masked) Predictions',
                    X_np[:6, :], outputs=T_np[:6, :], predictions=Y_np[:6, :],
                    num_samples_per_row=3, show=False, equalize_size=True,
                     mask_predictions=True, sample_ids=batch[2])
                writer.add_figure('train/%s' % subfolder_name, plt.gcf(),
                                  curr_iter, close=True)

            # Plot VAE inputs and reconstructions.
            # Note, inputs of the current task are actual data samples, where
            # as the inputs from previous tasks are replayed using a
            # checkpointed decoder.
            if isinstance(dhandler, SMNISTData) and dnet is not None:
                for ii, enc_tid in enumerate(enc_task_ids):
                    tid_msg = '%d_%d' % (enc_tid, task_id)
                    tu_smnist.draw_samples('Autoencoding', enc_inputs[ii],
                        writer, 'replay/autoencoder_%s' % tid_msg, curr_iter,
                        samples2=dnet_logits[ii])
            if isinstance(dhandler, CopyTask) and dnet is not None:
                for ii, enc_tid in enumerate(enc_task_ids):
                    tid_msg = '%d_%d' % (enc_tid, task_id)
                    tu_copy.draw_samples('Autoencoding', enc_inputs[ii],
                        writer, 'replay/autoencoder_%s' % tid_msg, curr_iter,
                        samples2=torch.sigmoid(dnet_logits[ii]))

            # Note, if the `hnet` is not used, but the VAE is trained, then the
            # replayed samples are already plotted above as encoder inputs.
            # If the hnet is used (note, then we don't train the VAE on replayed
            # data) or there is no decoder, then we plot the replayed data used
            # for distillation here.
            if isinstance(dhandler, SMNISTData) and use_replay and \
                    (dnet is None or dnet is not None and hnet is not None) \
                    and curr_task_id > 0:
                for ii, rep_tid in enumerate(X_rep_tids):
                    tid_msg = '%d_%d' % (rep_tid, task_id)
                    tu_smnist.draw_samples('Replayed', X_rep[ii], writer,
                        'replay/replayed_%s' % tid_msg, curr_iter)
            if isinstance(dhandler, CopyTask) and use_replay and \
                    (dnet is None or dnet is not None and hnet is not None) \
                    and curr_task_id > 0:
                for ii, rep_tid in enumerate(X_rep_tids):
                    tid_msg = '%d_%d' % (rep_tid, task_id)
                    tu_copy.draw_samples('Replayed', X_rep[ii], writer,
                        'replay/replayed_%s' % tid_msg, curr_iter)

            if isinstance(dhandler, MUDData) and not config.multitask:
                X_str, T_str = dhandler.decode_batch(X_before_preprocessing,
                                                     T, sample_ids=batch[2])
                _, Y_str = dhandler.decode_batch(X_before_preprocessing,
                                                 F.softmax(Y_logits, dim=2),
                                                 sample_ids=batch[2])

                example_sentence = ' '.join( \
                    ['%s (%s, %s)' % (X_str[0][ii], T_str[0][ii], Y_str[0][ii])\
                     for ii in range(len(X_str[0]))])
                writer.add_text('train/%s/train_sentence' % subfolder_name,
                                example_sentence, curr_iter)

            #########################################################
            ### Explore loss contribution of main network weights ###
            #########################################################

            if hnet is not None and calc_reg:
                # Compute the contribution to the loss of feedforward and
                # recurrent weights separately.

                def get_target_net_weights_grad_norm(weight_type='rec'):
                    """Get the norm of the gradient related to a specific type
                    of main network weights, recurrent or feedforward.

                    Args:
                        weight_type (optional, str): The type of main network
                            weights: recurrent (``'rec'``) or feedforward
                            (``'ff'``).

                    Returns:
                        (tuple): Tuple containing:

                        - **hnet_grad** (float): The flattened gradient.
                        - **hnet_grad_norm** (float): The gradient norm.
                        - **loss_weights** (float): The associated loss.
                        - **num_weights** (int): Number of weights of the 
                          specified type.
                    """

                    # In order to compute the loss associated with the recurrent
                    # or feedforward weights only, we provide a mask to the 
                    # computation of the loss, that associates zero values for
                    # weights that are not of the specified weight type, and 
                    # values of one for the specified weight type. These masks
                    # are then multiplied by the difference between target
                    # and predicted main network weights when computing the hnet
                    # regularizer. For coding simplicity, they are provided
                    # as fisher estimate values to 
                    # :meth:`hreg.calc_fix_target_reg` even though they have
                    # nothing to do with EWC.
                    weights_idxs = task_id*[stu.get_target_net_weight_masks(\
                            target_net, weight_type=weight_type, device=device)]

                    # Number of weights of the given type (counting all 
                    # output heads).
                    num_weights = int(np.sum([ww.sum() for ww in \
                        weights_idxs[0]]))

                    # Compute the loss associated to the weight type.
                    optimizer.zero_grad()
                    loss_weights = hreg.calc_fix_target_reg(hnet,
                         task_id, targets=hnet_targets,
                         mnet=target_net if dnet is None else None,
                         dTheta=None, dTembs=None, prev_theta=prev_hnet_theta,
                         prev_task_embs=prev_hnet_tembs,
                         inds_of_out_heads=inds_of_prev_out_heads \
                             if dnet is None else None,
                         batch_size=config.hnet_reg_batch_size,
                         fisher_estimates=weights_idxs)
                    loss_weights.backward()

                    # Get the gradient norm.
                    hnet_grad = []
                    for g in hnet.parameters():
                        # Note, future task embeddings haven't been used and 
                        # have no grads therefore.
                        if g.grad is None:
                            continue
                        hnet_grad.append(g.grad.flatten())
                    hnet_grad = torch.cat(hnet_grad)
                    hnet_grad_norm = torch.norm(hnet_grad, 2)

                    return hnet_grad, hnet_grad_norm, loss_weights, num_weights

                ## Feedforward weights.
                hnet_grad_ff, hnet_grad_norm_ff, loss_weights_ff, \
                    num_ff_weights = \
                        get_target_net_weights_grad_norm(weight_type='ff')
                writer.add_scalar('train/%s/hnet_grad_norm_per_ff_w'%\
                        subfolder_name, hnet_grad_norm_ff/num_ff_weights,
                        curr_iter)
                writer.add_scalar('train/%s/hnet_reg_ff_loss' % \
                    subfolder_name, loss_weights_ff, curr_iter)
                writer.add_histogram('train/%s/hnet_grad_ff' % \
                    subfolder_name, hnet_grad_ff, curr_iter)

                ## Recurrent weights.
                hnet_grad_rec, hnet_grad_norm_rec, loss_weights_rec, \
                    num_rec_weights = \
                        get_target_net_weights_grad_norm(weight_type='rec')
                writer.add_scalar('train/%s/hnet_grad_norm_per_rec_w'%\
                        subfolder_name, hnet_grad_norm_rec/num_rec_weights,
                        curr_iter)
                writer.add_scalar('train/%s/hnet_reg_rec_loss' % \
                    subfolder_name, loss_weights_rec, curr_iter)
                writer.add_histogram('train/%s/hnet_grad_rec' % \
                    subfolder_name, hnet_grad_rec, curr_iter)

        #####################################################
        ### Checkpoint if validation performance improved ###
        #####################################################
        if config.use_best_models and \
                (val_loss is not None or val_acc is not None):

            wembs = shared.word_emb_lookups if \
                hasattr(shared, 'word_emb_lookups') and \
                not config.dont_learn_wembs else None

            ckpt_task_id = None if config.multitask else curr_task_id
            if val_acc is not None and best_val_acc is not None and \
                    val_acc > best_val_acc:
                stu.save_models(config.out_dir, logger, val_acc, target_net,
                    hnet=hnet, dnet=dnet, wembs=wembs, task_id=ckpt_task_id,
                    train_iter=curr_iter, max_ckpts_to_keep=2)
                saved_models = True
            elif val_loss is not None and best_val_loss is not None and \
                    val_loss < best_val_loss:
                stu.save_models(config.out_dir, logger, -val_loss, target_net,
                    hnet=hnet, dnet=dnet, wembs=wembs, task_id=ckpt_task_id,
                    train_iter=curr_iter, max_ckpts_to_keep=2)
                saved_models = True

            if val_loss is not None and \
                    (best_val_loss is None or val_loss < best_val_loss):
                best_val_loss = val_loss
            if val_acc is not None and \
                    (best_val_acc is None or val_acc > best_val_acc):
                best_val_acc = val_acc

        ######################################
        ### Check Early Stopping Criterion ###
        ######################################
        # The goal of early stopping is to avoid unnecessary computation,
        # especially for curriculum tasks where initial tasks are much easier
        # than later ones.
        # Typical early stopping acts as soon as the validation performance
        # decreases. However, we try a different implementation due to the
        # many regularizers that are not reflected in the validation measure.
        # The early stopping works by fitting a straight line through all
        # validation measures taken so far (weighting past ones less than recent
        # ones). If the slope of this straight line falls below a user specified
        # threshold, then the training can be stopped.
        # However, initially, the slope might be very small due to random
        # flactuations (jumping, unstable validation measures).
        # To prevent such behavior, a sufficiently long "warmup" phase has to be
        # chosen during which the stopping criteria is not enforced.
        # Note, even after the warumup phase, the slope can suddenly decrease
        # if (for instance) the validation measures develope in the reverse
        # direction for some time (instead of further improving, they suddenly
        # decrease). To capture this behavior, early stooping can only occur at
        # best values seen so far.

        # Why don't we just whether the validation accuracy is above, e.g.,
        # 99%? it is important that we check that the loss stabilized and
        # converged. Otherwise, we might stop before the regularizers had time
        # to do their job. Note, this stopping criterion only indirectly
        # controls that as we do not track regularization losses.

        # Note, these are arbitrary numbers (hyperparameters) where some of them
        # could still be made  user configurable.
        #
        # How often to check the early stopping criterion?
        # Every epoch seems to be a reasonable value. Though, the copy task
        # might have a massive training set size and small mini-batch size.
        # Therefore, we simply couple it with `val_iter`, such that we don't
        # have to compute the validation measure again.
        check_es_iter = config.val_iter
        # How many validation measures to take before the criterion will be
        # checked.
        es_warmup = None
        if hasattr(config, 'es_warm_up_iter'):
            es_warmup = config.es_warm_up_iter
        # With what rate does the importance of old measures (exponentially)
        # decay?
        es_decay = .99
        # Are we maximizing or minimizing the validation measure?
        es_type = 'max'
        if curr_iter > 0 and curr_iter % check_es_iter == 0 and \
                hasattr(config, 'early_stopping_thld') and \
                config.early_stopping_thld > 0:
            assert val_acc is not None

            es_val_iters.append(curr_iter)
            es_val_history.append(val_acc)
            es_val_weights = (np.array(es_val_weights) * es_decay).tolist()
            es_val_weights.append(1)

            writer.add_scalar('early_stop/task_%d/val_acc' % curr_task_id,
                              val_acc, curr_iter)

            # Check if ealy stopping criterion is met!
            if curr_iter > es_warmup:
                coeffs = np.polyfit(es_val_iters, es_val_history, 1,
                                    w=es_val_weights)
                slope = coeffs[0]

                writer.add_scalar('early_stop/task_%d/slope' % curr_task_id,
                                  slope, curr_iter)
                # FIXME The following logs may help us to finetune our early
                # stopping mechanism but can be deleted once the implementation
                # is finalized.
                fit_err = np.sum((np.polyval(coeffs, es_val_iters) - \
                                  es_val_history)**2)
                writer.add_scalar('early_stop/task_%d/error' % curr_task_id,
                                  fit_err, curr_iter)
                writer.add_scalar('early_stop/task_%d/std_hist' % curr_task_id,
                                  np.std(es_val_history), curr_iter)

                best_val = np.min(es_val_history) if es_type == 'min' else \
                    np.max(es_val_history)
                # Note, early stopping will never fire if the best value was
                # only a random flactuation. Therefore, we slightly weaken the
                # best value constraint.
                best_val_criterion = \
                    np.abs(best_val - config.es_best_val_diff) > 0

                if np.abs(slope) < config.early_stopping_thld and \
                        best_val_criterion:
                    logger.info('Early stopping criterion met! ' +
                                'Training of current task will be interrupted.')
                    break

    ###########################
    ### Restore best models ###
    ###########################
    # Note, it is important that we checkpoint the best models before we compute
    # importance weights for CL regularizers!
    if config.use_best_models and saved_models:
        wembs = shared.word_emb_lookups if \
                hasattr(shared, 'word_emb_lookups') and \
                not config.dont_learn_wembs else None

        stu.load_models(config.out_dir, device, logger, target_net, hnet=hnet,
            dnet=dnet, wembs=wembs, task_id=ckpt_task_id, train_iter=-1)

        # Delete checkpoints as they are not needed anymore and just waste
        # memory.
        ckpt_mnet_fn, ckpt_hnet_fn, ckpt_dnet_fn, ckpt_wembs_fn = \
            stu.ckpt_filenames(config.out_dir, task_id=ckpt_task_id,
                               train_iter=-1)

        for ckpt_fn in [ckpt_mnet_fn, ckpt_hnet_fn, ckpt_dnet_fn,
                        ckpt_wembs_fn]:
            dname, fname = os.path.split(ckpt_fn)
            ckpt_fns = [os.path.join(dname, f) for f in os.listdir(dname) if
                os.path.isfile(os.path.join(dname, f)) and
                f.startswith(fname)]
            for fn in ckpt_fns:
                logger.debug('Removing checkpoint %s.' % fn)
                os.remove(fn)

    #############################
    ### Compute SI Importance ###
    #############################
    if (config.num_tasks == 1 or curr_task_id < config.num_tasks - 1) and \
            config.use_si:
        logger.debug('Computing SI importances ...')
        si.si_compute_importance(target_net, regged_tnet_weights)

    ######################
    ### Compute Fisher ###
    ######################
    if (config.num_tasks == 1 or curr_task_id < config.num_tasks - 1) and \
            config.use_ewc:
        # For some experiments we run a single task, and might still want to 
        # look at the fisher values for this single task, so in that case we
        # compute the Fisher values even if we are at the last task.
        logger.debug('Computing diagonal Fisher elements ...')

        # get current gain mods
        if hnet is not None:
            gain_shifts = stu.hnet_forward(config, hnet, curr_task_id)
            gain_shifts = [gs.detach() for gs in gain_shifts]
        else:
            gain_shifts = None
        if config.use_masks:
            gain_shifts = ctx_masks[curr_task_id]

        ## Estimate diagonal Fisher elements.
        # We need a custom forward method through the main network, as the
        # current implementation does not allow the passing of internal
        # weights.
        def mnet_only_forward(mnet, params, X, data, batch_ids):
            X = stu.preprocess_inputs(config, shared, X, curr_task_id)
            mnet_kwargs = {}
            if isinstance(mnet, BiRNN):
                mnet_kwargs['seq_lengths'] = data.get_in_seq_lengths(batch_ids)
            # Note, we do not specify internal weights (i.e., we ignore
            # argument `params`). Instead, we expect the network to use the
            # internal ones, which are identical to `params`.
            Y = mnet.forward(X, weights=gain_shifts, **mnet_kwargs)
            return Y

        assert config.tbptt_fisher == -1 or config.tbptt_fisher >= 0
        if config.tbptt_fisher != -1:
            logger.debug('Using T-BPTT to compute diag. Fisher elements.')
        target_net.bptt_depth = config.tbptt_fisher
        # Note that ewc is only called for non-multitask scenario, so the
        # list of allowed outputs are all identical ranges. We can thus just
        # chose the first index.
        # Note, option `regression` doesn't matter, since we pass a
        # `custom_nll`.
        ewc.compute_fisher(curr_task_id, dhandler, regged_tnet_weights, device,
            target_net, empirical_fisher=True, online=True,
            gamma=config.ewc_gamma, n_max=config.n_fisher, regression=False,
            allowed_outputs=allowed_outputs,
            custom_forward=mnet_only_forward,time_series=True,
            custom_nll=ewc_loss_func, pass_ids=True)
        target_net.bptt_depth = -1

    ##############################
    ### Select coreset to keep ###
    ##############################
    if use_replay and config.coreset_size != -1:
        stu.update_coresets(config, shared, curr_task_id, dhandler)

    ############################
    ### Log Forgetting Stats ###
    ############################
    # Store (or log) some quantities that may help understanding / debugging of
    # the used CL regularizers.
    if not config.multitask and not config.hnet_all:
        curr_tnet_weights = torch.cat([p.detach().flatten().cpu() for p in \
                                       target_net.get_non_cm_weights()])
        shared.tnet_weights.append(curr_tnet_weights)
        if curr_task_id > 0:
            # Euclidean distance to first task's weights.
            edist_first = torch.sqrt(torch.sum((curr_tnet_weights -\
                shared.tnet_weights[0])**2))
            writer.add_scalar('cl/edist_first', edist_first, curr_task_id)
        if curr_task_id > 1:
            # Euclidean distance to previous task's weights.
            edist_prev = torch.sqrt(torch.sum((curr_tnet_weights -\
                shared.tnet_weights[curr_task_id-1])**2))
            writer.add_scalar('cl/edist_prev', edist_prev, curr_task_id)

        if config.use_ewc:
            # Log histogram of diagonal Fisher elements to see distribution of
            # "importance".
            diag_fisher = []
            diag_fisher_hh = [] # only for hidden to hidden weights

            # Get the list of meta information about regularized weights.
            n_cm = target_net._num_context_mod_shapes()
            regged_tnet_meta = target_net._param_shapes_meta[n_cm:]

            for ii, _ in enumerate(regged_tnet_weights):
                _, buff_f_name = ewc._ewc_buffer_names(None, ii, True)
                diag_fisher.append(getattr(target_net, buff_f_name))
                if 'info' in regged_tnet_meta[ii].keys() and \
                        regged_tnet_meta[ii]['info'] == 'hh':
                    diag_fisher_hh.append(getattr(target_net, buff_f_name))

            diag_fisher = torch.cat([p.detach().flatten().cpu() for p in \
                                     diag_fisher])
            diag_fisher_hh = torch.cat([p.detach().flatten().cpu() for p in \
                                     diag_fisher_hh])

            writer.add_scalar('ewc/min_fisher', torch.min(diag_fisher),
                              curr_task_id)
            writer.add_scalar('ewc/max_fisher', torch.max(diag_fisher),
                              curr_task_id)
            writer.add_histogram('ewc/fisher', diag_fisher, curr_task_id)
            writer.add_histogram('ewc/fisher_hh', diag_fisher_hh, curr_task_id)

            try:
                writer.add_histogram('ewc/log_fisher', torch.log(diag_fisher),
                                     curr_task_id)
                writer.add_histogram('ewc/log_fisher_hh', 
                                     torch.log(diag_fisher_hh), curr_task_id)
            except:
                # Should not happen, since diagonal elements should be positive.
                logger.warn('Could not write histogram of diagonal fisher ' +
                            'elements.')

        if config.use_si:
            # Log histogram of importance weights estimated by SI.
            omega = []
            for ii, _ in enumerate(regged_tnet_weights):
                buff_omega_name_i, _, _, _ = si._si_buffer_names(ii)
                omega.append(getattr(target_net, buff_omega_name_i))

            omega = torch.cat([p.detach().flatten().cpu() for p in omega])

            writer.add_scalar('si/min_omega', torch.min(omega), curr_task_id)
            writer.add_scalar('si/max_omega', torch.max(omega), curr_task_id)
            writer.add_histogram('si/omega', omega, curr_task_id)
            try:
                writer.add_histogram('si/log_omega', torch.log(omega),
                                     curr_task_id)
            except:
                # Should not happen, since omega is forced to be positive.
                logger.warn('Could not write histogram of SI importance ' +
                            'weights.')

        # How much did the hypernet output change for each task?
        if hnet is not None:
            with torch.no_grad():
                curr_hnet_out = torch.cat([p.detach().flatten().cpu() \
                    for p in stu.hnet_forward(config, hnet, curr_task_id)])
            shared.hnet_out.append(curr_hnet_out)
            for tt in range(curr_task_id):
                with torch.no_grad():
                    curr_hnet_out_tt = torch.cat([p.detach().flatten().cpu() \
                        for p in stu.hnet_forward(config, hnet, tt)])
                edist_tt = torch.sqrt(torch.sum((curr_hnet_out_tt -\
                    shared.hnet_out[tt])**2))
                writer.add_scalar('hreg/task_%d/edist' % tt, edist_tt,
                                  curr_task_id)

        # Log histogram of gains and shifts.
        if config.use_context_mod:
            gains = []
            shifts = []
            if hnet is not None:
                with torch.no_grad():
                    hnet_out = stu.hnet_forward(config, hnet, curr_task_id)
                assert not config.checkpoint_context_mod
                for ii, p in enumerate(hnet_out):
                    i_ps = target_net.hyper_shapes_learned_ref[ii]
                    meta = target_net.param_shapes_meta[i_ps]
                    if meta['name'] == 'cm_scale':
                        gains.append(p)
                    elif meta['name'] == 'cm_shift':
                        shifts.append(p)
            else:
                for cm_layer in target_net.context_mod_layers:
                    gains.append(cm_layer.gain)
                    shifts.append(cm_layer.shift)

            assert len(gains) == len(target_net.context_mod_layers)
            for ii, g in enumerate(gains):
                gains[ii] = target_net.context_mod_layers[ii].preprocess_gain(g)

            gains = torch.cat([p.detach().flatten().cpu() for p in gains])
            shifts = torch.cat([p.detach().flatten().cpu() for p in shifts])

            writer.add_histogram('cm/task_%d/gains' % curr_task_id, gains, 1)
            writer.add_histogram('cm/task_%d/shifts' % curr_task_id, shifts, 1)

    logger.info('Trained with %i samples.'%total_samples)

    #################################
    ### Context-mod checkpointing ###
    #################################
    # Checkpoint the context-mod weights if requested by user.
    if config.use_context_mod and config.checkpoint_context_mod:
        assert hnet is None
        logger.debug('Checkpointing context-modulation weights ...')
        for cm_layer in target_net.context_mod_layers:
            assert cm_layer.num_ckpts == curr_task_id
            # Note, this reinit might be overwritten based on
            # `config.context_mod_init`.
            cm_layer.checkpoint_weights(no_reinit=False)

    return loss_dict

def train_tasks(dhandlers, target_net, hnet, dnet, device, config, shared,
                logger, writer, ctx_masks, summary_keywords, summary_filename,
                task_loss_func=None, accuracy_func=None,
                ewc_loss_func=None, replay_fcts=None):
    """ Train continual learning experiments by looping through tasks.

    Args:
        dhandlers: The dataset handlers for classification.
        target_net: The model of the classifier.
        hnet: The model of the classifier hyper network.
        dnet: If existing, the replay model. ``None`` otherwise.
        device: Torch device (cpu or gpu).
        config (argparse.Namespace): The command line arguments.
        shared (argparse.Namespace): Miscellaneous information shared across
            functions.
        logger: Console (and file) logger.
        writer: The tensorboard summary writer.
        ctx_masks: Binary masks used for context-mod.
        summary_keywords (list): See argument ``summary_keywords`` of function
            :func:`sequential.train_utils_sequential.save_performance_summary`.
        summary_filename (dict): See argument ``summary_filename`` of function
            :func:`sequential.train_utils_sequential.save_performance_summary`.
        task_loss_func (fct_handler): The function to compute task loss.
        accuracy_func (fct_handler): The function to compute the accuracy, only
            necessary for classification tasks.
        ewc_loss_func (fct_handler): The function to compute the loss for EWC.
        replay_fcts (dict): Dictionary of function handles that are required for
            replay training. The following dictionary keys are expected:

            - ``'rec_loss'``: Function handle for the reconstruction loss
              (see for instance
              :func:`sequential.replay_utils.gauss_reconstruction_loss`).
            - ``'distill_loss'``: Loss for training with soft targets.
            - ``'soft_trgt_acc'``: Compute the accuracy based on soft targets.
              May be ``None``.

    Returns:
        (tuple): Tuple containing:

        - **return_code** (int): The return code may have the following values.

          - ``-1``: Training finished successfull.
          - A positive number: The return code represents the task ID after
            which the training was stopped prematurely. Hence, not all tasks
            have been trained.
        - **train_loss** (list): The components of the training loss during
          training on all tasks. It has length number of tasks, and each
          element is a dictionary with the task, ewc, and hnet components
          of the loss.
        - **test_loss** (np.array): The task test loss after training on 
          all tasks. The 1st dimension corresponds to the index of the last 
          task that has been learned, and the 2nd dimension corresponds to the 
          task that is being evaluated. Therefore, the performance of task 3 
          after learning task 2 is given by the indices [2, 3].
        - **test_acc** (np.array): The test accuracies on all tasks, after
          training in each of them. Only returned for classification tasks.
          The 1st dimension corresponds to the index of the last task that has 
          been learned, and the 2nd dimension corresponds to the task that is 
          being evaluated. Therefore, the performance of task 3 after learning 
          task 2 is given by the indices [2, 3].
    """
    if config.multitask:
        logger.info('Training in multitask fashion ...')
        n = 1
    else:
        logger.info('Training continually ...')
        n = config.num_tasks

    wembs = shared.word_emb_lookups if hasattr(shared, 'word_emb_lookups') and \
        not config.dont_learn_wembs else None

    return_code = -1

    # lists for storing all accuracy values (ntasks x ntasks) and loss info
    if config.classification:
        test_acc = []
    train_loss = []
    test_loss = []

    during_acc_criterion = [-1] * (n-1)
    if hasattr(config, 'during_acc_criterion') and \
            config.during_acc_criterion != '-1':
        min_daccs = misc.str_to_floats(config.during_acc_criterion)
        if len(min_daccs) == 1:
            during_acc_criterion = [min_daccs[0]] * (n-1)
        elif len(min_daccs) < n-1:
            logger.warn('Too less values in argument "during_acc_criterion". ' +
                        'Will be filled up with "-1" values.')
            during_acc_criterion[:len(min_daccs)] = min_daccs
        elif len(min_daccs) > n-1:
            logger.warn('Too many values in argument "during_acc_criterion". ' +
                        'Only the first %d values are considered.' % (n-1))
            during_acc_criterion = min_daccs[:n]
        else:
            during_acc_criterion = min_daccs

    # Begin training loop for the single tasks
    store_activations = False
    for t in range(0, n):

        # Only save activations when testing after all tasks have been learned,
        # if so indicated by the command line arguments.
        if t==config.num_tasks-1:
            store_activations = config.store_activations

        if config.checkpoint_context_mod:
            for cm_layer in target_net.context_mod_layers:
                # Note, 'constant' init is applied when the weights are created
                # or checkpointed.
                if config.context_mod_init == 'normal':
                    cm_layer.normal_init()
                elif config.context_mod_init == 'uniform':
                    cm_layer.uniform_init()
                elif config.context_mod_init == 'sparse':
                    cm_layer.sparse_init(sparsity=config.mask_fraction)

        # Train on the current task.
        loss_dict = train_one_task(dhandlers, target_net, hnet, dnet, device, \
                                   config, shared, logger, writer, t,
                                   ctx_masks=ctx_masks,
                                   task_loss_func=task_loss_func,
                                   accuracy_func=accuracy_func,
                                   ewc_loss_func=ewc_loss_func,
                                   replay_fcts=replay_fcts)
        train_loss.append(loss_dict)

        # Test on all tasks.
        plot_output = (t == n-1) and config.show_plots
        task_loss, task_acc = test(dhandlers, device, config, shared, logger,
            writer, target_net, hnet, ctx_masks=ctx_masks,
            store_activations=store_activations,
            plot_output=plot_output, task_loss_func=task_loss_func,
            accuracy_func=accuracy_func, num_trained=t+1)
        task_loss = [tl.cpu().numpy() for tl in task_loss]
        test_loss.append(task_loss)
        if config.classification:
            test_acc.append(task_acc)

        # Log the current results: loss or accuracy (for classification tasks).
        if config.classification:
            logger.info('Acc on current task: %.3f' % np.around(task_acc[t], \
                        decimals=1))
            logger.info('Acc on all tasks: ' + str(np.around(task_acc, \
                        decimals=1)))
        else:
            logger.info('Loss on current task: %.3f' % np.around(task_loss[t], \
                                                                 decimals=5))
            logger.info('Loss on all tasks: ' + str(np.around(task_loss, \
                                                              decimals=5)))

        # Generate or renitialize the networks if needed.
        if config.train_from_scratch and t < config.num_tasks-1:
            logger.debug('Creating new networks, since training from scratch ' +
                         'is activated ...')
            target_net, hnet, dnet = stu.generate_networks(config, shared,
                                                           dhandlers, device)
        elif config.reinit_tnet and t < config.num_tasks-1:
            logger.debug('Reinitializing target network weights ...')
            # Note, also when using replay, we only want to reinitialize the
            # target network. In this way, a fresh target network would be
            # trained on all data seen so far (replayed or from the current
            # task).
            # FIXME Make initialization user config dependent.
            target_net.custom_init(normal_init=False, normal_std=0.02,
                                   zero_bias=True)

        ### Store results obtained so far ###
        stu.save_performance_summary(config, shared,
            np.asarray(test_acc) if config.classification else \
                np.asarray(test_loss),
            train_loss, summary_keywords=summary_keywords,
            summary_filename=summary_filename, finished_training = t==n-1)

        ### Save current model ###
        if config.store_during_models:
            if task_acc is None:
                perf = -task_loss[t]
            else:
                perf = task_acc[t]
            stu.save_models(config.out_dir, logger, perf, target_net,
                            hnet=hnet, dnet=dnet, wembs=wembs, task_id=t)

        ### Check if last task got "acceptable" accuracy ###
        if not config.multitask and t < n-1 and during_acc_criterion[t] != -1 \
                and during_acc_criterion[t] > task_acc[t]:
            logger.error('During accuracy of task %d too small (%f < %f).' % \
                         (t, task_acc[t], during_acc_criterion[t]))
            logger.error('Training of future tasks will be skipped')
            return_code = t
            break

    ### Save final model ###
    if config.store_final_models:
        if task_acc is None:
            perf = - float(np.mean(task_loss))
        else:
            perf = float(np.mean(task_acc)) # avg. final accuracy
        stu.save_models(config.out_dir, logger, perf, target_net,
                        hnet=hnet, dnet=dnet, wembs=wembs)

    if config.classification:
        return return_code, train_loss, np.asarray(test_loss), \
            np.asarray(test_acc)
    else:
        return return_code, train_loss, np.asarray(test_loss)
