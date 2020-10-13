#!/usr/bin/env python3
# Copyright 2019 Maria Cervera

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
# @title           :sequential/audioset/train_utils_audioset.py
# @author          :mc
# @contact         :mariacer@ethz.ch
# @created         :20/03/2020
# @version         :1.0
# @python_version  :3.6.8
"""
Useful functions for training a recurrent network on the Audioset task.
"""
import numpy as np
import torch

from data.timeseries import audioset_data
from data.timeseries.split_audioset import get_split_audioset_handlers
from mnets.classifier_interface import Classifier
from sequential.replay_utils import gauss_reconstruction_loss
from sequential import train_utils_sequential as tuseq

def _generate_tasks(config, logger, experiment='split_audioset', writer=None):
    """Generate a set of data handlers for Audioset task.

    Args:
        config: Command-line arguments.
        logger: Logger object.
        writer: Tensorboard writer.
        experiment (str,optional): Type of experiment. See argument `experiment` 
            of function :func:`probabilistic.prob_mnist.train_bbb.run` or
            :func:`probabilistic.prob_cifar.train_avb.run`.

    Returns:
        (list): A list of data handlers for each task according to
        ``config.num_tasks``.
    """
    assert experiment in ['audioset', 'split_audioset']

    if experiment == 'audioset':
        # Equivalent to doing multitask, but avoiding the dataset splitting.
        logger.info('Running multitask Audioset experiment.')
        dhandler = audioset_data.AudiosetData('../../datasets',
            use_one_hot=True, validation_size=config.val_set_size,
            rseed=config.data_random_seed)
        dhandler._data['task_id'] = 0
        dhandlers = [dhandler]
    elif experiment.startswith('split'):
        logger.info('Running SplitAudioset experiment.')
        dhandlers = get_split_audioset_handlers('../../datasets',
            use_one_hot=True, num_tasks=config.num_tasks,
            num_classes_per_task=config.num_classes_per_task,
            rseed=config.data_random_seed, validation_size=config.val_set_size)
        for t, dh in enumerate(dhandlers):
            # FIXME not a really nice solution to temper with internal
            # attributes.
            assert 'task_id' not in dh._data.keys()
            dh._data['task_id'] = t
        if config.num_tasks * config.num_classes_per_task < 10:
            logger.info('Running SplitAudioset experiments only for classes ' +
                        '0 - %d.' \
                        % (config.num_tasks * config.num_classes_per_task - 1))
    else:
        raise ValueError('Experiment type "%s" unknown.' % experiment)

    assert(len(dhandlers) == config.num_tasks)

    return dhandlers

def get_loss_func(config, device, logger, ewc_loss=False):
    """Get a function handle that can be used as task loss function.

    Note, this function makes use of function
    :func:`sequential.train_utils_sequential.sequential_nll`.

    Args:
        config (argparse.Namespace): The command line arguments.
        device: Torch device (cpu or gpu).
        logger: Console (and file) logger.
        ewc_loss (bool): Whether the loss is determined for task training or
            to compute Fisher elements via EWC. Note, based on the user
            configuration, the loss computation might be different.

    Returns:
        (func): A function handler as described by argument ``custom_nll``
        of function :func:`utils.ewc_regularizer.compute_fisher`, if option
        ``pass_ids=True``.

        Note:
            This loss **sums** the NLL across the batch dimension. A proper
            scaling wrt other loss terms during training would require a
            multiplication of the loss with a factor :math:`N/B`, where
            :math:`N` is the training set size and :math:`B` is the mini-batch
            size.
    """
    # Log-likelihoods of timesteps are usually just summed. Here, the user
    # can change this to a weighted sum.
    if not ewc_loss:
        ts_weighting = config.ts_weighting
    else:
        ts_weighting = config.ts_weighting_fisher

    # Note, there is no padding applied in this dataset.
    purpose = 'Fisher' if ewc_loss else 'loss'
    if ts_weighting == 'none' or ts_weighting == 'unpadded':
        logger.debug('Considering the NLL of all timesteps for %s computation.'
                     % purpose)
    elif ts_weighting == 'last':
        logger.debug('Considering the NLL of last timestep for ' +
                     '%s computation.' % purpose)
    elif ts_weighting == 'last_ten_percent':
        logger.debug('Considering the NLL of last 10% of timestep ' +
                     'for %s computation.' % purpose)
    else:
        assert ts_weighting == 'discount'
        logger.debug('Weighting the NLL of the later timesteps more than ' +
                     'the NLL of earlier timesteps for %s computation.' \
                     % purpose)

    ce_loss = tuseq.sequential_nll(loss_type='ce', reduction='sum')

    # Build batch specific timestep mask.
    # Note, all samples have the same sequence length.
    seq_length = 10
    ts_factors = torch.zeros(seq_length, 1).to(device)

    # FIXME We can compute the weigthings outside of this function, since
    # they are static for all batches (no padding).
    if ts_weighting == 'none' or ts_weighting == 'unpadded':
        ts_factors = None
    if ts_weighting == 'last':
        ts_factors[-1, :] = 1
    elif ts_weighting == 'last_ten_percent':
        sl_10 = seq_length // 10
        ts_factors[-sl_10:, :] = 1
    else:
        assert ts_weighting == 'discount'
        gamma = 1.
        discount = 0.9
        for tt in range(seq_length-1, -1, -1):
            ts_factors[tt, 0] = gamma
            gamma *= discount

    # FIXME What is a good way of normalizing weights?
    # The timestep factors should be normalized such that the final
    # NLL strength corresponds to the original one. But what is the
    # original one? Either the one, that only takes the last timestep
    # into account (hence, `ts_factors` should sum to 1) or the one that
    # takes all timesteps into account (hence, `ts_factors` should
    # sum to `seq_length`).
    # Since there is only one label per sample, I decided that only 1
    # timestep counts, the last one.
    if ts_factors is not None:
        ts_factors /= ts_factors.sum()

    # We need to ensure additionally that `batch_ids` can be passed to the loss,
    # even though we don't use them here as all sequences have the same length.
    # Note, `dh`, `ao`, `ef` are also unused by `ce_loss` and are just provided 
    # to certify a common interface.
    loss_func = lambda Y, T, dh, ao, ef, _: ce_loss(Y, T, None, None, None,
        ts_factors=ts_factors, beta=None)

    return loss_func

def get_accuracy_func(config):
    """Get the accuracy function for an Audioset task.

    Note:
        The accuracy will be computed depending **only on the prediction in
        the last timestep**, where the last timestep refers to the **unpadded
        sequence**.

    Args:
        config (argparse.Namespace): The command line arguments.

    Returns:
        (func): An accuracy function handle.
    """
    def get_accuracy(logit_outputs, targets, data, batch_ids):
        """Get the accuracy for an Audioset task.

        Note that here we expect that, in the multi-head scenario, the correct 
        output head has already been selected, and ``logit_outputs`` corresponds 
        to the outputs in the correct head.

        Args:
            (....) See docstring of function
                :func:`sequential.copy.train_utils_copy.get_accuracy`.

        Returns:
            (....) See docstring of function
                :func:`sequential.copy.train_utils_copy.get_accuracy`.
        """
        seq_length = targets.shape[0]
        batch_size = targets.shape[1]

        # Pick the last prediction per sample.
        logit_outputs = logit_outputs[seq_length-1, :, :]
        # Get the predicted classes.
        # Note, we don't need to apply the softmax, since it doesn't change the
        # argmax.
        predicted = logit_outputs.argmax(dim=1)
        # Targets are the same for all timesteps.
        targets = targets[0, :, :]
        targets = targets.argmax(dim=1)

        accuracy = 100. * (predicted == targets).sum().cpu().item() / \
            batch_size

        return accuracy, None # accuracy per ts not yet implemented

    return get_accuracy

def get_vae_rec_loss_func():
    """Get the reconstruction loss function for the replay VAE.

    Returns:
        (func): A function handle.
    """
    return gauss_reconstruction_loss

def get_distill_loss_func():
    """Get the loss function for distilling soft targets into the classifier.

    Returns:
        (func): A function handle.
    """
    # We ignore CLI argument `ts_weighting` here and just take the last time
    # step.
    def distill_loss_fct(config, X, Y_logits, T_soft_logits, data):
        # Note, targets and predictions might have different head sizes if a
        # growing softmax is used.
        assert np.all(np.equal(Y_logits.shape[:2], T_soft_logits.shape[:2]))

        # Only compute loss for last timestep.
        Y_logits = Y_logits[-1, :, :]
        T_soft_logits = T_soft_logits[-1, :, :]

        target_mapping = None
        if config.all_task_softmax:
            target_mapping = list(range(T_soft_logits.shape[1]))

        return Classifier.knowledge_distillation_loss(Y_logits, T_soft_logits,
            target_mapping=target_mapping, device=Y_logits.device, T=2.)

    return distill_loss_fct

def get_soft_trgt_acc_func():
    """Get the accuracy function that can deal with generated soft targets.

    Returns:
        (func): A function handle.
    """
    def soft_trgt_acc_fct(config, X, Y_logits, T_soft_logits, data):
        # Only compute accuracy for last timestep.
        Y_logits = Y_logits[-1, :, :]
        T_soft_logits = T_soft_logits[-1, :, :]

        predicted = Y_logits.argmax(dim=1)
        label = T_soft_logits.argmax(dim=1)

        accuracy = 100. * (predicted == label).sum().cpu().item() / X.shape[1]

        return accuracy

    return soft_trgt_acc_fct