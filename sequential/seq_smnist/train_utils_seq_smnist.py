#!/usr/bin/env python3
# Copyright 2020 Benjamin Ehret

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
# @title           :sequential/smnist/train_utils_seq_smnist.py
# @author          :be
# @contact         :behret@ethz.ch
# @created         :14/04/2020
# @version         :1.0
# @python_version  :3.6.8
"""
Useful functions for training a recurrent network on the sequential SMNIST task.
"""
import numpy as np
import torch
import torch.nn.functional as F

from data.timeseries.seq_smnist import SeqSMNIST
from mnets.classifier_interface import Classifier
from sequential.replay_utils import gauss_reconstruction_loss
from sequential import train_utils_sequential as tuseq

def _generate_tasks(config, logger, writer=None):
    """Generate a set of data handlers for SMNIST task.

    Args:
        config: Command-line arguments.
        logger: Logger object.
        writer: Tensorboard writer.

    Returns:
        (list): A list of data handlers for each task according to
        ``config.num_tasks``.
    """

    logger.info('Running Sequential SMNIST experiment.')

    # set random state
    r_state = np.random.RandomState(config.data_random_seed)

    if config.num_tasks <= 5:
        digit_pairs = [(0,1),(2,3),(4,5),(6,7),(8,9)]
    else:
        # generate digit pairs and randomize order
        digit_pairs = []
        for i in range(10):
            for j in range(i+1,10):
                digit_pairs.append((i,j))
            
        # shuffle digit pairs
        rand_idx = np.arange(len(digit_pairs))
        r_state.shuffle(rand_idx)
        digit_pairs = [digit_pairs[i] for i in rand_idx[:config.num_tasks]]

    dhandlers  = []
    for i,dp in enumerate(digit_pairs):
        rseed = r_state.randint(100000)
        d = SeqSMNIST('../../datasets', use_one_hot=True,
            sequence_length=config.ssmnist_seq_len, digits=dp, num_train=12000,
            num_test=2000, num_val=config.val_set_size, rseed=rseed,
            two_class=config.ssmnist_two_classes)

        # FIXME not a really nice solution to temper with internal attributes.
        assert 'task_id' not in d._data.keys()
        d._data['task_id'] = i

        dhandlers.append(d)

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

    purpose = 'Fisher' if ewc_loss else 'loss'
    if ts_weighting == 'none':
        logger.debug('Considering the NLL of all timesteps (including padded ' +
                     'ones) for %s computation.' % purpose)
    elif ts_weighting == 'unpadded':
        logger.debug('Considering the NLL of all unpadded timesteps for ' +
                     '%s computation.' % purpose)
    elif ts_weighting == 'last':
        logger.debug('Considering the NLL of last unpadded timestep for ' +
                     '%s computation.' % purpose)
    elif ts_weighting == 'last_ten_percent':
        logger.debug('Considering the NLL of last 10% of unpadded timestep ' +
                     'for %s computation.' % purpose)
    else:
        assert ts_weighting == 'discount'
        logger.debug('Weighting the NLL of the later timesteps more than ' +
                     'the NLL of earlier timesteps for %s computation.' \
                     % purpose)

    ce_loss = tuseq.sequential_nll(loss_type='ce', reduction='sum')

    # Unfortunately, we can't just use the above loss function, since we need
    # to respect the different sequence lengths.
    # We therefore create a custom time step weighting mask per sample in a
    # given batch.
    def task_loss_func(Y, T, data, allowed_outputs, empirical_fisher,
                       batch_ids):
        # Build batch specific timestep mask.
        ts_factors = torch.zeros(T.shape[0], T.shape[1]).to(T.device)

        seq_lengths = data.get_out_seq_lengths(batch_ids)

        if ts_weighting == 'none':
            ts_factors = None
        if ts_weighting == 'unpadded':
            for i, sl in enumerate(seq_lengths):
                ts_factors[:sl, i] = 1
        elif ts_weighting == 'last':
            ts_factors[seq_lengths-1, np.arange(seq_lengths.size)] = 1
        elif ts_weighting == 'last_ten_percent':
            for i, sl in enumerate(seq_lengths):
                sl_10 = sl // 10
                ts_factors[(sl-sl_10):sl, i] = 1
        else:
            assert ts_weighting == 'discount'
            gamma = 1.
            discount = 0.9
            max_num_ts = Y.shape[0]
            dc_factors = torch.zeros(max_num_ts)
            for tt in range(max_num_ts, -1, -1):
                dc_factors[tt] = gamma
                gamma *= discount
            for i, sl in enumerate(seq_lengths):
                ts_factors[:sl, i] = dc_factors[-sl:]

        # FIXME What is a good way of normalizing weights?
        # The timestep factors should be normalized such that the final
        # NLL strength corresponds to the original one. But what is the
        # original one? Either the one, that only takes the last timestep
        # into account (hence, `ts_factors` should sum to 1) or the one that
        # takes all unpadded timesteps into account (hence, `ts_factors` should
        # sum to `seq_lengths`).
        # Since there is only one label per sample, I decided that only 1
        # timestep counts, the last one.
        if ts_factors is not None:
            ts_factors /= ts_factors.sum(dim=0)[None, :]

        return ce_loss(Y, T, None, None, None, ts_factors=ts_factors, beta=None)

    return task_loss_func

def get_accuracy_func(config):
    """Get the accuracy function for an SMNIST task.

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
        """Get the accuracy for an SMNIST task.

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
        seq_lengths = data.get_out_seq_lengths(batch_ids)

        # Pick the last prediction per sample.
        logit_outputs = logit_outputs[seq_lengths-1, \
                                      np.arange(seq_lengths.size), :]
        # Get the predicted classes.
        # Note, we don't need to apply the softmax, since it doesn't change the
        # argmax.
        predicted = logit_outputs.argmax(dim=1)
        # Targets are the same for all timesteps.
        targets = targets[0, :, :]
        targets = targets.argmax(dim=1)

        accuracy = 100. * (predicted == targets).sum().cpu().item() / \
            seq_lengths.size

        return accuracy, None # accuracy per ts not yet implemented

    return get_accuracy

def get_vae_rec_loss_func():
    """Get the reconstruction loss function for the replay VAE.

    Returns:
        (func): A function handle.
    """
    # See comment in function `sequential.smnist.train_utils_smnist.\
    # get_vae_rec_loss_func`.
    return gauss_reconstruction_loss

def get_distill_loss_func():
    """Get the loss function for distilling soft targets into the classifier.

    The returned function will make use of the end-of-digit information
    associated with individual SMNIST images. If 3 SMNIST images are
    concatenated, then the correct class label should be outputted at the end-
    of-digit bit of last digit. Hence, we simply take the 3 maximum values
    across the end-of-digit feature and the last occuring max value is
    considered the distillation timestep.

    Returns:
        (func): A function handle.
    """
    def distill_loss_fct(config, X, Y_logits, T_soft_logits, data):
        assert np.all(np.equal(X.shape[:2], T_soft_logits.shape[:2]))
        # Note, targets and predictions might have different head sizes if a
        # growing softmax is used.
        assert np.all(np.equal(Y_logits.shape[:2], T_soft_logits.shape[:2]))

        # Disillation temperature.
        T=2.

        n_digs = config.ssmnist_seq_len

        # Note, smnist samples have the end-of-sequence bit as last
        # timestep, the rest is padded. Since there are `n_digs` digits per
        # sample, we consider the last end-of-sequence digit to determine the
        # unpadded sequence length.
        eod_features = X[:, :, 3].cpu().numpy()
        seq_lengths = np.argsort(eod_features, axis=0)[-n_digs:].max(axis=0)
        inds = seq_lengths - 1
        inds[inds < 0] = 0

        # Only compute loss for last timestep.
        Y_logits = Y_logits[inds, np.arange(inds.size), :]
        T_soft_logits = T_soft_logits[inds, np.arange(inds.size), :]

        target_mapping = None
        if config.all_task_softmax:
            target_mapping = list(range(T_soft_logits.shape[1]))

        return Classifier.knowledge_distillation_loss(Y_logits,
            T_soft_logits, target_mapping=target_mapping,
            device=Y_logits.device, T=T)

    return distill_loss_fct

def get_soft_trgt_acc_func():
    """Get the accuracy function that can deal with generated soft targets.

    Returns:
        (func): A function handle.
    """
    def soft_trgt_acc_fct(config, X, Y_logits, T_soft_logits, data):
        n_digs = config.ssmnist_seq_len

        eod_features = X[:, :, 3].cpu().numpy()
        seq_lengths = np.argsort(eod_features, axis=0)[-n_digs:].max(axis=0)
        inds = seq_lengths - 1
        inds[inds < 0] = 0

        # Only compute accuracy for last timestep.
        Y_logits = Y_logits[inds, np.arange(inds.size), :]
        T_soft_logits = T_soft_logits[inds, np.arange(inds.size), :]

        predicted = Y_logits.argmax(dim=1)
        label = T_soft_logits.argmax(dim=1)

        accuracy = 100. * (predicted == label).sum().cpu().item() / \
            inds.size

        return accuracy

    return soft_trgt_acc_fct