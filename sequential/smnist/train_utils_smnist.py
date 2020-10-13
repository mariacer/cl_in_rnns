#!/usr/bin/env python3
# Copyright 2020 Maria Cervera

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
# @title           :sequential/smnist/train_utils_smnist.py
# @author          :mc
# @contact         :mariacer@ethz.ch
# @created         :20/03/2020
# @version         :1.0
# @python_version  :3.6.8
"""
Useful functions for training a recurrent network on the SMNIST task.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from data.timeseries import smnist_data
from data.timeseries.split_smnist import get_split_smnist_handlers
from mnets.classifier_interface import Classifier
from sequential.replay_utils import gauss_reconstruction_loss
from sequential import train_utils_sequential as tuseq

def _generate_tasks(config, logger, experiment='split_smnist', writer=None):
    """Generate a set of data handlers for SMNIST task.

    Args:
        config: Command-line arguments.
        logger: Logger object.
        experiment (str): Type of experiment. See argument `experiment` of
            function :func:`probabilistic.prob_mnist.train_bbb.run` or
            :func:`probabilistic.prob_cifar.train_avb.run`.
        writer: Tensorboard writer.

    Returns:
        (list): A list of data handlers for each task according to
        ``config.num_tasks``.
    """
    assert experiment in ['smnist', 'split_smnist']

    if experiment == 'smnist':
        # Equivalent to doing multitask, but avoiding the dataset splitting.
        logger.info('Running multitask SMNIST experiment.')
        dhandler = smnist_data.SMNISTData('../../datasets', use_one_hot=True,
                                          validation_size=config.val_set_size)
        dhandler._data['task_id'] = 0
        dhandlers = [dhandler]
    elif experiment.startswith('split'):
        logger.info('Running SplitSMNIST experiment.')
        dhandlers = get_split_smnist_handlers('../../datasets',
            use_one_hot=True, validation_size=config.val_set_size,
            num_tasks=config.num_tasks,
            num_classes_per_task=config.num_classes_per_task)
        for t, dh in enumerate(dhandlers):
            # FIXME not a really nice solution to temper with internal attributes.
            assert 'task_id' not in dh._data.keys()
            dh._data['task_id'] = t
        if config.num_tasks * config.num_classes_per_task < 10:
            logger.info('Running SplitSMNIST experiments only for digits ' +
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
            See docstring of function
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
    # Note, I haven't thought to much about whether the MSE loss makes sense for
    # this dataset.
    # Note, we don't apply any masking to the loss for a reason. One might
    # think, it is smart to mask the zero-padded part in the reconstruction
    # loss. However, we want to be able to replay those sequences in its
    # totality (including the zero-padding), because the samples drawn from the
    # latent space will always have the length of the maximum number of
    # timesteps.
    return gauss_reconstruction_loss

def get_distill_loss_func():
    """Get the loss function for distilling soft targets into the classifier.

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

        if config.distill_across_time:
            # Emd-of-digit feature.
            # Note, the softmax would be an easy way to obtain per-timestep
            # probabilities, that a digit has ended. But since this feature
            # vector `X[:, :, 3]` is ideally a 1-hot encoding, it shouldn't
            # be squashed via a softmax, that would blur out the probabilities.
            #ts_weights = F.softmax(X[:, :, 3], dim=0).detach()
            ts_weights = X[:, :, 3].clone()
            ts_weights[ts_weights < 0] = 0
            # Avoid division by zero in case all elements of `X[:, :, 3]` are
            # negative.
            ts_weights /= ts_weights.sum() + 1e-5
            ts_weights = ts_weights.detach()

            # For distillation, we use a tempered softmax.
            T_soft = F.softmax(T_soft_logits / T, dim=2)
            if config.all_task_softmax and Y_logits.shape[2] != T_soft.shape[2]:
                # Pad new classes to soft targets.
                T_soft = F.pad(T_soft, (0, data.num_classes), mode='constant',
                               value=0)
                assert Y_logits.shape[2] == T_soft.shape[2]

            # Distillation loss.
            loss = -(T_soft * F.log_softmax(Y_logits / T, dim=2)).sum(dim=2) * \
                T**2
            loss *= ts_weights

            # Sum across time (note, weights sum to 1) and mean across batch
            # dimension.
            return loss.sum(dim=0).mean()
        else:
            # Note, smnist samples have the end-of-sequence bit as last
            # timestep, the rest is padded.
            seq_lengths = X[:, :, 3].argmax(dim=0)
            inds = seq_lengths.cpu().numpy() - 1
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
        # Note, smnist samples have the end-of-sequence bit as last timestep,
        # the rest is padded.
        seq_lengths = X[:, :, 3].argmax(dim=0)
        inds = seq_lengths.cpu().numpy() - 1
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

def draw_samples(title, samples, writer, tb_label, tb_iter, samples2=None):
    """Plot a batch of SMNIST inputs to tensorboard.

    Args:
        title (str): Title of plot.
        samples (torch.Tensor): Batch of SMNIST samples.
        writer: Tensorboard writer.
        tb_label: Tensorboard label.
        tb_iter: Tensorboard iteration of plot.
        samples2 (torch.Tensor, optional): Another batch of SMNIST samples,
            drawn next to the batch ``samples``.
    """
    assert len(samples.shape) == 3 and samples.shape[2] == 4

    all_samples = [samples]
    if samples2 is not None:
        all_samples.append(samples2)

    all_imgs = []

    for smp in all_samples:
        smp = smp.cpu()
        imgs = torch.zeros(smp.shape[1], 1, 28, 28)

        # Build image from stroke data.
        for b in range(imgs.shape[0]):
            x_idx = 0
            y_idx = 0

            # End-of-digit
            eod = smp[:,b, 3].argmax()

            for i in range(smp.shape[0]):
                if i == eod: # end-of-digit
                    break

                x_idx += int(smp[i, b, 0])
                y_idx += int(smp[i, b, 1])

                x_idx = 0 if x_idx < 0 else x_idx
                y_idx = 0 if y_idx < 0 else y_idx
                x_idx = 27 if x_idx > 27 else x_idx
                y_idx = 27 if y_idx > 27 else y_idx
                imgs[b, 0, x_idx, y_idx] = 255

        imgs = imgs.permute(0, 1, 3, 2)

        img = make_grid(imgs[:36,:,:,:], nrow=6)
        all_imgs.append(img)

    if len(all_imgs) == 1:
        img = all_imgs[0]
    else:
        img = make_grid(torch.stack(all_imgs), nrow=2, padding=10,
                        pad_value=255.)

    img = np.transpose(img, (1,2,0)) / 255.

    plt.figure(figsize=(10, 6))
    plt.gca().set_axis_off()
    plt.title(title, size=20)
    plt.imshow(img)

    writer.add_figure(tb_label, plt.gcf(), tb_iter, close=True)