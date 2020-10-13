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
# @title          :sequential/pos_tagging/train_utils_pos.py
# @author         :ch, mc, be
# @contact        :henningc@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# @python_version :3.6.10
"""
Utility functions for the training a multilingual PoS tagger
------------------------------------------------------------
"""
from sklearn.metrics import f1_score
import torch
import numpy as np

from data.timeseries.mud_data import get_mud_handlers
from mnets.classifier_interface import Classifier
from sequential.replay_utils import gauss_reconstruction_loss
from sequential import train_utils_sequential as tuseq


def generate_tasks(config, logger, writer=None):
    """Generate a set of data handlers for PoS tagging.

    Each dataset (and thus task) will represent a different language. See class
    :class:`data.timeseries.mud_data.MUDData` for details.

    Args:
        config (argparse.Namespace): Command-line arguments.
        logger: Logger object.
        writer: Tensorboard writer.

    Returns:
        (list): A list of data handlers for each task according to
        ``config.num_tasks``.
    """

    logger.info('Running PoS experiment.')
    dhandlers = get_mud_handlers('../../datasets', num_tasks=config.num_tasks)

    for t, d in enumerate(dhandlers):
        # FIXME not a really nice solution to temper with internal
        # attributes.
        assert 'task_id' not in d._data.keys()
        d._data['task_id'] = t

    return dhandlers

def get_loss_func(config, device, logger, ewc_loss=False):
    """Get a function handle that can be used as task loss function.

    Note, this function makes use of function
    :func:`sequential.train_utils_sequential.sequential_nll`.

    Since PoS tagging is a classification task, this function implements a
    sequential cross-entropy loss.

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
    if hasattr(config, 'ts_weighting') or \
            hasattr(config, 'ts_weighting_fisher'):
        raise NotImplementedError('The copy task dataset has a fixed loss ' +
            'weighting scheme, which is not configurable.')

    ce_loss = tuseq.sequential_nll(loss_type='ce', reduction='sum')

    sample_loss_func = lambda Y, T, tsf, beta: ce_loss(Y, T, None, None, None,
        ts_factors=tsf, beta=beta)

    # Unfortunately, we can't just use the above loss function, since we need
    # to respect the different sequence lengths.
    # We therefore create a custom time step weighting mask per sample in a
    # given batch.
    def task_loss_func(Y, T, data, allowed_outputs, empirical_fisher,
                       batch_ids):
        # Build batch specific timestep mask.
        tsf = torch.zeros(T.shape[0], T.shape[1]).to(T.device)

        seq_lengths = data.get_out_seq_lengths(batch_ids)

        for i in range(batch_ids.size):
            sl = int(seq_lengths[i])

            tsf[:sl, i] = 1

        return sample_loss_func(Y, T, tsf, None)

    return task_loss_func

def get_accuracy_func(config):
    """Get the accuracy function for a PoS task.

    Note:
        The accuracy will be computed only for **unpadded timesteps**.

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
        input_data = data._data['in_data'][batch_ids,:]

        predicted = logit_outputs.argmax(dim=2)
        targets = targets.argmax(dim=2)
        all_compared = predicted == targets

        num_correct = 0
        num_total = 0

        for i in range(batch_ids.size):
            # we exclude tokens for which there is no word embedding from the
            # accuracy computation. these tokens have dict_idx == 0
            comp_idx = np.arange(0, int(seq_lengths[i]))
            exclude_idx = np.where(input_data[i,:] == 0)
            comp_idx = np.setdiff1d(comp_idx, exclude_idx)

            num_correct += all_compared[comp_idx, i].sum().cpu().item()
            num_total += len(comp_idx)

        if num_total != 0:
            accuracy = 100. * num_correct / num_total
        else:
            # FIXME Can this case really appear?
            accuracy = 0

        return accuracy, None # accuracy per ts not yet implemented

    return get_accuracy

def get_vae_rec_loss_func():
    """Get the reconstruction loss function for the replay VAE.

    Returns:
        (func): A function handle.
    """
    # Note, we don't apply any masking to the loss for a reason. One might
    # think, it is smart to mask the zero-padded part in the reconstruction
    # loss. However, we want to be able to replay those sequences in its
    # totality (including the zero-padding), because the samples drawn from the
    # latent space will always have the length of the maximum number of
    # timesteps.

    # MSE loss seems sensible for reconstructing word embeddings, that's why we
    # chose it.
    return gauss_reconstruction_loss

def get_distill_loss_func():
    """Get the loss function for distilling soft targets into the classifier.

    Returns:
        (func): A function handle.
    """
    def distill_loss_fct(config, X, Y_logits, T_soft_logits, data,
                         in_seq_lens=None):
        if in_seq_lens is None:
            raise NotImplementedError('This distillation loss is currently ' +
                'only implemented if sequence lengths are provided, as they ' +
                'can\'t be inferred easily.')
        # Note, input and output sequence lengths are identical for the PoS
        # dataset.

        assert np.all(np.equal(X.shape[:2], T_soft_logits.shape[:2]))
        # Note, targets and predictions might have different head sizes if a
        # growing softmax is used.
        assert np.all(np.equal(Y_logits.shape[:2], T_soft_logits.shape[:2]))

        # Disillation temperature.
        T=2.

        target_mapping = None
        if config.all_task_softmax:
            target_mapping = list(range(T_soft_logits.shape[2]))

        dloss = 0
        total_num_ts = 0

        for bid in range(X.shape[1]):
            sl = int(in_seq_lens[bid])
            total_num_ts += sl

            Y_logits_i = Y_logits[:sl, bid, :]
            T_soft_logits_i = T_soft_logits[:sl, bid, :]

            dloss += Classifier.knowledge_distillation_loss(Y_logits_i,
                T_soft_logits_i, target_mapping=target_mapping,
                device=Y_logits.device, T=T) * sl

        return dloss / total_num_ts

    return distill_loss_fct

def get_soft_trgt_acc_func():
    """Get the accuracy function that can deal with generated soft targets.

    Returns:
        (func): A function handle.
    """
    def soft_trgt_acc_fct(config, X, Y_logits, T_soft_logits, data,
                          in_seq_lens=None):
        if in_seq_lens is None:
            raise NotImplementedError('This soft accuracy is currently ' +
                'only implemented if sequence lengths are provided, as they ' +
                'can\'t be inferred easily.')
        # Note, input and output sequence lengths are identical for the PoS
        # dataset.

        assert np.all(np.equal(X.shape[:2], T_soft_logits.shape[:2]))
        # Note, targets and predictions might have different head sizes if a
        # growing softmax is used.
        assert np.all(np.equal(Y_logits.shape[:2], T_soft_logits.shape[:2]))

        num_correct = 0
        total_num_ts = 0

        for bid in range(X.shape[1]):
            sl = int(in_seq_lens[bid])
            total_num_ts += sl

            Y_logits_i = Y_logits[:sl, bid, :]
            T_soft_logits_i = T_soft_logits[:sl, bid, :]

            predicted = Y_logits_i.argmax(dim=1)
            label = T_soft_logits_i.argmax(dim=1)

            num_correct += (predicted == label).sum().cpu().item()

        return num_correct / total_num_ts * 100.

    return soft_trgt_acc_fct

def compute_f_score(logit_outputs, targets, data, batch_ids,
                    average='weighted'):
    """Compute `f-score <https://scikit-learn.org/stable/modules/generated\
/sklearn.metrics.f1_score.html>`.

    Note:
        PoS labels might be highly imbalanced. Thus, looking at plain accuracies
        might give the wrong impression.

    Args:
        (....) See docstring of function
            :func:`sequential.copy.train_utils_copy.get_accuracy`.
        average (str): See argument ``average`` of function
            :func:`sklearn.metrics.f1_score`.

    Returns:
        (float): The f-score.
    """

    seq_lengths = data.get_out_seq_lengths(batch_ids)
    input_data = data._data['in_data'][batch_ids,:]

    predicted = logit_outputs.argmax(dim=2)
    targets = targets.argmax(dim=2)

    y_true = []
    y_pred = []

    for i in range(batch_ids.size):
        # we exclude tokens for which there is no word embedding from the
        # fscore computation. these tokens have dict_idx == 0
        comp_idx = np.arange(0,int(seq_lengths[i]))
        exclude_idx = np.where(input_data[i,:] == 0)
        comp_idx = np.setdiff1d(comp_idx,exclude_idx)

        y_true.extend(targets[comp_idx, i].cpu().numpy().tolist())
        y_pred.extend(predicted[comp_idx, i].cpu().numpy().tolist())

    return f1_score(y_true, y_pred, labels=range(17), average=average,
                    zero_division=1)


if __name__ == '__main__':
    pass


