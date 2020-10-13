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
# @title          :sequential/student_teacher/train_utils_st.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :07/28/2020
# @version        :1.0
# @python_version :3.6.10
"""
Utility functions for the training in a student teacher setting
---------------------------------------------------------------
"""
import numpy as np
import torch.nn.functional as F

from data.timeseries.rnd_rec_teacher import RndRecTeacher
from sequential.replay_utils import gauss_reconstruction_loss
from sequential import train_utils_sequential as tuseq

def generate_tasks(config, logger, writer=None):
    """Generate a set of data handlers for copy tasks.

    Args:
        config (argparse.Namespace): Command-line arguments.
        logger: Logger object.
        writer: Tensorboard writer.

    Returns:
        (list): A list of data handlers for each task according to
        ``config.num_tasks``.
    """
    # Hardcoded dataset sizes.
    num_train = 10000
    num_test  = 1000
    num_val = config.val_set_size

    if config.last_task_only:
        raise NotImplementedError

    # We use a pseudo-random random seed for each new task.
    rstate_all = np.random.RandomState(config.data_random_seed)

    dhandlers = []

    for t in range(config.num_tasks):
        rseed_t = rstate_all.randint(1e6)

        logger.info('Creating data handler for task %d.' % t)

        d = RndRecTeacher(num_train=num_train, num_test=num_test,
            num_val=num_val, n_in=config.input_feature_dim,
            n_out=config.output_feature_dim, sigma='tanh',
            orth_A=config.orth_teacher_hid_mat,
            rank_A=config.rank_teacher_hid_mat,
            max_sv_A=config.max_sv_teacher_hid_mat, no_extra_fc=False,
            input_range=(-1, 1), n_ts_in=config.num_timesteps_in,
            n_ts_out=config.num_timesteps_out, rseed=rseed_t)

        # FIXME not a really nice solution to temper with internal attributes.
        assert 'task_id' not in d._data.keys()
        d._data['task_id'] = t

        dhandlers.append(d)

    return dhandlers

def get_loss_func(config, device, logger, ewc_loss=False):
    """Get a function handle that can be used as task loss function.

    Note, this function makes use of function
    :func:`sequential.train_utils_sequential.sequential_nll`.

    We use the MSE loss.

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
        raise NotImplementedError('The student-teacher setting has a fixed ' +
            'loss weighting scheme, which is not configurable.')

    mse_loss = tuseq.sequential_nll(loss_type='mse', reduction='sum')

    sample_loss_func = lambda Y, T, data, allowed_outputs, empirical_fisher, \
        batch_ids: mse_loss(Y, T, None, None, None, ts_factors=None, mask=None)

    return sample_loss_func

def get_vae_rec_loss_func():
    """Get the reconstruction loss function for the replay VAE.

    Returns:
        (func): A function handle.
    """
    return gauss_reconstruction_loss

def get_distill_loss_func():
    """Get the loss function for distilling soft targets into the network.

    Returns:
        (func): A function handle.
    """
    def distill_loss_fct(config, X, Y_logits, T_soft_logits, data):
        # What is a proper way of distilling here? We could increase the
        # variance of the assumed output likelihood, i.e., multiplying the MSE
        # loss with a scalar < 1.
        loss = F.mse_loss(Y_logits, T_soft_logits, reduction='sum')

        return loss / X.shape[1]

    return distill_loss_fct


if __name__ == '__main__':
    pass


