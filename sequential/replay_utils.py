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
# @title          :sequential/replay_utils.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :04/22/2020
# @version        :1.0
# @python_version :3.6.10
"""
Utility functions for sequential replay methods
-----------------------------------------------

A collection of helper functions regarding the training of replay methods
(see, for instance, **HNET+R** in
`Oswald et al. <https://arxiv.org/abs/1906.00695>`__ or
`van de Ven et al. <https://arxiv.org/abs/1809.10635>`__). Here, we specifically
focus on the replay of sequential data.

The functions are outsourced from the training function
:func:`sequential.train_sequential.train_one_task` to improve readability and
increase modularity.
"""
import numpy as np
import torch
from torch.distributions import Normal
import torch.nn.functional as F

from data.timeseries.audioset_data import AudiosetData
from data.timeseries.copy_data import CopyTask
from data.timeseries.mud_data import MUDData
from data.timeseries.smnist_data import SMNISTData
from data.timeseries.seq_smnist import SeqSMNIST
from mnets.bi_rnn import BiRNN
from sequential import train_utils_sequential as stu

def replay_samples(config, shared, device, all_dhandlers, task_ids, batch_size,
                   dnet, hnet=None, dnet_weights=None, hnet_weights=None,
                   hnet_tembs=None, split_by_id=False,
                   replay_all_data=False, coresets=None, ret_seq_lens=False):
    """Replay samples using the decoder network.

    This function will create a batch of replayed samples. If the decoder can be
    task-conditioned, then the samples will be uniformly split across the IDs
    provided via ``task_ids``.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Miscellaneous information shared across
            functions.
        device: PyTorch device.
        all_dhandlers (list): A list of data handlers for all tasks. Required
            to obtain meta information, like the maximum number of timesteps per
            task.
        task_ids (list or tuple): List of task IDs. If ``hnet`` is not ``None``,
            these are used to select task embeddings. If the classifier has
            a multihead output, then the decoder expects the task id as 1-hot
            input. Otherwise, it is only used to determine the maximum number
            of timesteps per sequence.
        batch_size (int): How many samples should be replayed.
        dnet: The decoder network.
        hnet (optional): The hypernetwork.
        dnet_weights: Weights to be passed to the ``dnet``.
        hnet_weights: Weights to be passed to the ``hnet``.
        hnet_tembs (list): The task embeddings of the ``hnet`` to be used.
        split_by_id (bool): If ``True``, return value ``inputs`` will always be
            provided as a list, each entry corresponding to only one task ID.
        replay_all_data (bool): If set, the decoder is ignored and actual
            training data from ``all_dhandlers`` is replayed.

            Caution:
                Setting this option breaks the continual learning assumption
                and should only be used as a sanity check!
        coresets (list, optional): A list of coreset samples per previous task.
            If provided, the decoder ``dnet`` is ignored.
        ret_seq_lens (bool): If ``True``, and additional return value
            ``in_seq_lens`` will be returned.

    Returns:
        (tuple): Tuple containing:

        - **inputs** (torch.Tensor or list): The batch of replayed samples.

          Note:
              If the maximum number of timesteps deviates between the tasks
              (according to ``all_dhandlers``), then ``inputs`` will be a list.

          Note:
              Those inputs will be detached from the computational graph.
        - **task_labels** (numpy.ndarray or list): The task ID of each sample in
          the batch ``inputs``.

          Note:
              Will be a list of numpy arrays if ``inputs`` is a list.
        - **perm_task_ids**: A permutation of ``task_ids``, according to which
          samples in ``inputs`` are distributed if returned as a list. Note, the
          random permutation is done to ensure that each task appears on average
          equally often in the returned batches.
        - **in_seq_lens** (list, optional): Only returned if ``ret_seq_lens``
          is set. The return value will usually be ``None`` except if real data
          is replayed (see options ``replay_all_data`` and ``coresets``). In
          this case, the actual sequence lengths of the replayed inputs are
          returned.
    """
    task_labels = None
    in_seq_lens = None

    assert isinstance(task_ids, (list, tuple))

    ### Distribute the task IDs equally across the batch ###
    task_ids = np.random.permutation(task_ids)

    max_num_ts = []
    num_per_id = []

    task_labels = []
    num_ids = len(task_ids)
    rem = batch_size - num_ids * (batch_size // num_ids)
    for i, t in enumerate(task_ids):
        max_num_ts.append(all_dhandlers[t].max_num_ts_in)

        num = batch_size // num_ids
        if rem > 0:
            rem -= 1
            num += 1

        num_per_id.append(num)
        task_labels.append(np.ones(num) * t)
    assert np.sum(num_per_id) == batch_size

    if replay_all_data or coresets is not None:
        # Mutual exclusive.
        assert not replay_all_data or coresets is None

        ###########################
        ### Replay of true data ###
        ###########################
        # FIXME We make our life simple and ignore `single_batch`.
        assert split_by_id
        inputs = []
        in_seq_lens = []

        for i, t in enumerate(task_ids):
            bs = num_per_id[i]
            if replay_all_data:
                batch = all_dhandlers[t].next_train_batch(bs, return_ids=True)
                samples = batch[0]
                sample_lengths = all_dhandlers[t].get_out_seq_lengths(batch[2])
            else:
                coreset = shared.coresets[t]
                batch_inds = np.random.randint(0, coreset.shape[0], bs)
                samples = coreset[batch_inds, :]
                sample_lengths = all_dhandlers[t].get_out_seq_lengths( \
                    shared.coreset_sample_ids[t][batch_inds])
            # Note, I use `train` mode as the actual replay decoder is also
            # trained with training samples.
            X_t = all_dhandlers[t].input_to_torch_tensor(samples, device,
                                                         mode='train')
            X_t = stu.preprocess_inputs(config, shared, X_t, t)

            # This might seem a bit hacky, but can reduce runtime a lot. Since
            # we enforce `split_by_id`, we just assume that noone wants to
            # concatenate the separate `X_t` again afterwards.
            max_sl = int(sample_lengths.max())
            X_t = X_t[:max_sl, :, :]

            inputs.append(X_t)
            in_seq_lens.append(sample_lengths)

    else:
        ###########################
        ### Replay of fake data ###
        ###########################
        # Can we pack everything into a single batch or are the sequences of
        # different lengths?
        single_batch = len(np.unique(max_num_ts)) == 1

        if single_batch:
            task_labels = np.concatenate(task_labels)
            assert task_labels.size == batch_size

        ### Draw latent input (from prior) ###
        #if single_batch:
        #    num_ts = max_num_ts[0]
        #    z = torch.normal(torch.zeros(num_ts, batch_size,
        #                                 config.latent_dim),
        #                     config.latent_std).to(device)
        z = []
        for i, num_ts in enumerate(max_num_ts):
            bs = num_per_id[i]

            z.append(torch.normal(torch.zeros(num_ts, bs, config.latent_dim),
                                  config.latent_std).to(device))

        ### Compute replay samples ###
        inputs = []

        if hnet is not None:
            assert dnet_weights is None
            # Alternatively, we could just take the task embeddings from `hnet`.
            assert hnet_tembs is not None

            for i, t in enumerate(task_ids):
                if hasattr(config, 'use_new_hnet') and config.use_new_hnet:
                    # Assuming `hnet_tembs` are all conditional weights.
                    dnet_weights = hnet.forward(cond_id=int(t), weights={
                        'uncond_weights': hnet_weights,
                        'cond_weights': hnet_tembs})
                else:
                    temb = hnet_tembs[t]
                    dnet_weights = hnet.forward(task_emb=temb,
                                                theta=hnet_weights)

                inputs.append(dnet.forward(z[i], weights=dnet_weights))

            if single_batch:
                inputs = torch.cat(inputs, dim=1)

        else:
            if config.multi_head:
                # Append task ID to decoder input.
                for i, t in enumerate(task_ids):
                    z[i] = F.pad(z[i], (0, config.num_tasks), mode='constant',
                                 value=0)
                    z[i][:,:,-config.num_tasks+t] = 1.

            if single_batch:
                z = torch.cat(z, dim=1)
                inputs = dnet.forward(z, weights=dnet_weights)
            else:
                for i, t in enumerate(task_ids):
                    inputs.append(dnet.forward(z[i], weights=dnet_weights))

    if isinstance(inputs, list):
        inputs = [inp.detach() for inp in inputs]
    else:
        inputs = inputs.detach()

    if split_by_id and not isinstance(inputs, list):
        inputs_tmp = inputs
        task_labels_tmp = task_labels
        inputs = []
        task_labels = []
        for tid in task_ids:
            inputs.append(inputs_tmp[:, task_labels_tmp == tid, :])
            task_labels.append(task_labels_tmp[task_labels_tmp == tid])

    if not replay_all_data and coresets is None:
        # FIXME A bit hacky. We might need to apply a dataset specific
        # activation function on the logits retrieved from the decoder.
        if isinstance(all_dhandlers[0], CopyTask):
            if isinstance(inputs, list):
                inputs = [torch.sigmoid(inp) for inp in inputs]
            else:
                inputs = torch.sigmoid(inputs)
        else:
            assert isinstance(all_dhandlers[0],
                              (AudiosetData, MUDData, SMNISTData, SeqSMNIST))

    if ret_seq_lens:
        return inputs, task_labels, task_ids, in_seq_lens
    return inputs, task_labels, task_ids

def get_soft_targets(config, all_dhandlers, inputs, task_labels, cnet,
                     cnet_weights, trained_task_id=None, input_lens=None):
    """Compute soft targets wit classifier ``cnet``.

    Soft targets can be used for
    `distillation <https://arxiv.org/abs/1503.02531>`__. In classification
    tasks, soft targets are simply the softmax outputs of a classifier.

    Note:
        The targets outputted by this function already have the correct
        output size. Hence, there is **no** need to call
        :func:`sequential.train_utils_sequential.adjust_targets_to_head`.

    Note:
        The targets will be provided as logits.

    Args:
        config (argparse.Namespace): Command-line arguments.
        all_dhandlers (list): A list of data handlers for all tasks. Required
            to obtain meta information, like the type of output required.
        inputs (torch.Tensor): The classifier inputs.
        task_labels (numpy.ndarray): Task ID of each sample in the batch
            ``inputs``.
        cnet: The classifier networks.
        cnet_weights: The weights that should be passed to the classifier
            network.
        trained_task_id: See argument ``trained_task_id`` of function
            :func:`sequential.train_utils_sequential.out_units_of_task`
        input_lens (numpy.ndarray, optional): Only utilized if network ``cnet``
            is of :class:`mnets.bi_rnn.BiRNN`, where sequence lengths are
            expected to be passed.

    Returns:
        (torch.Tensor): The `soft` **logit** targets corresponding to
        ``inputs``.

        Note:
            The returned targets are detached from the computational graph.
    """
    task_ids = np.unique(task_labels).astype(np.int).tolist()

    if isinstance(cnet, BiRNN) and input_lens is not None:
        Y_logits = cnet.forward(inputs, weights=cnet_weights,
                                seq_lengths=input_lens)
    else:
        Y_logits = cnet.forward(inputs, weights=cnet_weights)

    # In a multihead setting, we have to select the correct output head from
    # Y_logits.
    # Note, in all cases, we don't want to consider the full `Y_logits`, as it
    # also contains the VAE latent space.
    Y_logits_new = None
    for t in task_ids:
        allowed_outputs = stu.out_units_of_task(config, all_dhandlers[t], t,
                dhandlers=None, trained_task_id=trained_task_id)
        if Y_logits_new is None:
            Y_logits_new = torch.empty(*Y_logits.shape[:2],
                                       len(allowed_outputs)).to(Y_logits.device)
        else:
            assert Y_logits_new.shape[2] == len(allowed_outputs)

        # FIXME Didn't find a better way of solving indexing errors.
        Y_logits_new[:, task_labels == t, :] = \
            Y_logits[:, task_labels == t, :][:, :, allowed_outputs]

    # Compute actual target vectors from logits.
    #if all_dhandlers[0].classification:
    #    T_soft = F.softmax(Y_logits_new, dim=2)
    #else:
    #    assert isinstance(all_dhandlers[0], CopyTask)
    #    # Binary targets are expected.
    #    T_soft = F.sigmoid(Y_logits_new)

    T_soft = Y_logits_new
    #Z_latent = Y_logits[-(2*config.latent_dim):]

    return T_soft.detach()

def gauss_reconstruction_loss(in_logits, out_logits):
    r"""Compute the reconstruction loss assuming a Gaussian likelihood.

    We assume the Gaussian likelihood to have the identity matrix as covariance
    matrix. Hence, the decoder output ``out_logits`` will be interpreted as
    mean of this distribution and we simply compute

    .. math::

        \text{NLL} = \frac{1}{2} \sum_t \lVert x - y \rVert^2

    Note:
        We reconstruct over the whole time sequence, disregarding the fact that
        sequences might be zero-padded. We do this because when producing
        random sequences (e.g., see :func:`replay_samples`), those will always
        span the maximum number of timesteps. Hence, zero-padding has to be
        faithfully reconstructed.

    Note:
        This function does not mean over the batch dimension, as this operation
        is performed within :func:`compute_vae_loss`.

    Args:
        in_logits (torch.Tensor): The original inputs to be reconstructed.
        out_lgits (torch.Tensor): The corresponding decoder outputs.

    Returns:
        (torch.Tensor): A scalar tensor.
    """
    loss = F.mse_loss(in_logits, out_logits, reduction='sum')

    return loss

def bernoulli_reconstruction_loss(in_logits, out_logits):
    r"""Compute the reconstruction loss assuming a Bernoulli likelihood.

    This function is similar to :func:`gauss_reconstruction_loss`.

    It computes the following loss (assuming features are modelled as
    independent Bernoulli distributions):

    .. math::

        \text{NLL} = \sum_t \sum_f - x_t^{(f)} \log y_t^{(f)} - \
            (1 - x_t^{(f)}) \log (1 - y_t^{(f)})

    Args:
        (....): See docstring of function :func:`gauss_reconstruction_loss`.

    Returns:
        (torch.Tensor): A scalar tensor.
    """
    loss = F.binary_cross_entropy_with_logits(out_logits, in_logits,
                                              reduction='sum')

    return loss

def compute_vae_loss(config, cnet_inputs, cnet_logits, task_ids, rec_fct, dnet,
                     hnet=None):
    r"""Compute the Variational Autoencoder (VAE) loss.

    The VAE loss consists of a reconstruction term (e.g., see function
    :func:`gauss_reconstruction_loss` or :func:`bernoulli_reconstruction_loss`)
    and a latent space regularization

    .. math::

        KL \big( q_\varphi(z_{1:T} \mid x_{1:T}) \,||\, p(z_{1:T}) \big) \
            \approx \sum_t \
            KL \big( q_\varphi(z_t \mid z_{<t}, x_{\leq t}) \,||\, p(z_t) \big)

    Since we assume the prior and the approximate posterior to be Gaussian
    (on a per timestep basis), the KL terms within the sum can be analytically
    evaluated.

    Args:
        config (argparse.Namespace): Command-line arguments.
        cnet_inputs (list): The inputs the the classifier used to retrieve the
            classifier outputs ``cnet_logits``.
        cnet_logits (list): List of tensors, each containing logit outputs of
            the classifier network. From these outputs, the last
            ``2 * config.latent_dim`` dimensions are assumed to encode the
            distribution :math:`q_\varphi(z_t \mid z_{<t}, x_{\leq t})`.
        task_ids (list): List of task IDs, corresponding to the entries of
            ``cnet_logits``.
        rec_fct (func): A function handle (such as
            :func:`gauss_reconstruction_loss` or
            :func:`bernoulli_reconstruction_loss`), that can be used to compute
            the reconstruction loss.
        dnet: Decoder network.
        hnet (optional): Hypernetwork.

    Returns:
        (tuple): Tuple containing:

        - **rec_loss** (torch.Tensor): Scalar tensor encoding the reconstruction
          loss.
        - **pm_loss** (torch.Tensor): Scalar tensor encoding the prior-matching
          loss.
        - **dnet_logits** (list): List of tensors containing the raw decoder
          outputs for samples in ``cnet_logits``.
    """
    assert len(cnet_logits) == len(task_ids)
    batch_size = 0
    rec_loss = 0
    pm_loss = 0
    dnet_logits = []

    prior_var = config.latent_std**2
    prior_logvar = 2 * np.log(config.latent_std)

    for i, c_out in enumerate(cnet_logits):
        batch_size += c_out.shape[1]

        ######################
        ### Prior-matching ###
        ######################
        z_mean = c_out[:,:,-(2*config.latent_dim):-config.latent_dim]
        z_logvar = c_out[:,:,-config.latent_dim:]
        z_std = torch.exp(0.5 * z_logvar)

        # KL per timestep.
        #kl_t = 0.5 * torch.sum(-1 + \
        #    (z_logvar.exp() + (0 - z_mean).pow(2)) / prior_var + \
        #    prior_var - z_logvar, dim=2)

        # We directly sum over every dimension, but later we divide by the
        # batchsize, i.e., essentially we summed over feature and time dimension
        # and meaned over the batch dimension.
        kl = 0.5 * torch.sum(-1 + \
            (z_logvar.exp() + (0 - z_mean).pow(2)) / prior_var + \
            prior_logvar - z_logvar)

        pm_loss += kl

        ##############################
        ### Compute decoder logits ###
        ##############################
        # Sample from latent distribution `q`.
        z = Normal(z_mean, z_std).rsample()

        if hnet is not None:
            dnet_weights = stu.hnet_forward(config, hnet, task_ids[i])

            d_out = dnet.forward(z, weights=dnet_weights)

        else:
            if config.multi_head:
                # Append task ID to decoder input.
                z = F.pad(z, (0, config.num_tasks), mode='constant', value=0)
                z[:,:,-config.num_tasks+task_ids[i]] = 1.

            d_out = dnet.forward(z)

        dnet_logits.append(d_out)

        ###########################
        ### Reconstruction loss ###
        ###########################
        rec_loss += rec_fct(cnet_inputs[i], d_out)

    pm_loss /= batch_size
    rec_loss /= batch_size

    return rec_loss, pm_loss, dnet_logits

if __name__ == '__main__':
    pass


