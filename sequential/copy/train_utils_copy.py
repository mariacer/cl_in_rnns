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
# @title           :sequential/copy/train_utils_copy.py
# @author          :mc
# @contact         :mariacer@ethz.ch
# @created         :20/03/2020
# @version         :1.0
# @python_version  :3.6.8
"""
Useful functions for training a recurrent network on the copy task.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from data.timeseries import copy_data, permuted_copy
from sequential.replay_utils import bernoulli_reconstruction_loss
from sequential import train_utils_sequential as tuseq


def generate_copy_tasks(config, logger, writer=None):
    """Generate a set of data handlers for copy tasks.

    Args:
        config (argparse.Namespace): Command-line arguments.
        logger: Logger object.
        writer: Tensorboard writer.

    Returns:
        (list): A list of data handlers for each task according to
        ``config.num_tasks``.
    """
    num_tasks = config.num_tasks
    # Note, the training set size is kind of ad-hoc since related work is not
    # properly reporting the details when training on this dataset.
    # One reference that has enough details is
    #   https://openreview.net/forum?id=HklliySFDS
    # They generate training (and test?) data on the fly. Hence, the number of
    # training samples depends on the chosen number of training iterations and
    # batchsize. In their case, the training set size per task must have been at
    # least 300,000 (given the specs that the authors provided after an email
    # correspondence -> though, we still couldn't reproduce the results).
    #
    # Anyway, we decided to use a fixed training set size to follow typical
    # Machine Learning habits (it also makes sense for EWC if one aims to
    # compute the empirical Fisher).
    # Note, the training set size may not be too low or overfitting becomes an
    # issue (on the other hand, a cleaner solution would be to combat that).
    num_train = 100000
    num_test  = 1000
    num_val = 1000

    # Set random state for permutations. This generator is instantiated
    # here such that we provide a deterministic set of random seeds for the
    # permutations.
    rstate_permute = np.random.RandomState(config.data_random_seed)

    scatter_pattern = False
    if hasattr(config, 'scatter_pattern'):
        scatter_pattern = config.scatter_pattern
    rstate_scatter = np.random.RandomState(config.data_random_seed)

    permute_xor = False
    if hasattr(config, 'permute_time'):
        permute_xor = config.permute_xor
    permute_xor_iter = None
    if hasattr(config, 'permute_xor_iter'):
        permute_xor_iter = config.permute_xor_iter
    permute_xor_separate = False
    if hasattr(config, 'permute_xor_separate'):
        permute_xor_separate = config.permute_xor_separate

    # Generate data.
    if config.use_new_permuted_dhandler and (config.permute_time or \
            config.permute_width or config.scatter_pattern):

        # Permute not implemented for sequences of varying length.
        assert config.input_len_step == 0
        assert config.input_len_variability == 0
        input_len = config.first_task_input_len
        
        pat_len = config.pat_len
        if config.scatter_pattern:
            if pat_len == -1 or pat_len > input_len:
                raise ValueError('Option "pat_len=%d" invalid when ' % pat_len +
                    'activating "scatter_pattern".')
        if pat_len == -1:
            pat_len = input_len

        # Create a permutation and scatter steps vectors per task.
        permutations = []
        scatter_steps = []
        for t in range(num_tasks):

            # Get random seed for the current task.
            # FIXME 1000 might be a bit low?
            rseed_permute = rstate_permute.randint(1000)
            rstate_task = np.random.RandomState(rseed_permute)
            rseed_scatter = rstate_scatter.randint(1e6)
            rstate_scatter_task = np.random.RandomState(rseed_scatter)

            permute_width = config.permute_width
            permute_time= config.permute_time
            if (permute_width or permute_time) and permute_xor and \
                    permute_xor_separate:
                permutations.append([])
                for _ in range(permute_xor_iter):
                    permutations[t].append(\
                        copy_data.CopyTask.create_permutation_matrix( \
                            permute_time, permute_width, pat_len,
                            config.seq_width, rstate_task))
            else: # `None` added in case no permutations are used.
                permutations.append( \
                    copy_data.CopyTask.create_permutation_matrix( \
                        permute_time, permute_width, pat_len, config.seq_width,
                        rstate_task))

            # Select timesteps to be used from the input to create the output
            # pattern.
            out_pat_steps = np.sort(rstate_scatter_task.choice(\
                np.arange(input_len), pat_len, replace=False))
            scatter_steps.append(out_pat_steps)

        if not config.scatter_pattern:
            scatter_steps = None

        data_handlers = permuted_copy.PermutedCopyList(permutations, input_len,
                seq_width=config.seq_width, num_train=num_train,
                num_test=num_test, num_val=num_val, pat_len=config.pat_len,
                rseed=config.data_random_seed, show_perm_change_msg=False,
                scatter_pattern=scatter_pattern, rseed_scatter=rseed_scatter,
                permute_xor=permute_xor, permute_xor_iter=permute_xor_iter,
                permute_xor_separate=permute_xor_separate,
                scatter_steps_list=scatter_steps,
                random_pad=config.random_pad, out_width=config.seq_out_width,
                pad_after_stop=config.pad_after_stop,
                pairwise_permute=config.pairwise_permute,
                revert_output_seq=config.revert_output_seq)

    else:
        data_handlers = []
        for t in range(num_tasks):
            mean_input_len = config.first_task_input_len + \
                t * config.input_len_step
            min_input_len = mean_input_len - config.input_len_variability
            max_input_len = mean_input_len + config.input_len_variability

            # Get random seed for the current task.
            rseed_permute = rstate_permute.randint(1000)
            rseed_scatter = rstate_scatter.randint(1e6)

            print('Creating data handler for task %d ...' % t)
            d = copy_data.CopyTask(min_input_len, max_input_len,
                seq_width=config.seq_width, out_width=config.seq_out_width,
                num_train=num_train, num_test=num_test, num_val=num_val,
                pat_len=config.pat_len, rseed=config.data_random_seed,
                permute_width=config.permute_width,
                permute_time=config.permute_time, rseed_permute=rseed_permute,
                scatter_pattern=scatter_pattern, rseed_scatter=rseed_scatter,
                permute_xor=permute_xor, permute_xor_iter=permute_xor_iter,
                permute_xor_separate=permute_xor_separate,
                random_pad=config.random_pad,
                pad_after_stop=config.pad_after_stop,
                pairwise_permute=config.pairwise_permute,
                revert_output_seq=config.revert_output_seq)
            data_handlers.append(d)
            print(d)

            # FIXME not a really nice solution to temper with internal
            # attributes.
            assert 'task_id' not in d._data.keys()
            d._data['task_id'] = t

            # Plot example data in Tensorboard.
            test_outs = d.output_to_torch_tensor( \
                d.get_train_outputs()[:6], 'cpu', mode='inference')
            test_outs = d._flatten_array(test_outs.numpy(), ts_dim_first=True)
            d.plot_samples('Training Samples - Task %d' % t,
                d.get_train_inputs()[:6], outputs=test_outs,
                num_samples_per_row=3, show=False, equalize_size=False)
            if writer is not None:
                writer.add_figure('data', plt.gcf(), t, close=True)

    if config.last_task_only:
        return [data_handlers[-1]]
    else:
        return data_handlers

def get_copy_loss_func(config, device, logger, ewc_loss=False):
    """Get a function handle that can be used as task loss function.

    Note, this function makes use of function
    :func:`sequential.train_utils_sequential.sequential_nll`.

    We use the Binary Cross Entropy loss, since our desired outputs should
    always be 0s or 1s. This function can be used to do multi-label binary
    classification, which is what we are interested in with the copy task,
    since several output units should be active at any given time.

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

    bce_loss = tuseq.sequential_nll(loss_type='bce', reduction='sum')

    sample_loss_func = lambda Y, T, tsf, beta: bce_loss(Y, T, None, None, None,
        ts_factors=tsf, beta=beta)

    # Unfortunately, we can't just use the above loss function, since we need
    # to respect the different sequence lengths.
    # We therefore create a custom time step weighting mask per sample in a
    # given batch.
    def task_loss_func(Y, T, data, allowed_outputs, empirical_fisher,
                       batch_ids):
        # Build batch specific timestep mask.
        tsf = torch.zeros(T.shape[0], T.shape[1]).to(T.device)

        pat_starts, pat_lengths = data.get_out_pattern_bounds(batch_ids)

        for i in range(batch_ids.size):
            ps = pat_starts[i]
            pe = ps + pat_lengths[i]

            tsf[ps:pe, i] = 1

            # Note, the `[i]` is necessary to avoid loosing the batch dimension.
            #loss += sample_loss_func(out_logits[s_start:s_end, [i], :],
            #    targets[s_start:s_end, [i], :], None, None)

        return sample_loss_func(Y, T, tsf, None)

    return task_loss_func


def get_accuracy(logit_outputs, targets, data, batch_ids):
    """Get the accuracy for the copy task.

    Accuracies are computed based only on the reconstructed input pattern. I.e.,
    for each sequence in the batch, we mask out the part where the input pattern
    is presented (including the stop bit) as well as the padded part.

    Args:
        logit_outputs (torch.Tensor): The linear network outputs.
        targets (torch.Tensor): The targets.
        data (data.Dataset): The dataset handler from which the ``targets``
            stem.
        batch_ids (numpy.ndarray): The IDs of the samples represented in
            ``targets`` as assigned by ``data``. This information is required
            to determine the actual sequence lengths.

    Returns:
        (tuple): Where tuple is containing:

        - (float): The classification accuracy.
        - (np.array): The classification accuracy per timestep. Note that this
          analysis only makes sense if the boundaries of the output sequences
          are identical for all samples.
    """
    predictions = logit_outputs >= 0

    # For simplicity we just loop over all samples in the given batch.
    num_correct = 0
    num_total = 0

    pat_starts, pat_lengths = data.get_out_pattern_bounds(batch_ids)

    compute_per_ts = False
    if np.unique(pat_lengths).size == 1:
        # Not correctly implemented for copy tasks with different pattern
        # lengths!
        compute_per_ts = True

        num_correct_per_ts = np.zeros(np.max(pat_lengths))
        num_total_width = 0

    for i in range(batch_ids.size):
        ps = pat_starts[i]
        pe = ps + pat_lengths[i]

        num_correct += torch.eq(predictions[ps:pe, i, :],
                                targets[ps:pe, i, :].byte()).sum().cpu().item()
        num_total += targets[ps:pe, i, :].numel()


        # Get the number of correct bits per timestep.
        if compute_per_ts:
            num_correct_per_ts += torch.eq(predictions[ps:pe, i, :],
                targets[ps:pe, i, :].byte()).sum(dim=1).cpu().numpy()
            num_total_width += targets[ps:pe, i, :].shape[1]

    accuracy = num_correct / num_total * 100.0

    if compute_per_ts:
        accuracy_per_ts = num_correct_per_ts / num_total_width * 100
    else:
        accuracy_per_ts = None

    return accuracy, accuracy_per_ts

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
    return bernoulli_reconstruction_loss

def get_distill_loss_func():
    """Get the loss function for distilling soft targets into the classifier.

    Returns:
        (func): A function handle.
    """
    def distill_loss_fct(config, X, Y_logits, T_soft_logits, data):
        if data._data['pat_len'] != -1:
            raise NotImplementedError

        # Detect stop bit.
        pattern_length = X[:, :, data.in_shape[0]-1].argmax(dim=0)

        # Distillation temperature. We just chose an arbitrary temperature
        # above 1 (no specific reason).
        T = 2.

        Y = torch.sigmoid(Y_logits / T)
        T_soft = torch.sigmoid(T_soft_logits / T)

        loss = 0
        for i in range(X.shape[1]):
            ps = pattern_length[i] + 1
            pe = ps + pattern_length[i]

            # Note, the multiplication by T^2 is done as it was proposed in the
            # original distillation paper. See comment in function:
            # mnets.classifier_interface.Classifier.knowledge_distillation_loss
            loss += F.binary_cross_entropy(Y[ps:pe, i, :], T_soft[ps:pe, i, :],
                                           reduction='sum') * T**2

        return loss / X.shape[1]

    return distill_loss_fct

def get_soft_trgt_acc_func():
    """Get the accuracy function that can deal with generated soft targets.

    Returns:
        (func): A function handle.
    """
    def soft_trgt_acc_fct(config, X, Y_logits, T_soft_logits, data):
        if data._data['pat_len'] != -1:
            raise NotImplementedError

        # The input pattern presentation ends with the stop bit. Hence, we can
        # use that information to obtain the pattern length.
        pattern_length = X[:, :, data.in_shape[0]-1].argmax(dim=0)

        predictions = Y_logits >= 0
        targets = T_soft_logits >= 0

        num_correct = 0
        num_total = 0
        for i in range(X.shape[1]):
            ps = pattern_length[i] + 1
            pe = ps + pattern_length[i]

            num_correct += torch.eq(predictions[ps:pe, i, :],
                                targets[ps:pe, i, :].byte()).sum().cpu().item()
            num_total += targets[ps:pe, i, :].numel()

        if num_total == 0:
            return 0.
        return num_correct / num_total * 100.0

    return soft_trgt_acc_fct

def draw_samples(title, samples, writer, tb_label, tb_iter, samples2=None):
    """Plot a batch of Copy Task inputs to tensorboard.

    Args:
        (....): See docstring of
            :func:`sequential.smnist.train_utils_smnist.draw_samples`.
    """
    all_samples = [samples]
    if samples2 is not None:
        all_samples.append(samples2)

    all_imgs = []

    for smp in all_samples:
        smp = smp.detach().cpu()
        n_T, n_B, n_F = smp.shape

        imgs = smp.permute(1, 2, 0)
        imgs = imgs.view(n_B, 1, n_F, n_T)

        img = make_grid(imgs[:6,:,:,:], nrow=1, padding=2)
        all_imgs.append(img)

    if len(all_imgs) == 1:
        img = all_imgs[0]
    else:
        img = make_grid(torch.stack(all_imgs), nrow=2, padding=10,
                        pad_value=0.)

    img = np.transpose(img, (1,2,0))

    plt.figure(figsize=(10, 6))
    plt.gca().set_axis_off()
    plt.title(title, size=20)
    plt.imshow(img)

    writer.add_figure(tb_label, plt.gcf(), tb_iter, close=True)

def compute_chance_level(dhandlers, config):
    """Compute chance level in copy task assuming perfect during accuracy.

    it is given by:

    .. math::
        100 + \sum_{k=1}^(K-1) (100*o_k + 50*(1-o_k))

    where :math:`K` is the number of tasks, and :math:`o_k` refers to the bit
    overlap between task :math:`k` and the final task.

    Args:
        dhandlers (list): The datahandlers
        config: The config.

    Returns:
        (float): The chance level.

    """

    reference_permutation = dhandlers[-1].permutation.copy()
    # FIXME not implemented.
    if isinstance(reference_permutation, list):
        return None

    chance_level = 0
    for i, dhandler in enumerate(dhandlers):
        if config.multi_head or config.use_masks:
            # If we use multi-head, the output head is always different, so it
            # doesn't matter whether there is overlap between the tasks.
            # If we use masks, each task has a different subnetwork, so it also
            # doesn't help if there is overlap.
            if i<len(dhandlers)-1:
                overlap = 0.
            else:
                # Perfect overlap for the last task and itself.
                overlap = 1.
        else:
            overlap = (dhandler.permutation==reference_permutation).mean()
        chance_level += overlap*100 + (1-overlap)*50

    chance_level /= len(dhandlers)

    return chance_level