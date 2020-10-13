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
# @title           :sequential/ht_analyses/supervised_dimred_utils.py
# @author          :mc
# @contact         :mariacer@ethz.ch
# @created         :04/08/2020
# @version         :1.0
# @python_version  :3.6.8
"""
Utils for making supervised dimensionality reduction on the hidden state of RNNs
--------------------------------------------------------------------------------
"""
import argparse
import numpy as np
import os
import shutil
from tensorboardX import SummaryWriter
import torch

from mnets.simple_rnn import SimpleRNN
import sequential.train_utils_sequential as stu
from utils.torch_utils import get_optimizer

def forward_post_rec(srnn, h, weights=None, distilled_params=None,
                     condition=None):
    """Compute the output :math:`y` of this network given the hidden states
    :math:`h`.

    Note that for an LSTM, :math:`h` refers to the output of the
    recurrent layer, whereas for a vanilla RNN :math:`h` refers to the
    internal hidden state before the fully-connected layer within the
    recurrent layer.

    Args:
        (....): See docstring of method :meth:`mnets.SimpleRNN.forward`.
        srnn (mnets.simple_rnn.SimpleRNN): An instance of class
            :class:`mnets.simple_rnn.SimpleRNN`. This modified forward method
            only works for this specific network type.

    Returns:
        (torch.Tensor): The output of the network.
    """
    assert isinstance(srnn, SimpleRNN)
    assert distilled_params is None

    if ((not srnn._use_context_mod and srnn._no_weights) or \
            (srnn._no_weights or srnn._context_mod_no_weights)) and \
            weights is None:
        raise Exception('Network was generated without weights. ' +
                        'Hence, "weights" option may not be None.')

    #######################
    ### Extract weights ###
    #######################
    # Extract which weights should be used.
    int_weights, cm_weights = srnn.split_weights(weights)

    ### Split context-mod weights per context-mod layer.
    cm_inputs_weights, cm_fc_pre_layer_weights, cm_fc_layer_weights, \
        cm_rec_layer_weights, n_cm_rec, cmod_cond = srnn.split_cm_weights(
            cm_weights, condition, num_ts=h.shape[0])

    ### Extract internal weights.
    fc_pre_w_weights, fc_pre_b_weights, rec_weights, fc_w_weights, \
        fc_b_weights = srnn.split_internal_weights(int_weights)

    ##########################
    ### Output Computation ###
    ##########################
    cm_offset = 0
    if srnn._use_context_mod and srnn._context_mod_inputs:
        cm_offset = 1

    ### Compute recurrent output for vanilla RNNs.
    d = 0 # we assume there is only one recurrent layer
    assert len(srnn._rnn_layers) == 1
    if not srnn._use_lstm:
        if srnn._use_context_mod:
            h = _compute_recurrent_outputs(srnn, h, d, rec_weights[d],
                                           cm_rec_layer_weights[d], cmod_cond)
        else: 
            h = _compute_recurrent_outputs(srnn, h, d, rec_weights[d],
                                           None, None)

    ### Fully-connected output layer activities.
    cm_offset = srnn._cm_rnn_start_ind + n_cm_rec
    ret_hidden, h = srnn.compute_fc_outputs(h, fc_w_weights, fc_b_weights, \
        srnn._num_fc_cm_layers, cm_fc_layer_weights, cm_offset, cmod_cond,
        True, [])

    return h

def compute_next_hidden_states(srnn, x, h0, weights=None, distilled_params=None,
                               condition=None):
    """Compute the internal recurrent states subsequent to the current timestep.

    This function receives as input an initial hidden state :math:`h_t` and a
    set of external inputs :math:`x` and computes all subsequent hidden states.

    Args:
        (....): See docstring of function :func:`forward_post_rec`.
        x (torch.Tensor): The external input for all subsequent inputs.
        h0 (torch.Tensor): The initial hidden states.

    Returns:
        (torch.Tensor): The internal hidden activities of the recurrent network.
    """
    assert distilled_params is None
    # This function assumes there is a single recurrent layer. To be implemented
    # for all other cases.
    assert len(srnn._rnn_layers) == 1

    if ((not srnn._use_context_mod and srnn._no_weights) or \
            (srnn._no_weights or srnn._context_mod_no_weights)) and \
            weights is None:
        raise Exception('Network was generated without weights. ' +
                        'Hence, "weights" option may not be None.')

    #######################
    ### Extract weights ###
    #######################
    # Extract which weights should be used.
    int_weights, cm_weights = srnn.split_weights(weights)

    ### Split context-mod weights per context-mod layer.
    cm_inputs_weights, cm_fc_pre_layer_weights, cm_fc_layer_weights, \
        cm_rec_layer_weights, n_cm_rec, cmod_cond = srnn.split_cm_weights(
            cm_weights, condition, num_ts=x.shape[0])

    ### Extract internal weights.
    fc_pre_w_weights, fc_pre_b_weights, rec_weights, fc_w_weights, \
        fc_b_weights = srnn.split_internal_weights(int_weights)

    ###########################
    ### Forward Computation ###
    ###########################

    ### Recurrent layer activities.
    d = 0 # we assume there is only one recurrent layer
    assert len(srnn._rnn_layers) == 1
    if srnn._use_context_mod:
        if srnn.use_lstm:
            raise NotImplementedError
        else:
            _, h_int = srnn.compute_hidden_states(x, d, rec_weights[d],
                cm_rec_layer_weights[d], cmod_cond, h_0=h0)
    else: 
        if srnn.use_lstm:
            raise NotImplementedError
        else:
            _, h_int = srnn.compute_hidden_states(x, d, rec_weights[d],
                None, None, h_0=h0)

    return h_int


def _compute_recurrent_outputs(srnn, h, layer_ind, int_weights, cm_weights,
                           ckpt_id):
    """Compute the outputs for a vanilla recurrent layer from a sequence of
    internal hidden recurrent states :math:`h`.

    If so specified, context modulation is applied before or after the
    nonlinearities.

    Args:
        (....): See docstring of method
            :meth:`mnets.SimpleRNN.compute_hidden_states`.
        srnn (mnets.simple_rnn.SimpleRNN): An instance of class
            :class:`mnets.simple_rnn.SimpleRNN`. This method only works for this
            specific network type.
        h: The internal recurrent state :math:`h` of the layer.
            :math:`h` has shape ``[sequence_len, batch_size, n_hidden]``.

    Returns:
        (torch.Tensor): The sequence of output recurrent states given the
            internal recurrent states. It has shape
            ``[sequence_len, batch_size, n_hidden]``.
    """
    assert isinstance(srnn, SimpleRNN)

    use_cm = srnn._use_context_mod and layer_ind < srnn._num_rec_cm_layers
    n_cm_per_rec = srnn._context_mod_num_ts if \
        srnn._context_mod_num_ts != -1 and \
        srnn._context_mod_separate_layers_per_ts else 1
    cm_idx = srnn._cm_rnn_start_ind + layer_ind * n_cm_per_rec

    seq_length, batch_size, n_hidden_prev = h.shape

    # If we want to apply context modulation in each time step, we need
    # to split the input sequence and call pytorch function at every
    # time step.
    outputs = []
    for t in range(seq_length):
        if srnn._context_mod_num_ts != -1 and \
                srnn._context_mod_separate_layers_per_ts:
            cm_idx += 1
        h_t = h[t,:,:]

        if cm_weights is not None and srnn._context_mod_num_ts != -1:
            curr_cm_weights = cm_weights[t]
        elif cm_weights is not None:
            assert len(cm_weights) == 1
            curr_cm_weights = cm_weights[0]
        else:
            curr_cm_weights = cm_weights

        # Compute the output.
        is_last_step = t==(seq_length-1)
        y_t = srnn.compute_basic_rnn_output(h_t, int_weights, use_cm,
            curr_cm_weights, cm_idx, ckpt_id, is_last_step)

        if srnn.bptt_depth != -1:
            if t < (seq_length - srnn.bptt_depth):
                # Detach hidden/output states, such that we don't backprop
                # through these timesteps.
                y_t = y_t.detach()

        outputs.append(y_t)

    return torch.stack(outputs)

def gram_schmidt_process(U, u_n):
    """Find a vector orthogonal to the current basis.

    Given a random vector :math:`u_n`, find a projection of this vector that is
    orthogonal to the existing base :math:`U`. For this we use the
    `Gram-Schmidt process <http://mlwiki.org/index.php/Gram-Schmidt_Process>`__.

    Args:
        U (torch.Tensor): The basis.
        u_n (torch.Tensor): The vector to be made orthogonal to the basis.

    Returns:
        (torch.Tensor): The transformed vector.
    """
    num_hidden, basis_size = U.shape
    u_n_ortho = u_n.clone()
    for i in range(basis_size):
        u_i = U[:,i].view(num_hidden, -1)
        coeff = torch.dot(u_i.t().view(-1), u_n.view(-1))/  \
                torch.dot(u_i.t().view(-1), u_i.view(-1))
        u_n_ortho += - coeff*u_i

    # Normalize the vector.
    u_n_ortho = u_n_ortho/torch.norm(u_n_ortho)

    # Sanity check.
    #for i in range(basis_size):
        # Some boundary is discussed here:
        # <https://math.stackexchange.com/questions/995623/why-are-randomly-
        # drawn-vectors-nearly-perpendicular-in-high-dimensions>
        #assert np.abs(torch.dot(U[:,i], u_n_ortho.view(-1)).item()) < 0.1

        # # Note that randomly sampled vectors have much larger dot products.
        # random1 = torch.randn(num_hidden)
        # random2 = torch.randn(num_hidden)
        # print(np.abs(torch.dot(random1,random2).item()))

    return u_n_ortho


def get_loss_vs_supervised_n_dim(mnet, hnet, task_loss_func, accuracy_func,
        dhandlers, config, device, stop_timestep=None, task_id=-1,
        lambda_ortho=1e3, criterion=97, lr=0.01, n_iter=100, batch_size=64,
        writer_dir=None):
    """Estimate the loss on all past trained tasks for a number of dimensions
    obtained using supervised linear dimensionality reduction.

    Given a set of hidden activations, this function iteratively learns an
    orthogonal basis for a lower-dimensional subspace such that the loss on the
    past tasks is minimized.

    If the timestep of the stop bit is provided, the analysis is done at this
    timestep only, else it is done on all timesteps as follows.
    i) For the stop bit analysis, we first project the hidden state of this
    timestep to a lower-dimensional space (increasingly bigger subspace) and
    then back to the original space. Then we use this reconstructed hidden
    activity to compute subsequent recurrent hidden state and make output
    projections. The superivison signal used is the accuracy on all subsequent
    timesteps, which in the case of the stop bit is the entirety of the output
    sequence where accuracy is computed.
    ii) For the analysis across all timesteps, we first compute the hidden
    states across all timesteps, then project the matrix containing all
    timesteps to a lower-dimensional space, reconstruct back, and use as
    supervision signal again the entirety of the output signal. Note however,
    that in this case the approximated hidden activations only affect the
    outputs, and not the recurrent processing, which might be a bit cheaty.

    The loss is computed as:

    .. math::

        loss = \lambda_{ortho} \mid \mid U U^T - I \mid \mid ^2 +
            \mathcal{L} (T, P)

    where ``U`` is the projection matrix, ``I`` is the identity matrix,
    and \mathcal{L} is the task-specific loss computed on the targets ``T`` and
    predictions ``P``. The latter term is computed separately for each task.

    Args:
        mnet: The main network.
        hnet: The hypernetwork (can be ``None``).
        task_loss_func: The function to compute the task loss.
        accuracy_func: The function to compute the task accuracy.
        dhandlers: The data handlers for the different tasks.
        config: The config.
        device: The pytorch device.
        stop_timestep (int, optional): The timestep of the stop bit. If
            ``None``, the analysis is performed by pulling all timesteps
            together. Else, it is just performed at the stop bit.
        task_id (int, optional): The last task that has been trained on.
        lambda_ortho (float, optional): The strength of the orthogonal
            regularization of the projection matrix.
        criterion (float, optional): Optionally a performance criterion can be
            provided such that, when it is reached, we consider that the
            reduced dimensional hidden state is enough to solve the task.
        lr (float, optional): The learning rate.
        n_iter (int, optional): The number of training iterations.
        batch_size (int, optional): The batch size.
        writer_dir (str, optional): If specified, a tensorboard writer is
            created that logs training details.

    Returns:
            (tuple): Tuple containing:

            - **loss_vs_dim** (list): The loss on the task for the given number 
              of dimensions. For example, the `i`-th element of the list 
              indicates the loss on the task when `i` dimensions obtained using 
              supervised linear dimensionality reduction are used.
            - **accu_vs_dim** (list): Same as `loss_vs_dim` but for accuracies.
    """
    if task_id == -1:
        task_id = len(dhandlers) - 1

    # Get final feedforward layers of the main network (after recurrent layer).
    if hnet is not None:
        raise NotImplementedError
    if config.use_masks:
        raise NotImplementedError

    writer = None
    if writer_dir is not None:
        summary_dir = 'dim_red_summary'
        if stop_timestep is not None:
            summary_dir += '_ts_%d' % stop_timestep
        summary_path = os.path.join(writer_dir, summary_dir)
        if os.path.exists(summary_path):
            shutil.rmtree(summary_path)
        writer = SummaryWriter(logdir=summary_path)

    # Iteratively compute the supervised components.
    num_hidden = int(config.rnn_arch)
    U = torch.tensor(())
    loss_vs_dim = []
    accu_vs_dim = []
    for n in range(num_hidden):

        # Initialize the vector for the new dimension. Note, we initialize as
        # a vector orthogonal to the existing `U` to make optimization easier
        # (one objective is already fulfilled).
        if n == 0:
            u_n = torch.zeros((num_hidden, 1))
            u_n[0] = 1
        else:
            # Gram-schmidt to obtain an orthogonal vector to `U`.
            u_n = torch.randn((num_hidden, 1))
            with torch.no_grad():
                u_n = gram_schmidt_process(U, u_n)
        u_n.requires_grad = True

        # Append new dimension to existing `U`.
        U_pre = U.clone().detach()
        U = torch.cat((U, u_n), 1)

        # Get the optimizer for the new dimension.
        optimizer = get_optimizer([u_n], lr=lr, use_adam=True)

        for i in range(n_iter):
            optimizer.zero_grad()

            # Get the task-specific loss for the different tasks.
            loss = 0
            for t in range(task_id+1):
                ### Get the batch data.
                dhandler = dhandlers[t]
                dhandler.reset_batch_generator()
                shared = argparse.Namespace()
                shared.feature_size = dhandler.in_shape[0]

                # Get a training batch.
                batch = dhandler.next_train_batch(batch_size, return_ids=True)
                sample_ids = batch[2]
                X = dhandler.input_to_torch_tensor(batch[0], device,
                    mode='train', sample_ids=sample_ids)
                X = stu.preprocess_inputs(config, shared, X, t)
                T = dhandler.output_to_torch_tensor(batch[1], device,
                    mode='train', sample_ids=sample_ids)
                T = stu.adjust_targets_to_head(config, dhandler, T, t,
                    trained_task_id=task_id, is_one_hot=True)

                ### Get the hidden states.
                _, hidden, hidden_int = mnet.forward(X, return_hidden=True,
                                                     return_hidden_int=True)
                # This analysis assumes there is only one recurrent layer.
                assert len(hidden_int) == 1
                # Note, `len(hidden)` can be bigger, in case there have been
                # fully-connected layer prior the recurrent layer.
                if mnet._use_lstm:
                    H = hidden[-1]
                else:
                    H = hidden_int[0]
                # H has dimensions [T, B, N], where T is the number of
                # timesteps, B is the batch size and N the number of neurons.

                if stop_timestep is not None:

                    ### Project the hidden state at stop bit using U.
                    h_stop = torch.matmul(H[stop_timestep,:,:], \
                        torch.mm(U, U.t()))

                    # Use this approximated hidden state at stop bit to
                    # compute the hidden state at all subsequent timesteps.
                    H_subs = compute_next_hidden_states(mnet, \
                        X[stop_timestep+1:, :, :], h_stop)

                    # Concatenate past hidden activations (original), the
                    # approximated stop timestep, and the subsequent timesteps
                    # computed from this approximated stop timestep.
                    # TODO fix inplace operation.
                    H_approx = torch.cat((H[:stop_timestep, :, :], \
                        torch.unsqueeze(h_stop, 0), H_subs), 0)

                else:
                    ### Project the hidden states using U.
                    H_approx = torch.matmul(H, torch.mm(U, U.t()))

                ### Get the outputs P.
                P = forward_post_rec(mnet, H_approx)
                allowed_outputs = stu.out_units_of_task(config, dhandler, t,
                    dhandlers=dhandlers, trained_task_id=task_id)
                P = P[:, :, allowed_outputs]

                ### Compute the task-specific loss.
                loss += task_loss_func(P, T, dhandler, None, None, sample_ids)

            # Compute the loss contribution related to orthogonality of U.
            orthogonal_loss = torch.norm(torch.mm(U.t(), U) - \
                torch.eye(U.shape[1]))**2
            # print(loss.item(), orthogonal_loss.item())
            loss += lambda_ortho * orthogonal_loss

            loss.backward()
            optimizer.step()

            # Overwrite the last column of U. Note, the `torch.cat()` operation
            # works with backprop, so whenever we use `U` in the code above,
            # gradients are backpropagated to `u_n` when calling `backward`.
            # However, the step function subsequently only updates `u_n` and not
            # `U`. So we need to manually write the new value of `u_n` into `U`.
            U = U.detach()
            U.requires_grad = False
            U[:, n] = u_n.view(-1)

            # Make sure that all previous dimensions didn't change.
            if n > 0:
                assert np.all(np.equal(U_pre, U[:,:-1].detach()).numpy())

            if writer is not None and i % 10 == 0 and n % 10 == 0:
                writer.add_scalar('train/dim_%d/total_loss' % n, loss, i)
                writer.add_scalar('train/dim_%d/orth_loss' % n, orthogonal_loss,
                                  i)

        ### Compute the loss and performance on the test set.
        with torch.no_grad():
            test_accs = []
            test_losses = []
            for t in range(task_id+1):
                dhandler = dhandlers[t]
                sample_ids = dhandler.get_test_ids()
                X = dhandler.input_to_torch_tensor( \
                    dhandler.get_test_inputs(), device, mode='inference',
                    sample_ids=sample_ids)
                X = stu.preprocess_inputs(config, shared, X, t)

                T = dhandler.output_to_torch_tensor( \
                    dhandler.get_test_outputs(), device, mode='inference',
                    sample_ids=sample_ids)
                T = stu.adjust_targets_to_head(config, dhandler, T, t,
                    dhandlers=dhandlers, trained_task_id=task_id,
                    is_one_hot=True)

                ### Get the hidden states.
                _, hidden, hidden_int = mnet.forward(X, return_hidden=True,
                        return_hidden_int=True)
                # This analysis assumes there is only one recurrent layer.
                assert len(hidden_int) == 1
                if mnet._use_lstm:
                    H = hidden[-1]
                else:
                    H = hidden_int[0]

                ### Project the hidden states using U.
                if stop_timestep is not None:
                    h_stop = torch.matmul(H[stop_timestep,:,:], \
                        torch.mm(U, U.t()))
                    H_subs = compute_next_hidden_states(mnet, \
                        X[stop_timestep+1:, :, :], h_stop)
                    H_approx = torch.cat((H[:stop_timestep, :, :], \
                        torch.unsqueeze(h_stop, 0), H_subs), 0)
                else:
                    H_approx = torch.matmul(H, torch.mm(U, U.t()))

                ### Get the outputs P.
                P = forward_post_rec(mnet, H_approx)
                allowed_outputs = stu.out_units_of_task(config, dhandler, t,
                    dhandlers=dhandlers, trained_task_id=task_id)
                P = P[:, :, allowed_outputs]

                ### Compute the task-specific loss.
                test_loss = task_loss_func(P, T, dhandler, None, None, \
                    sample_ids)
                test_losses.append(test_loss)
                accuracy, _ = accuracy_func(P, T, dhandler, sample_ids)
                test_accs.append(accuracy)

            orthogonal_loss = torch.norm(torch.mm(U.t(), U) - \
                torch.eye(U.shape[1]))**2

        if n == 0 or (n+1) % 10 == 0:
            print("Trained a projection " +
                  ("at the stopbit " if stop_timestep is not None else "") +
                  "with {0:3} dimensions: ".format(n+1) +\
                  "Total Loss: {0:8.2f} +- {1:4.2f}, ".\
                  format(np.mean(test_losses), np.std(test_losses)) +
                  "Orth-Loss: {0:3.2f}, Accu: {1:.2f} +- {2:.2f}".\
                  format(orthogonal_loss.cpu().item(), np.mean(test_accs),
                         np.std(test_accs)))

        if writer is not None:
            writer.add_scalar('test/total_loss', np.mean(test_losses), n)
            writer.add_scalar('test/orth_loss', orthogonal_loss, n)
            writer.add_scalar('test/acc', np.mean(test_accs), n)

        loss_vs_dim.append(np.mean(test_losses))
        accu_vs_dim.append(np.mean(test_accs))

        # Optionally we can define a stopping criterion such that ,whenever
        # the performance achieved on the current task with the given number
        # of reduced dimensions is high enough, we don't keep training further
        # dimensions.
        if criterion is not None and np.mean(test_accs) > criterion:
            break

    print("Finished training a projection " +
          ("at the stopbit " if stop_timestep is not None else "") +
          "with {0:3} dimensions: ".format(n+1) + \
          "Total Loss: {0:8.2f} +- {1:4.2f}, ".\
          format(np.mean(test_losses), np.std(test_losses)) +
          "Orth-Loss: {0:3.2f}, Accu: {1:.2f} +- {2:.2f}".\
          format(orthogonal_loss.cpu().item(), np.mean(test_accs),
                 np.std(test_accs)))

    if writer is not None:
        writer.close()

    return loss_vs_dim, accu_vs_dim
