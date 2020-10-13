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
# @title          :sequential/student_teacher/hidden_dim_ideal_student.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :07/27/2020
# @version        :1.0
# @python_version :3.6.10
"""
Checking hidden dimensionality in ideal student network
-------------------------------------------------------

The ideal student network solves all tasks (stemming from different teachers)
to perfection (see method
:meth:`data.timeseries.rnd_rec_teacher.RndRecTeacher.construct_ideal_student`
for details).

In this script we verify how the dimensionality of the hidden state of an RNN
behaves when the number of tasks, which it can perfectly solve, increases.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.timeseries.rnd_rec_teacher import RndRecTeacher
from mnets.simple_rnn import SimpleRNN
import sequential.ht_analyses.state_space_analysis as ssa
import sequential.student_teacher.train_args_st as sta
import sequential.student_teacher.train_utils_st as stu
from sequential.ht_analyses import pca_utils
import utils.ewc_regularizer as ewc
import utils.sim_utils as sutils

if __name__ == '__main__':
    config = sta.parse_cmd_arguments(mode='student_teacher')
    device, writer, logger = sutils.setup_environment(config)
    writer.close()

    num_samples = 1000
    ### Construct datasets ###
    scenario = 2
    if scenario == 0: # All tasks identical -> hidden dim should remain constant
        d1 = RndRecTeacher(num_test=num_samples, rseed=1)
        d2 = RndRecTeacher(num_test=num_samples, rseed=1)
        d3 = RndRecTeacher(num_test=num_samples, rseed=1)
        dhandlers = [d1, d2, d3]
    elif scenario == 1: # All tasks different -> hidden dim should increase
        d1 = RndRecTeacher(num_test=num_samples, rseed=1)
        d2 = RndRecTeacher(num_test=num_samples, rseed=2)
        d3 = RndRecTeacher(num_test=num_samples, rseed=3)
        dhandlers = [d1, d2, d3]
    else: # Slightly more complex scenario.
        dhandlers = stu.generate_tasks(config, logger)


    ### Construct student RNN ###
    n_out = dhandlers[0].out_shape[0]
    n_out_tot = 0
    for dh in dhandlers:
        n_out_tot += dh.out_shape[0]
        assert dh.out_shape[0] == n_out
    net = SimpleRNN(n_in=dhandlers[0].in_shape[0], rnn_layers=[256],
                    fc_layers=[n_out_tot])

    _, axes = plt.subplots(3, 1, figsize=(7, 3*2.5))

    ### Compute hidden dim for increasing number of tasks ###
    diag_hh_fisher_means = []
    diag_ho_fisher_means = []
    for n_learned in range(1, len(dhandlers)+1):
        print('Constructing ideal student for %d tasks.' % n_learned)
        RndRecTeacher.construct_ideal_student(net, dhandlers[:n_learned])
        # Compute fisher elements.
        curr_tid = n_learned-1
        n_start = curr_tid * n_out
        print('Computing fisher ...')
        ewc.compute_fisher(curr_tid, dhandlers[curr_tid],
            net.internal_params, device, net, empirical_fisher=True,
            online=True, gamma=config.ewc_gamma, n_max=config.n_fisher,
            regression=False, allowed_outputs=range(n_start, n_start+n_out),
            custom_forward=None, time_series=True,
            custom_nll=stu.get_loss_func(config, device, logger, ewc_loss=True),
            pass_ids=True)
        diag_hh_fisher = ssa.get_fisher_estimates(net, connection_type='hh')
        diag_ho_fisher = ssa.get_fisher_estimates(net, connection_type='ho')
        diag_hh_fisher_means.append(np.mean(diag_hh_fisher))
        diag_ho_fisher_means.append(np.mean(diag_ho_fisher))

        # Compute hidden states.
        hidden_states_ht = []
        hidden_states_yt = []
        n_start = 0
        for ii in range(n_learned):
            dh = dhandlers[ii]
            n_out = dh.out_shape[0]

            X = dh.input_to_torch_tensor( \
                dh.get_test_inputs(), 'cpu', mode='inference')
            T = dh.output_to_torch_tensor( \
                dh.get_test_outputs(), 'cpu', mode='inference')

            with torch.no_grad():
                Y_logits, hidden, hidden_int = net.forward(X,
                    return_hidden=True, return_hidden_int=True)
                Y_logits = Y_logits[:, :, n_start:n_start+n_out]

            hidden_states_ht.append(hidden_int[0].numpy())
            hidden_states_yt.append(hidden[0].numpy())

            n_start += n_out

            print('Test MSE of task %d when solving %d tasks: %e.' \
                  % (ii+1, n_learned, torch.sum((T - Y_logits)**2)))

        hidden_states_ht = np.concatenate(hidden_states_ht, axis=1)
        hidden_states_yt = np.concatenate(hidden_states_yt, axis=1)

        # Compute hidden dimensionality for each timestep.
        n_ts = hidden_states_ht.shape[0]
        ht_dims = []
        yt_dims = []
        for t in range(n_ts):
            h_t = hidden_states_ht[t,:,:]
            y_t = hidden_states_yt[t,:,:]

            ev_ht, _ = pca_utils.get_expl_var(h_t, do_kernel_pca=False,
                                              n_samples=num_samples)
            ev_yt, _ = pca_utils.get_expl_var(y_t, do_kernel_pca=False,
                                              n_samples=num_samples)

            dim_ht = pca_utils.get_num_dimensions(ev_ht)
            dim_yt = pca_utils.get_num_dimensions(ev_yt)
            ht_dims.append(dim_ht)
            yt_dims.append(dim_yt)

        print('Hidden dimensionality of h_t per timestep: %s' % ht_dims)
        print('Hidden dimensionality of y_t per timestep: %s' % yt_dims)

        axes[0].plot(range(len(ht_dims)), ht_dims, label='task %d' % curr_tid)
        axes[1].plot(range(len(yt_dims)), yt_dims, label='task %d' % curr_tid)

    print('Mean fisher value after adding each task for hh: %s' \
          % diag_hh_fisher_means)
    print('Mean fisher value after adding each task for ho: %s' \
          % diag_ho_fisher_means)
    axes[2].plot(range(len(dhandlers)), diag_hh_fisher_means, label='fisher hh')
    axes[2].plot(range(len(dhandlers)), diag_ho_fisher_means, label='fisher ho')

    axes[0].set_ylabel('num dimensions h_t')
    axes[0].set_xticks(range(len(ht_dims)))
    axes[1].set_xlabel('timesteps')
    axes[1].set_ylabel('num dimensions y_t')
    axes[1].set_xticks(range(len(yt_dims)))
    axes[2].set_xlabel('tasks learned')
    axes[2].set_ylabel('fisher val')
    axes[2].set_xticks(range(len(dhandlers)))

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()

    plt.show()