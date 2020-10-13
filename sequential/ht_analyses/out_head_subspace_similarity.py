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
# @title          :sequential/ht_analyses/out_head_subspace_similarity.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :08/02/2020
# @version        :1.0
# @python_version :3.6.10
r"""
Similarity of subspaces selected by output heads
------------------------------------------------

In this script we investigate, whether the subspaces chosen by (linear) output
heads are distinct for different tasks. Consider the output head of task
:math:`k`: :math:`y_t^{(k)} = W^{(k)} a_t + b^{(k)}`, where :math:`a_t` denotes
a shared hidden activation within the RNN that is read out by all output heads.

We consider :math:`a_t \in \mathbb{R}^m` and :math:`y_t^{(k)} \in \mathbb{R}^n`
and we assume that :math:`m > n`. To obtain an orthonormal basis in
:math:`\mathbb{R}^m` for the subspace selected by matrix :math:`W^{(k)}`, we
perform SVD for each output head matrix:

.. math::

    W^{(k)} = U^{(k)} S^{(k)} V^{(k)}

where :math:`S^{(k)}` has at most :math:`n` non-zero entries. The corresponding
rows in the unitary matrix :math:`V^{(k)}` describe the desired basis, denoted
by :math:`\hat{V}^{(k)} \in \mathbb{R}^{n \times m}`. Note, we might neglect
very small singular values such that we can have an effective basis
:math:`\hat{V}^{(k)}` with less than :math:`n` rows.

We determine the subspace similarity between two tasks :math:`k, k'` using

.. math::

    \text{sim}(k, k') = \lVert \hat{V}^{(k)} (\hat{V}^{(k')})^T \rVert_F

"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tempfile

import sequential.copy.train_utils_copy as copytu
import sequential.ht_analyses.state_space_analysis as ssa
import sequential.train_args_sequential as sta
import sequential.train_utils_sequential as stu
import utils.sim_utils as sutils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= \
        'Studying output head subspace similarity.')
    parser.add_argument('out_dir', type=str,
                        help='The output directory of the simulation to be ' +
                             'analyzed.')
    args = parser.parse_args()
    out_dir = args.out_dir

    # Temporary simulation directory, required by method `setup_environment`.
    args.out_dir = os.path.join(tempfile.gettempdir(),
        'tmp_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    args.loglevel_info = False
    args.random_seed = 42 # Note, this script doesn't perform random computation
    args.deterministic_run = False
    args.no_cuda = True
    device, writer, logger = sutils.setup_environment(args)

    # FIXME Code below copied from script `state_space_analysis`.
    # Load the config
    if not os.path.exists(out_dir):
        raise ValueError('The directory "%s" does not exist.'% out_dir)
    with open(os.path.join(out_dir, "config.pickle"), "rb") as f:
        config = pickle.load(f)
    # Overwrite the directory it it's not the same as the original.
    if config.out_dir != out_dir:
        config.out_dir = out_dir
    # Check for old command line arguments and make compatible with new version.
    config = sta.update_cli_args(config)

    # FIXME only for copy task!
    generate_tasks_func = copytu.generate_copy_tasks
    dhandlers = generate_tasks_func(config, logger, writer=writer)
    mnet, hnet, _ = stu.generate_networks(config, dhandlers, device)
    ssa.load_models(out_dir, device, logger, mnet, hnet=hnet,
                    task_id=config.num_tasks-1)
    writer.close()

    # FIXME no hnet support yet.
    assert len(mnet.param_shapes) == len(mnet.internal_params)

    V_per_task = []

    i_start = 0
    for tid in range(config.num_tasks):
        i_end = i_start + dhandlers[tid].out_shape[0]
        omask = mnet.get_output_weight_mask(out_inds=range(i_start, i_end))

        W_tid = None
        for ii, mm in enumerate(omask):
            if mm is None:
                continue
            elif mnet.param_shapes_meta[ii]['name'] == 'weight':
                assert W_tid is None
                pid = mnet.param_shapes_meta[ii]['index']
                W_tid = mnet.internal_params[pid][mm]
                W_tid = W_tid.view(-1, mm.shape[1])
            else:
                assert mnet.param_shapes_meta[ii]['name'] == 'bias'

        _, S, V = np.linalg.svd(W_tid.detach().numpy(), full_matrices=0)
        print('Singular values task %d: %s' % (tid, S))
        V_per_task.append(V)

        i_start = i_end

    sims = np.zeros((config.num_tasks, config.num_tasks))
    #for t1 in range(1, config.num_tasks):
        #for t2 in range(t1):
    for t1 in range(config.num_tasks):
        for t2 in range(t1+1):
            sim = np.linalg.norm(V_per_task[t1] @ V_per_task[t2].T, ord='fro')
            sims[t1, t2] = sim
            #print('Subspace similarity betweem tasks %d and %d: %f' % \
            #      (t2, t1, sim))

    fig = plt.figure()
    plt.imshow(sims)
    plt.colorbar()
    plt.show()
