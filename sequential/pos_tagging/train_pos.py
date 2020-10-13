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
# @title          :sequential/pos_tagging/train_pos.py
# @author         :ch, mc, be
# @contact        :henningc@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# @python_version :3.6.10
"""
Training script for PoS tagging experiments
-------------------------------------------

See the help text for more usage information:

.. code-block::

    python3 train_pos.py --help
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np

import sequential.embedding_utils as eu
import sequential.train_utils_sequential as stu
import sequential.train_sequential as sts
from sequential.pos_tagging import train_args_pos
import sequential.pos_tagging.train_utils_pos as ctu
import utils.sim_utils as sutils

import sequential.pos_tagging.hpconfig_pos as hpsearch_cl
import sequential.pos_tagging.hpconfig_pos_multitask as hpsearch_mt

def run():
    """Run the script"""
    #############
    ### Setup ###
    #############

    config = train_args_pos.parse_cmd_arguments()
    device, writer, logger = sutils.setup_environment(config)
    dhandlers = ctu.generate_tasks(config, logger, writer=writer)

    # Load preprocessed word embeddings, see
    # :mod:`data.timeseries.preprocess_mud` for details.
    wembs_path = '../../datasets/sequential/mud/embeddings.pickle'
    wemb_lookups = eu.generate_emb_lookups(config, filename=wembs_path,
                                           device=device)
    assert len(wemb_lookups) == config.num_tasks

    # We will use the namespace below to share miscellaneous information between
    # functions.
    shared = Namespace()
    # The embedding size is fixed due to the use of pretrained polyglot
    # embeddings.
    # FIXME Could be made configurable in the future in case we don't initialize
    # embeddings via polyglot.
    shared.feature_size = 64
    shared.word_emb_lookups = wemb_lookups

    target_net, hnet, dnet = stu.generate_networks(config, shared, dhandlers,
                                                   device)

    # generate masks if needed
    ctx_masks = None
    if config.use_masks:
        ctx_masks = stu.generate_binary_masks(config, device, target_net)

    # We store the target network weights (excluding potential context-mod
    # weights after every task). In this way, we can quantify changes and
    # observe the "stiffness" of EWC.
    shared.tnet_weights = []
    # We store the context-mod weights (or all weights) coming from the hypernet
    # after every task, in order to quantify "forgetting". Note, the hnet
    # regularizer should keep them fix.
    shared.hnet_out = []

    # Get the task-specific functions for loss and accuracy.
    task_loss_func = ctu.get_loss_func(config, device, logger, ewc_loss=False)
    accuracy_func = ctu.get_accuracy_func(config)
    ewc_loss_func = ctu.get_loss_func(config, device, logger, \
        ewc_loss=True) if config.use_ewc else None

    replay_fcts = None
    if config.use_replay:
        replay_fcts = dict()
        replay_fcts['rec_loss'] = ctu.get_vae_rec_loss_func()
        replay_fcts['distill_loss'] = ctu.get_distill_loss_func()
        replay_fcts['soft_trgt_acc'] = ctu.get_soft_trgt_acc_func()

    if config.multitask:
        summary_keywords=hpsearch_mt._SUMMARY_KEYWORDS
        summary_filename=hpsearch_mt._SUMMARY_FILENAME
    else:
        summary_keywords=hpsearch_cl._SUMMARY_KEYWORDS
        summary_filename=hpsearch_cl._SUMMARY_FILENAME

    ########################
    ### Train classifier ###
    ########################

    shared.f_scores = None

    # Train the network task by task. Testing on all tasks is run after 
    # finishing training on each task.
    ret, train_loss, test_loss, test_acc = sts.train_tasks(dhandlers,
        target_net, hnet, dnet, device, config, shared, logger, writer,
        ctx_masks, summary_keywords, summary_filename,
        task_loss_func=task_loss_func, accuracy_func=accuracy_func,
        ewc_loss_func=ewc_loss_func, replay_fcts=replay_fcts)

    stu.log_results(test_acc, config, logger)

    writer.close()

    if ret == -1:
        logger.info('Program finished successfully.')

        if config.show_plots:
            plt.show()
    else:
        logger.error('Only %d tasks have completed training.' % (ret+1))

if __name__ == '__main__':
    run()