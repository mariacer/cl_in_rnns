Split Audioset Experiments for Continual Learning
=================================================

.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.
.. content-inclusion-marker-do-not-remove

In this subpackage we conduct experiments on the `Audioset <https://research.google.com/audioset/>`__ experiment. In analogy to the famous SplitMNIST experiment often conducted in feedforward Continual Learning experiments, we split the Audioset dataset into several tasks, each containing a certain amount of classes.
The following results all correspond to 10 tasks each containing 10 different classes.

Please run the following command to see the available options for running Split Audioset experiments.

.. code-block:: console

    $ python3 train_split_audioset.py --help

Experiments - CL1
-----------------

Multitask
^^^^^^^^^

The following run on a **multi-head 32 RNN** leads to around 77% final accuracy:

.. code-block:: console

    $ python3 train_split_audioset.py --multi_head --num_tasks=10 --num_classes_per_task=10 --batch_size=64 --n_iter=50000 --lr=0.0001 --clip_grad_norm=1 --rnn_arch="32" --net_act=tanh --use_cuda --multitask --orthogonal_hh_reg=-1

Main network from scratch
^^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a **single-head 32 RNN** leads to around 79% during/final accuracy:

.. code-block:: console

    $ python3 train_split_audioset.py --si_lambda=0.0 --ewc_lambda=0.0 --ewc_gamma=0.0 --n_fisher=0 --tbptt_fisher=-1 --train_from_scratch --num_tasks=10 --num_classes_per_task=10 --batch_size=128 --n_iter=25000 --lr=0.001 --clip_grad_norm=1 --rnn_arch="32" --net_act=tanh --use_cuda

Main network fine-tuning
^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a **multi-head 32 RNN** leads to around 72% during and 51% final accuracy:

.. code-block:: console

    $ python3 train_split_audioset.py --multi_head --num_tasks=10 --num_classes_per_task=10 --batch_size=64 --n_iter=50000 --lr=1e-05 --clip_grad_norm=-1 --rnn_arch="32" --net_act=tanh --use_cuda --orthogonal_hh_reg=0.1


**Note**, the following results have been selected by using the runs with **best during** (**not final**) accuracy from the hp-search.

The following run on a **multi-head 256 RNN** leads to around 81% during and 69% final accuracy:

.. code-block:: console

    $ python3 train_split_audioset.py --multi_head --num_tasks=10 --num_classes_per_task=10 --batch_size=64 --n_iter=10000 --lr=0.0001 --clip_grad_norm=1 --rnn_arch="256" --net_act=tanh --use_cuda

The following run on a **multi-head 128 RNN** leads to around 81% during and 67% final accuracy:

.. code-block:: console

    $ python3 train_split_audioset.py --multi_head --num_tasks=10 --num_classes_per_task=10 --batch_size=64 --n_iter=10000 --lr=0.0001 --clip_grad_norm=-1 --rnn_arch="128" --net_act=tanh --use_cuda

The following run on a **multi-head 64 RNN** leads to around 80% during and 53% final accuracy:

.. code-block:: console

    $ python3 train_split_audioset.py --multi_head --num_tasks=10 --num_classes_per_task=10 --batch_size=64 --n_iter=25000 --lr=0.0001 --clip_grad_norm=-1 --rnn_arch="64" --net_act=tanh --use_cuda

The following run on a **multi-head 32 RNN** leads to around 79% during and 41% final accuracy:

.. code-block:: console

    $ python3 train_split_audioset.py --multi_head --num_tasks=10 --num_classes_per_task=10 --batch_size=64 --n_iter=25000 --lr=0.0001 --clip_grad_norm=1 --rnn_arch="32" --net_act=tanh --use_cuda

The following run on a **multi-head 16 RNN** leads to around 77% during and 32% final accuracy:

.. code-block:: console

    $ python3 train_split_audioset.py --multi_head --num_tasks=10 --num_classes_per_task=10 --batch_size=64 --n_iter=25000 --lr=0.0001 --clip_grad_norm=-1 --rnn_arch="16" --net_act=tanh --use_cuda

The following run on a **multi-head 8 RNN** leads to around 73% during and 29% final accuracy:

.. code-block:: console

    $ python3 train_split_audioset.py --multi_head --num_tasks=10 --num_classes_per_task=10 --batch_size=64 --n_iter=25000 --lr=0.0001 --clip_grad_norm=-1 --rnn_arch="8" --net_act=tanh --use_cuda --use_best_models

Using orthogonal regularization & initialization seems to lead to a slight increase in during accuracy.

The following run on a **multi-head 32 RNN** leads to close to 80% during and 43% final accuracy:

.. code-block:: console

    $ python3 train_split_audioset.py --multi_head --num_tasks=10 --num_classes_per_task=10 --batch_size=64 --n_iter=25000 --lr=0.0001 --clip_grad_norm=1 --rnn_arch="32" --net_act=tanh --use_cuda --orthogonal_hh_init --orthogonal_hh_reg=1.0

Chunked Hypernetwork (HNET)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a **multi-head 32 RNN** leads to around 72.65% final accuracy (compression ratio: 0.98):

.. code-block:: console

    $ python3 train_split_audioset.py --nh_chmlp_chunk_size=2000 --beta=1.0 --multi_head --num_tasks=10 --num_classes_per_task=10 --batch_size=128 --n_iter=15000 --lr=0.0001 --weight_decay=0.01 --clip_grad_norm=-1 --rnn_arch="32" --srnn_pre_fc_layers="" --net_act=tanh --nh_hnet_type=chunked_hmlp --nh_hmlp_arch="10,10" --nh_cond_emb_size=32 --nh_chunk_emb_size=32 --nh_hnet_dropout_rate=-1 --use_new_hnet --std_normal_temb=1.0 --std_normal_emb=0.1 --use_cuda --hnet_all --during_acc_criterion=60

Online Elastic Weight Consolidation (Online EWC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a **multi-head 32 RNN** leads to around 67.5% final accuracy:

.. code-block:: console

    $ python3 train_split_audioset.py --multi_head --num_tasks=10 --num_classes_per_task=10 --batch_size=128 --n_iter=25000 --lr=0.0001 --clip_grad_norm=-1 --rnn_arch="32" --net_act=tanh --use_cuda --use_ewc --ewc_lambda=100.0 --n_fisher=-1

The following run on a **multi-head 32 RNN where the task identity is provided as additional input** leads to around 71.74% during and 66.35% final accuracy:

.. code-block:: console

    $ python3 train_split_audioset.py --multi_head --num_tasks=10 --num_classes_per_task=10 --batch_size=64 --n_iter=25000 --lr=0.0001 --clip_grad_norm=1 --rnn_arch="32" --net_act=tanh --use_cuda --input_task_identity --use_ewc --ewc_gamma=1.0 --ewc_lambda=10.0 --n_fisher=-1

Synaptic Intelligence (SI)
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a **multi-head 32 RNN** leads to around 67% final accuracy:

.. code-block:: console

    $ python3 train_split_audioset.py --multi_head --num_tasks=10 --num_classes_per_task=10 --batch_size=64 --n_iter=25000 --lr=0.0001 --clip_grad_norm=-1 --rnn_arch="32" --net_act=tanh --use_cuda --use_si --si_lambda=1.0 --si_task_loss_only

Masking
^^^^^^^

The following run on a **multi-head 32 RNN** leads to around 55% final accuracy:

.. code-block:: console

    $ python3 train_split_audioset.py --no_context_mod_outputs --dont_softplus_gains --si_lambda=0.0 --multi_head --num_tasks=10 --num_classes_per_task=10 --batch_size=64 --n_iter=25000 --lr=0.0001 --clip_grad_norm=-1 --rnn_arch="32" --net_act=tanh --use_cuda --use_masks --mask_fraction=0.6 --during_acc_criterion=25

Masking + Synpatic Intelligence (Masking + SI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a **multi-head 32 RNN using SI** leads to around 65% final accuracy:

.. code-block:: console

    $ python3 train_split_audioset.py --no_context_mod_outputs --dont_softplus_gains --multi_head --num_tasks=10 --num_classes_per_task=10 --batch_size=64 --n_iter=25000 --lr=0.0001 --clip_grad_norm=1 --rnn_arch="32" --net_act=tanh --use_cuda --use_masks --mask_fraction=0.4 --during_acc_criterion=25 --use_si --si_lambda=100.0 --si_task_loss_only

Coresets
^^^^^^^^

The following run on a **multi-head 32 RNN with coreset-size of 100 per task** leads to around 72.5% final accuracy:

.. code-block:: console

    $ python3 train_split_audioset.py --multi_head --num_tasks=10 --num_classes_per_task=10 --batch_size=64 --n_iter=25000 --lr=0.0001 --clip_grad_norm=1 --rnn_arch="32" --net_act=tanh --use_cuda --use_best_models --use_replay --replay_distill_reg=10.0 --coreset_size=100


The following run on a **multi-head 32 RNN with coreset-size of 500 per task** leads to around 73.5% final accuracy:

.. code-block:: console

    $ python3 train_split_audioset.py --multi_head --num_tasks=10 --num_classes_per_task=10 --batch_size=64 --n_iter=10000 --lr=0.0001 --clip_grad_norm=-1 --rnn_arch="32" --net_act=tanh --use_cuda --use_best_models --use_replay --replay_distill_reg=0.1 --coreset_size=500
