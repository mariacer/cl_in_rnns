Multilingual Part-of-Speech Tagging experiments for continual learning
======================================================================

.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.
.. content-inclusion-marker-do-not-remove

In this subpackage we conduct experiments using the `Universal Dependencies <https://universaldependencies.org/>`__ dataset. The dataset provides, among other things, Part-of-Speech (PoS) tags for multiple languages. Therefore, it is ideally suited to learn a PoS tagger sequentially, one language after another.

Please run the following command to see the available options for running experiments.

.. code-block:: console

    $ python3 train_pos.py --help

Multitask
^^^^^^^^^

The following run on a multi-head BiLSTM leads to around 92.52% accuracy:

.. code-block:: console

    $ python3 train_pos.py --multi_head --num_tasks=20 --batch_size=64 --n_iter=5000 --lr=0.005 --clip_grad_norm=100 --rnn_arch="32" --use_bidirectional_net --use_cuda --multitask --orthogonal_hh_reg=-1 --dont_learn_wembs

Main network from scratch
^^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a single-head BiLSTM leads to around 95.04% during accuracy:

.. code-block:: console

    $ python3 train_pos.py --train_from_scratch --num_tasks=20 --batch_size=64 --n_iter=5000 --lr=0.005 --clip_grad_norm=100 --rnn_arch="32" --use_bidirectional_net --use_cuda --orthogonal_hh_reg=1 --dont_learn_wembs

Main network fine-tuning
^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a multi-head BiLSTM leads to around 91.64% during and 49.11% final accuracy:

.. code-block:: console

	$ python3 train_pos.py --beta=0.0 --si_lambda=0.0 --ewc_lambda=0.0 --ewc_gamma=0.0 --n_fisher=0 --tbptt_fisher=-1 --multi_head --num_tasks=20 --batch_size=64 --n_iter=2500 --lr=0.0005 --clip_grad_norm=100 --rnn_arch="32" --use_bidirectional_net --use_cuda --orthogonal_hh_reg=1 --dont_learn_wembs

Chunked Hypernetwork (HNET)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a multi-head BiLSTM leads to around 89.83% during and 89.30% final accuracy:

.. code-block:: console

    $ python3 train_pos.py --nh_chmlp_chunk_size=190 --beta=0.5 --multi_head --num_tasks=20 --batch_size=64 --n_iter=2500 --lr=0.005 --clip_grad_norm=-1 --rnn_arch="32" --use_bidirectional_net --nh_hnet_type=chunked_hmlp --nh_hmlp_arch="75,125" --nh_cond_emb_size=32 --nh_chunk_emb_size="8" --nh_hnet_net_act=sigmoid --use_new_hnet --use_cuda --hnet_all --orthogonal_hh_reg=-1 --dont_learn_wembs

Online Elastic Weight Consolidation (Online EWC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a multi-head BiLSTM leads to around 87.49% during and 86.89% final accuracy:

.. code-block:: console

    $ python3 train_pos.py --multi_head --num_tasks=20 --batch_size=64 --n_iter=2500 --lr=0.005 --clip_grad_norm=100 --rnn_arch="32" --use_bidirectional_net --use_cuda --orthogonal_hh_reg=1 --use_ewc --ewc_lambda=10.0 --n_fisher=200 --dont_learn_wembs

The following run on a multi-head BiLSTM **where the task identity is provided as additional input** leads to around 89.78% during and 89.67% final accuracy:

.. code-block:: console

    $ python3 train_pos.py --multi_head --num_tasks=20 --batch_size=64 --n_iter=5000 --lr=0.005 --clip_grad_norm=100 --rnn_arch="32" --use_bidirectional_net --use_cuda --input_task_identity --orthogonal_hh_reg=1 --use_ewc --ewc_lambda=10.0 --n_fisher=200 --dont_learn_wembs

Synaptic Intelligence (SI)
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a multi-head BiLSTM leads to around 85.96% during and 85.08% final accuracy:

.. code-block:: console

    $ python3 train_pos.py --beta=0.0 --ewc_lambda=0.0 --ewc_gamma=0.0 --n_fisher=0 --tbptt_fisher=-1 --multi_head --num_tasks=20 --batch_size=64 --n_iter=5000 --lr=0.001 --clip_grad_norm=100 --rnn_arch="32" --use_bidirectional_net --use_cuda --orthogonal_hh_reg=1 --use_si --si_lambda=0.1 --si_task_loss_only --dont_learn_wembs

Masking
^^^^^^^

The following run on a multi-head BiLSTM leads to around 91.36% during and 49.54% final accuracy:

.. code-block:: console

    $ python3 train_pos.py --no_context_mod_outputs --dont_softplus_gains --beta=0.0 --ewc_lambda=0.0 --ewc_gamma=0.0 --n_fisher=0 --tbptt_fisher=-1 --multi_head --num_tasks=20 --batch_size=64 --n_iter=2500 --lr=0.0005 --clip_grad_norm=-1 --rnn_arch="32" --use_bidirectional_net --use_cuda --use_masks --mask_fraction=0.2 --orthogonal_hh_reg=1 --dont_learn_wembs

Masking + Synpatic Intelligence (Masking + SI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a multi-head BiLSTM leads to around 82.74% during and 82.66% final accuracy:

.. code-block:: console

    $ python3 train_pos.py --no_context_mod_outputs --dont_softplus_gains --beta=0.0 --ewc_lambda=0.0 --ewc_gamma=0.0 --n_fisher=0 --tbptt_fisher=-1 --multi_head --num_tasks=20 --batch_size=64 --n_iter=5000 --lr=0.001 --clip_grad_norm=-1 --rnn_arch="32" --use_bidirectional_net --use_cuda --use_masks --mask_fraction=0.2 --orthogonal_hh_reg=1 --use_si --si_lambda=1.0 --si_task_loss_only --dont_learn_wembs

Coresets-100
^^^^^^^^^^^^

The following run on a multi-head BiLSTM leads to around 92% during and 90% final accuracy:

.. code-block:: console

    $ python3 train_pos.py --multi_head --num_tasks=20 --batch_size=64 --n_iter=5000 --lr=0.005 --clip_grad_norm=-1 --rnn_arch="32" --use_bidirectional_net --use_cuda --orthogonal_hh_reg=-1 --use_replay --replay_distill_reg=10 --coreset_size=100 --dont_learn_wembs

Coresets-500
^^^^^^^^^^^^

The following run on a multi-head BiLSTM leads to around 92% during and 90% final accuracy:

.. code-block:: console

    $ python3 train_pos.py --multi_head --num_tasks=20 --batch_size=64 --n_iter=2500 --lr=0.005 --clip_grad_norm=100 --rnn_arch="32" --use_bidirectional_net --use_cuda --orthogonal_hh_reg=-1 --use_replay --replay_distill_reg=10 --coreset_size=500 --dont_learn_wembs
