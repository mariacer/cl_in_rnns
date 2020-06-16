Split-Stroke MNIST experiments for continual learning
=====================================================

.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.
.. content-inclusion-marker-do-not-remove

In this subpackage we conduct experiments on the `Sequential Stroke MNIST <https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/wiki/MNIST-digits-stroke-sequence-data>`__ experiment. In analogy to the famous SplitMNIST experiment often conducted in feedforward Continual Learning experiments, we split the SMNIST dataset into 5 tasks, each containing only one pair of digits.

Please run the following command to see the available options for running Split SMNIST experiments.

.. code-block:: console

    $ python3 train_split_smnist.py --help

Experiments - CL1
-----------------

Continual learning experiments in this section are **CL1** only, i.e., the task identity is provided to the system at test time (cmp. `van de Ven et al. <https://arxiv.org/abs/1904.07734>`__).

Multitask
^^^^^^^^^

Multitask training refers to the joint training on data from all tasks.

The following run on a **multi-head 256 RNN** leads to around 99% accuracy:

.. code-block:: console

    $ python3 train_split_smnist.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=10000 --lr=0.001 --rnn_arch="256" --use_cuda --multitask


Replay via stored Coresets
^^^^^^^^^^^^^^^^^^^^^^^^^^

In this case, all heads are always trained using data from all tasks. However, only a small subset of the previous data will be available (the coreset).

The following run on a **multi-head 256 RNN with coresets of size 10** leads to around 97.5% accuracy:

.. code-block:: console

    $ python3 train_split_smnist.py --multi_head --num_tasks=5 --batch_size=64 --n_iter=5000 --lr=0.001 --rnn_arch="256" --use_cuda --use_replay --replay_distill_reg=10.0 --coreset_size=10

The following run on a **multi-head 256 RNN with coresets of size 50** leads to around 98.5% accuracy:

.. code-block:: console

    $ python3 train_split_smnist.py --multi_head --num_tasks=5 --batch_size=64 --n_iter=10000 --lr=0.0001 --rnn_arch="256" --use_cuda --use_replay --replay_distill_reg=10.0 --coreset_size=50

The following run on a **multi-head 256 RNN with coresets of size 100** leads to around 99% accuracy:

.. code-block:: console

    $ python3 train_split_smnist.py --multi_head --num_tasks=5 --batch_size=64 --n_iter=5000 --lr=0.0001 --rnn_arch="256" --use_cuda --use_replay --replay_distill_reg=10.0 --coreset_size=100

Generative Replay
^^^^^^^^^^^^^^^^^

In this case, multitask training is simulated by replaying fake input data from old tasks.

The following run on a **multi-head 256 RNN** leads to around 97.5% accuracy:

.. code-block:: console

    $ python3 train_split_smnist.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=5000 --lr=0.001 --rnn_arch="256" --use_cuda --use_replay --replay_pm_strength=1.0 --replay_rec_strength=10.0 --replay_distill_reg=10.0 --latent_dim=32 --dec_srnn_rec_layers="256" --dec_srnn_pre_fc_layers="" --dec_srnn_post_fc_layers=""

The following run on a **multi-head 256 RNN where the decoder has additional pre- and post-FC layer** leads to around 98% accuracy:

.. code-block:: console

    $ python3 train_split_smnist.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=10000 --lr=0.0001 --rnn_arch="256" --use_cuda --use_replay --replay_pm_strength=0.1 --replay_rec_strength=1.0 --replay_distill_reg=10.0 --latent_dim=16 --dec_srnn_rec_layers="256" --dec_srnn_pre_fc_layers="256" --dec_srnn_post_fc_layers="256"

Generative replay with hypernet-protected decoder (HNET+R)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this case, multitask training is simulated by replaying fake input data from old tasks with a task-specific decoder generated from a hypernetwork.

The following run on a **multi-head 256 RNN** leads to around 99% accuracy:

.. code-block:: console

    $ python3 train_split_smnist.py --beta=0.1 --multi_head --num_tasks=5 --batch_size=64 --n_iter=5000 --lr=0.001 --rnn_arch="256" --nh_hnet_type=chunked_hmlp --nh_hmlp_arch="50,50" --nh_cond_emb_size=32 --nh_chmlp_chunk_size=5000 --nh_chunk_emb_size=32 --use_new_hnet --std_normal_temb=1.0 --std_normal_emb=0.1 --use_cuda --hnet_all --use_replay --replay_pm_strength=0.1 --replay_rec_strength=0.1 --replay_distill_reg=10.0 --latent_dim=16 --dec_srnn_rec_layers="256" --dec_srnn_pre_fc_layers="" --dec_srnn_post_fc_layers=""

The following run on a **multi-head 256 RNN where the decoder has additional pre- and post-FC layer** leads to around 99% accuracy:

.. code-block:: console

    $ python3 train_split_smnist.py --beta=100.0 --multi_head --num_tasks=5 --batch_size=64 --n_iter=10000 --lr=0.001 --rnn_arch="256" --nh_hnet_type=chunked_hmlp --nh_hmlp_arch="50,50" --nh_cond_emb_size=32 --nh_chmlp_chunk_size=5000 --nh_chunk_emb_size=32 --use_new_hnet --std_normal_temb=1.0 --std_normal_emb=0.1 --use_cuda --hnet_all --use_replay --replay_pm_strength=0.1 --replay_rec_strength=1.0 --replay_distill_reg=10.0 --latent_dim=16 --dec_srnn_rec_layers="256" --dec_srnn_pre_fc_layers="256" --dec_srnn_post_fc_layers="256"