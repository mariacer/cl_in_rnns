Copy Task experiments for continual learning
============================================

.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.
.. content-inclusion-marker-do-not-remove

In this subpackage we conduct experiments on variants of the `Copy Task <https://arxiv.org/abs/1410.5401>`__ experiment.
Unless noted otherwise, all experiments are performed using vanilla RNNs with 256 hidden neurons.

Please run the following command to see the available options for running Copy Task experiments.

.. code-block:: console

    $ python3 train_copy.py --help


Permuted Copy Task
------------------

We consider a variant of the Copy Task where output patterns are permuted across time. We report results for input sequences of length ``p=i=5``.

Multitask
^^^^^^^^^

The following run on a multi-head RNN leads to around 100.00% accuracy:

.. code-block:: console

    $ python3 train_copy.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=20000 --lr=0.0005 --clip_grad_norm=1  --use_vanilla_rnn --use_cuda --multitask --orthogonal_hh_reg=1.0 --permute_time --input_len_step=0 --input_len_variability=0 


Main network from scratch
^^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a single-head RNN leads to around 100.00% during accuracy:

.. code-block:: console

    $ python3 train_copy.py  --train_from_scratch --num_tasks=5  --batch_size=128 --n_iter=20000 --lr=0.0005 --clip_grad_norm=-1 --use_vanilla_rnn --use_cuda --orthogonal_hh_reg=1.0 --permute_time --input_len_step=0 --input_len_variability=0 

Main network fine-tuning
^^^^^^^^^^^^^^^^^^^^^^^^

Using the recurrent main net only, and fine-tuning all weights for each task on a multi-head RNN (around 99.99% during accuracy):

.. code-block:: console

    $ python3 train_copy.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=20000 --clip_grad_norm=1 --use_vanilla_rnn --use_cuda --permute_time --input_len_step=0 --input_len_variability=0

Chunked Hypernetwork (HNET)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run using a multi-head RNN leads to around 100.00% during and 100.00% final accuracy:

.. code-block:: console

    $ python3 train_copy.py  --hyper_chunks=2000 --temb_size=32 --emb_size=64 --multi_head --num_tasks=5 --batch_size=128 --n_iter=20000 --lr=0.005 --clip_grad_norm=1 --use_vanilla_rnn --hnet_arch="64,64,64" --use_cuda --hnet_all --orthogonal_hh_init --orthogonal_hh_reg=1.0 --permute_time

If regularizing on a single randomly picked task at each loss evaluation, the following run obtains 100.00% final accuracy:

.. code-block:: console

    $ python3 train_copy.py --nh_chmlp_chunk_size=2500 --beta=1 --multi_head --num_tasks=5 --batch_size=128 --n_iter=25000 --lr=0.0005 --clip_grad_norm=1 --net_act=tanh --use_vanilla_rnn --nh_hnet_type=chunked_hmlp --nh_hmlp_arch="50,50" --nh_cond_emb_size=32 --nh_chunk_emb_size="32" --use_new_hnet --std_normal_temb=1.0 --std_normal_emb=0.1 --use_cuda --hnet_all --hnet_reg_batch_size=1 --orthogonal_hh_reg=1.0 --input_len_step=0 --input_len_variability=0 --permute_time


Online Elastic Weight Consolidation (Online EWC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a multi-head RNN leads to around 99.91% during and 99.30% final accuracy:

.. code-block:: console

    $ python3 train_copy.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=20000 --lr=0.001 --clip_grad_norm=1.0 --use_vanilla_rnn --use_cuda --random_seed=2 --use_ewc --ewc_gamma=1.0 --ewc_lambda=100.0 --n_fisher=-1  --orthogonal_hh_reg=0.01 --permute_time

Synaptic Intelligence (SI)
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a multi-head RNN leads to around 98.7% during and 94.5 final accuracy:

.. code-block:: console

    $ python3 train_copy.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=20000 --clip_grad_norm=1 --use_vanilla_rnn --use_cuda --orthogonal_hh_reg=1.0 --use_si --si_lambda=0.01 --si_task_loss_only --permute_time --input_len_step=0 --input_len_variability=0

Masking
^^^^^^^

The following run on a multi-head RNN leads to around 99.93% during and 73.73% final accuracy:

.. code-block:: console

    $ python3 train_copy.py --no_context_mod_outputs --dont_softplus_gains --multi_head --num_tasks=5 --batch_size=128 --n_iter=20000 --lr=0.005 --clip_grad_norm=100  --use_vanilla_rnn --orthogonal_hh_init --orthogonal_hh_reg=-1 --use_cuda --use_masks --permute_time --input_len_step=0 --input_len_variability=0 


Masking + Synpatic Intelligence (Masking + SI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a multi-head RNN leads to around 100.00% during and 100.00% final accuracy:

.. code-block:: console

    $ python3 train_copy.py --no_context_mod_outputs --dont_softplus_gains --multi_head --num_tasks=5 --batch_size=128 --n_iter=20000 --lr=0.005 --clip_grad_norm=100  --use_vanilla_rnn --orthogonal_hh_init --orthogonal_hh_reg=-1 --use_cuda --use_masks --use_si --si_task_loss_only --permute_time --input_len_step=0 --input_len_variability=0

Generative Replay 
^^^^^^^^^^^^^^^^^

The following run on a multi-head RNN leads to around 100.00% during and 100.00% final accuracy:

.. code-block:: console

    $ python3 train_copy.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=20000 --lr=0.0001 --clip_grad_norm=100 --rnn_arch="256" --use_vanilla_rnn --use_cuda --use_replay --orthogonal_hh_init --orthogonal_hh_reg=1.0 --replay_pm_strength=1.0 --replay_rec_strength=10.0 --replay_distill_reg=1.0 --latent_dim=8 --dec_srnn_rec_layers="256" --dec_srnn_rec_type=elman --permute_time --input_len_step=0 --input_len_variability=0

Coresets-100
^^^^^^^^^^^^

The following run on a multi-head RNN with Coresets of size 100 leads to around 100% final accuracy:

.. code-block:: console

    $ python3 train_copy.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=10000 --lr=0.0001 --clip_grad_norm=-1 --use_vanilla_rnn --use_cuda --use_replay --orthogonal_hh_init --orthogonal_hh_reg=10.0 --replay_distill_reg=10.0 --coreset_size=100 --permute_time --input_len_step=0 --input_len_variability=0


Padded Copy Task
----------------

We consider a variant of the Copy Task where input patterns are padded with zeros, yielding longer input sequences. We report results for input sequences of length ``i=25`` and pattern output sequences of length ``p=5``.

Chunked Hypernetwork (HNET)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a multi-head RNN leads to around 100% final accuracy:

.. code-block:: console

    $ python3 train_copy.py --nh_chmlp_chunk_size=4000 --beta=10.0 --multi_head --num_tasks=5 --batch_size=128 --n_iter=10000 --lr=0.001 --clip_grad_norm=10 --net_act=tanh --use_vanilla_rnn --nh_hnet_type=chunked_hmlp --nh_hmlp_arch="60,60,30" --nh_cond_emb_size=16 --nh_chunk_emb_size="32" --use_new_hnet --std_normal_temb=0.1 --std_normal_emb=0.1 --use_cuda --hnet_all --orthogonal_hh_reg=10.0 --first_task_input_len=25 --input_len_step=0 --input_len_variability=0 --pat_len=5

Online Elastic Weight Consolidation (Online EWC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a multi-head RNN leads to around 98.03% during and 98.07% final accuracy:

.. code-block:: console

    $ python3 train_copy.py --multi_head --num_tasks=5 --first_task_input_len=25 --pat_len=5 --batch_size=128 --n_iter=20000 --lr=0.005 --clip_grad_norm=1  --use_vanilla_rnn --use_cuda --orthogonal_hh_init --orthogonal_hh_reg=1.0 --use_ewc --ewc_lambda=10000.0 --n_fisher=200 --permute_time --input_len_step=0 --input_len_variability=0

Pattern Manipulation Task
-------------------------

We consider a variant of the Copy Task where the output is computed from the input pattern by applying a binary XOR operation iteratively with a series of ``r`` fixed permutations.

Chunked Hypernetwork (HNET)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a multi-head RNN for ``r=1`` leads to around **100.00** % during and **100.00** % final accuracy:

.. code-block:: console
    
    $ python3 train_copy.py --hyper_chunks=4000 --beta=1.0 --multi_head --num_tasks=5 --batch_size=128 --n_iter=20000 --lr=0.005 --clip_grad_norm=1 --use_vanilla_rnn --hnet_arch="64,64,32" --temb_size=32 --emb_size=32 --use_cuda --data_random_seed=12 --hnet_all --orthogonal_hh_reg=1.0 --permute_time --input_len_step=0 --input_len_variability=0 --permute_xor --permute_xor_iter=1 --permute_xor_separate

The following run on a multi-head RNN for ``r=5`` leads to around **97.07** % during and **93.93** % final accuracy:

.. code-block:: console
    
    $ python3 train_copy.py --hyper_chunks=4000 --beta=10.0 --multi_head --num_tasks=5 --batch_size=128 --n_iter=20000 --lr=0.005 --clip_grad_norm=1 --use_vanilla_rnn --hnet_arch="64,64,32" --temb_size=32 --emb_size=32 --use_cuda --data_random_seed=12 --hnet_all --orthogonal_hh_reg=1.0 --permute_time --input_len_step=0 --input_len_variability=0 --permute_xor --permute_xor_iter=5 --permute_xor_separate

Online Elastic Weight Consolidation (Online EWC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run on a multi-head RNN for ``r=1`` leads to around **99.65** % during and **95.92** % final accuracy:

.. code-block:: console

    $ python3 train_copy.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=20000 --lr=0.005 --clip_grad_norm=1 --use_vanilla_rnn --use_cuda --data_random_seed=12 --orthogonal_hh_init --orthogonal_hh_reg=10 --use_ewc --ewc_lambda=1000.0 --n_fisher=200 --permute_time --input_len_step=0 --input_len_variability=0 --permute_xor --permute_xor_separate --permute_xor_iter=1

The following run on a multi-head RNN for ``r=5`` leads to around **94.41** % during and **86.39** % final accuracy:

.. code-block:: console

    $ python3 train_copy.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=20000 --lr=0.001 --clip_grad_norm=-1 --use_vanilla_rnn --use_cuda --data_random_seed=12 --orthogonal_hh_init --orthogonal_hh_reg=10 --use_ewc --ewc_lambda=1000.0 --n_fisher=200 --permute_time --input_len_step=0 --input_len_variability=0 --permute_xor --permute_xor_separate --permute_xor_iter=5