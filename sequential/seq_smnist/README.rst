Split Sequential Stroke-MNIST experiments for continual learning
================================================================

.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.
.. content-inclusion-marker-do-not-remove

In this subpackage we conduct experiments on Split Sequential Stroke MNIST. The sequence length **m** determines the difficulty of individual tasks.
Please run the following command to see the available options for running Split SMNIST experiments.

.. code-block:: console

    $ python3 train_seq_smnist.py --help


Online Elastic Weight Consolidation (Online EWC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with **Online EWC** and **m = 1** leads to around 98.53% during and 98.45% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=64 --n_iter=2000 --lr=0.005 --clip_grad_norm=-1 --rnn_arch="256" --use_cuda --ssmnist_seq_len=1 --ssmnist_two_classes --use_ewc --ewc_lambda=10000000.0 --n_fisher=200 

The following run with **Online EWC** and **m = 2** leads to around 95.4% during and 88.62% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=64 --n_iter=8000 --lr=0.005 --clip_grad_norm=100 --rnn_arch="256" --use_cuda --ssmnist_seq_len=2 --ssmnist_two_classes --use_ewc --ewc_lambda=10000000.0 --n_fisher=200 

The following run with **Online EWC** and **m = 3** leads to around 83.06% during and 82.28% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=8000 --lr=0.001 --clip_grad_norm=-1 --rnn_arch="256" --use_cuda --ssmnist_seq_len=3 --ssmnist_two_classes --use_ewc --ewc_lambda=10000000.0 --n_fisher=200 

The following run with **Online EWC** and **m = 4** leads to around 90.04% during and 79.01% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=64 --n_iter=20000 --lr=0.001 --clip_grad_norm=100 --rnn_arch="256" --use_cuda --ssmnist_seq_len=4 --ssmnist_two_classes --use_ewc --ewc_lambda=100000.0 --n_fisher=200 

The following run with **Online EWC** and **m = 1 where the task identity is provided as additional input** leads to around 99.39% during and 98.36% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=3000 --lr=0.005 --clip_grad_norm=1 --rnn_arch=256 --use_cuda --input_task_identity --ssmnist_seq_len=1 --ssmnist_two_classes --use_ewc --ewc_lambda=100000000.0 --n_fisher=-1

The following run with **Online EWC** and **m = 2 where the task identity is provided as additional input** leads to around 95.71% during and 88.51% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=8000 --lr=0.0001 --clip_grad_norm=-1 --rnn_arch=256 --use_cuda --input_task_identity --ssmnist_seq_len=2 --ssmnist_two_classes --use_ewc --ewc_lambda=10000.0 --n_fisher=-1

The following run with **Online EWC** and **m = 3 where the task identity is provided as additional input** leads to around 96.63% during and 93.14% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=64 --n_iter=12000 --lr=0.001 --clip_grad_norm=100 --rnn_arch=256 --use_cuda --input_task_identity --ssmnist_seq_len=3 --ssmnist_two_classes --use_ewc --ewc_lambda=10000000.0 --n_fisher=-1

The following run with **Online EWC** and **m = 4 where the task identity is provided as additional input** leads to around 90.65% during and 88.94% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=64 --n_iter=20000 --lr=0.001 --clip_grad_norm=-1 --rnn_arch=256 --use_cuda --input_task_identity --ssmnist_seq_len=4 --ssmnist_two_classes --use_ewc --ewc_lambda=10000000000.0 --n_fisher=-1


Hypernetwork (HNET)
^^^^^^^^^^^^^^^^^^^

The following run with **HNET** and **m = 1** leads to around 99.6% during and 99.56% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --nh_chmlp_chunk_size=10000 --beta=0.01 --multi_head --num_tasks=5 --batch_size=128 --n_iter=2000 --lr=0.001 --clip_grad_norm=-1 --rnn_arch="256" --nh_hnet_type=chunked_hmlp --nh_hmlp_arch="25,25" --nh_cond_emb_size=32 --nh_chunk_emb_size="16" --use_new_hnet --std_normal_temb=1.0 --std_normal_emb=0.1 --use_cuda --hnet_all --ssmnist_seq_len=1 --ssmnist_two_classes 

The following run with **HNET** and **m = 2** leads to around 99.0% during and 98.99% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --nh_chmlp_chunk_size=10000 --beta=0.1 --multi_head --num_tasks=5 --batch_size=128 --n_iter=8000 --lr=0.001 --clip_grad_norm=100 --rnn_arch="256" --nh_hnet_type=chunked_hmlp --nh_hmlp_arch="25,25" --nh_cond_emb_size=32 --nh_chunk_emb_size="16" --use_new_hnet --std_normal_temb=1.0 --std_normal_emb=0.1 --use_cuda --hnet_all --ssmnist_seq_len=2 --ssmnist_two_classes 

The following run with **HNET** and **m = 3** leads to around 98.45% during and 98.48% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --nh_chmlp_chunk_size=5000 --beta=1 --multi_head --num_tasks=5 --batch_size=64 --n_iter=8000 --lr=0.001 --clip_grad_norm=-1 --rnn_arch="256" --nh_hnet_type=chunked_hmlp --nh_hmlp_arch="50,50" --nh_cond_emb_size=32 --nh_chunk_emb_size="16" --use_new_hnet --std_normal_temb=0.1 --std_normal_emb=0.1 --use_cuda --hnet_all --ssmnist_seq_len=3 --ssmnist_two_classes 

The following run with **HNET** and **m = 4** leads to around 98.1% during and 98.02% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --nh_chmlp_chunk_size=10000 --beta=0.01 --multi_head --num_tasks=5 --batch_size=128 --n_iter=20000 --lr=0.001 --clip_grad_norm=-1 --rnn_arch="256" --nh_hnet_type=chunked_hmlp --nh_hmlp_arch="25,25" --nh_cond_emb_size=16 --nh_chunk_emb_size="16" --use_new_hnet --std_normal_temb=0.1 --std_normal_emb=0.1 --use_cuda --hnet_all --ssmnist_seq_len=4 --ssmnist_two_classes 


Main network fine-tuning
^^^^^^^^^^^^^^^^^^^^^^^^

The following run with **Fine-tuning** and **m = 1** leads to around 99.58% during and 94.38% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=64 --n_iter=3000 --lr=0.001 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --ssmnist_seq_len=1 --ssmnist_two_classes 

The following run with **Fine-tuning** and **m = 2** leads to around 99.29% during and 82.54% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=6000 --lr=0.001 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --ssmnist_seq_len=2 --ssmnist_two_classes 

The following run with **Fine-tuning** and **m = 3** leads to around 99.34% during and 70.49% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=64 --n_iter=16000 --lr=0.001 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --ssmnist_seq_len=3 --ssmnist_two_classes 

The following run with **Fine-tuning** and **m = 4** leads to around 99.03% during and 73.34% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=64 --n_iter=25000 --lr=0.001 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --ssmnist_seq_len=4 --ssmnist_two_classes 


Synaptic Intelligence (SI)
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with **SI** and **m = 1** leads to around 99.15% during and 98.54% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=64 --n_iter=3000 --lr=0.005 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --ssmnist_seq_len=1 --ssmnist_two_classes --use_si --si_lambda=0.01 --si_task_loss_only 

The following run with **SI** and **m = 2** leads to around 94.51% during and 90.97% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=6000 --lr=0.001 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --ssmnist_seq_len=2 --ssmnist_two_classes --use_si --si_lambda=0.1 --si_task_loss_only 

The following run with **SI** and **m = 3** leads to around 88.0% during and 83.16% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=16000 --lr=0.0001 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --ssmnist_seq_len=3 --ssmnist_two_classes --use_si --si_lambda=100.0 --si_task_loss_only 

The following run with **SI** and **m = 4** leads to around 88.02% during and 76.61% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=25000 --lr=0.005 --clip_grad_norm=-1 --rnn_arch="256" --use_cuda --ssmnist_seq_len=4 --ssmnist_two_classes --use_si --si_lambda=10.0 --si_task_loss_only 


Masking
^^^^^^^

The following run with **masking** and **m = 1** leads to around 99.58% during and 99.42% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --no_context_mod_outputs --dont_softplus_gains --si_lambda=0.0 --multi_head --num_tasks=5 --batch_size=64 --n_iter=4000 --lr=0.005 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --use_masks --mask_fraction=0.8 --ssmnist_seq_len=1

The following run with **masking** and **m = 2** leads to around 99.26% during and 95.87% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --no_context_mod_outputs --dont_softplus_gains --si_lambda=0.0 --multi_head --num_tasks=5 --batch_size=128 --n_iter=8000 --lr=0.001 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --use_masks --mask_fraction=0.6 --ssmnist_seq_len=2 --ssmnist_two_classes

The following run with **masking** and **m = 3** leads to around 99.30% during and 88.06% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --no_context_mod_outputs --dont_softplus_gains --si_lambda=0.0 --multi_head --num_tasks=5 --batch_size=64 --n_iter=12000 --lr=0.001 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --use_masks --mask_fraction=0.6 --ssmnist_seq_len=3 --ssmnist_two_classes

The following run with **masking** and **m = 4** leads to around 89.15% during and 80.03% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --no_context_mod_outputs --dont_softplus_gains --si_lambda=0.0 --multi_head --num_tasks=5 --batch_size=64 --n_iter=15000 --lr=0.001 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --use_masks --mask_fraction=0.6 --ssmnist_seq_len=4 --ssmnist_two_classes


Masking + SI
^^^^^^^^^^^^

The following run with **Masking + SI** and **m = 1** leads to around 99.73% during and 99.73% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --no_context_mod_outputs --dont_softplus_gains --multi_head --num_tasks=5 --batch_size=128 --n_iter=4000 --lr=0.005 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --use_masks --ssmnist_seq_len=1 --ssmnist_two_classes --use_si --si_lambda=0.01 --si_task_loss_only 

The following run with **Masking + SI** and **m = 2** leads to around 98.86% during and 98.87% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --no_context_mod_outputs --dont_softplus_gains --multi_head --num_tasks=5 --batch_size=128 --n_iter=10000 --lr=0.001 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --use_masks --ssmnist_seq_len=2 --ssmnist_two_classes --use_si --si_lambda=1.0 --si_task_loss_only 

The following run with **masking + SI** and **m = 3** leads to around 98.91% during and 98.91% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --no_context_mod_outputs --dont_softplus_gains --multi_head --num_tasks=5 --batch_size=64 --n_iter=16000 --lr=0.001 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --use_masks --mask_fraction=0.4 --ssmnist_seq_len=3 --ssmnist_two_classes --use_si --si_lambda=1.0 --si_task_loss_only

The following run with **masking + SI** and **m = 4** leads to around 98.85% during and 98.85% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --no_context_mod_outputs --dont_softplus_gains --multi_head --num_tasks=5 --batch_size=64 --n_iter=25000 --lr=0.001 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --use_masks --mask_fraction=0.4 --ssmnist_seq_len=4 --ssmnist_two_classes --use_si --si_lambda=0.01 --si_task_loss_only


Main network from scratch 
^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with **From scratch** and **m = 1** leads to around 99.78% during and 77.37% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --train_from_scratch --num_tasks=5 --batch_size=128 --n_iter=3000 --lr=0.001 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --ssmnist_seq_len=1 --ssmnist_two_classes 

The following run with **From scratch** and **m = 2** leads to around 99.15% during and 49.58% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --train_from_scratch --num_tasks=5 --batch_size=128 --n_iter=8000 --lr=0.001 --clip_grad_norm=-1 --rnn_arch="256" --use_cuda --ssmnist_seq_len=2 --ssmnist_two_classes 

The following run with **From scratch** and **m = 3** leads to around 98.19% during and 57.16% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --train_from_scratch --num_tasks=5 --batch_size=64 --n_iter=12000 --lr=0.005 --clip_grad_norm=100 --rnn_arch="256" --use_cuda --ssmnist_seq_len=3 --ssmnist_two_classes 

The following run with **From scratch** and **m = 4** leads to around 89.42% during and 49.83% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --train_from_scratch --num_tasks=5 --batch_size=64 --n_iter=25000 --lr=0.001 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --ssmnist_seq_len=4 --ssmnist_two_classes 


Coresets
^^^^^^^^

The following run with **Coresets** and **m = 1** leads to around 99.59% during and 99.47% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=64 --n_iter=4000 --lr=0.001 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --use_replay --replay_distill_reg=10 --coreset_size=500 --ssmnist_seq_len=1 --ssmnist_two_classes 

The following run with **Coresets** and **m = 2** leads to around 99.07% during and 98.37% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=64 --n_iter=6000 --lr=0.001 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --use_replay --replay_distill_reg=1.0 --coreset_size=500 --ssmnist_seq_len=2 --ssmnist_two_classes 

The following run with **Coresets** and **m = 3** leads to around 98.72% during and 97.82% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=64 --n_iter=16000 --lr=0.001 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --use_replay --replay_distill_reg=10 --coreset_size=500 --ssmnist_seq_len=3 --ssmnist_two_classes 

The following run with **Coresets** and **m = 4** leads to around 96.35% during and 94.16% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=15000 --lr=0.0001 --clip_grad_norm=100 --rnn_arch="256" --use_cuda --use_replay --replay_distill_reg=10 --coreset_size=500 --ssmnist_seq_len=4 --ssmnist_two_classes 


Multitask
^^^^^^^^^

The following run with **Multitask** and **m = 1** leads to around 99.84% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=512 --n_iter=4000 --lr=0.005 --clip_grad_norm=-1 --rnn_arch="256" --use_cuda --multitask --ssmnist_seq_len=1 --ssmnist_two_classes 

The following run with **Multitask** and **m = 2** leads to around 99.37% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=256 --n_iter=10000 --lr=0.001 --clip_grad_norm=1 --rnn_arch="256" --use_cuda --multitask --ssmnist_seq_len=2 --ssmnist_two_classes 

The following run with **Multitask** and **m = 3** leads to around 99.04% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=128 --n_iter=8000 --lr=0.001 --clip_grad_norm=-1 --rnn_arch="256" --use_cuda --multitask --ssmnist_seq_len=3 --ssmnist_two_classes 

The following run with **Multitask** and **m = 4** leads to around 98.74% final accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --batch_size=512 --n_iter=20000 --lr=0.001 --clip_grad_norm=-1 --rnn_arch="256" --use_cuda --multitask --ssmnist_seq_len=4 --ssmnist_two_classes 


