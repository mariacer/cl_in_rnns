Split Sequential Stroke-MNIST experiments for continual learning
================================================================

.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.
.. content-inclusion-marker-do-not-remove

In this subpackage we conduct experiments on Split Sequential Stroke MNIST. The sequence length **m** determines the difficulty of individual tasks.
Please run the following command to see the available options for running Split SMNIST experiments.

.. code-block:: console

    $ python3 train_split_smnist.py --help


Online Elastic Weight Consolidation (Online EWC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with **Online EWC** and **m = 1** leads to around 97% accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --n_iter=2000 --multi_head --num_tasks=5 --lr=0.001 --use_cuda --ssmnist_seq_len=1 --ssmnist_two_classes --use_ewc --ewc_lambda=1000000000.0 --n_fisher=200

The following run with **Online EWC** and **m = 2** leads to around 79% accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --multi_head --num_tasks=5 --n_iter=8000 --lr=0.001 --use_cuda --ssmnist_seq_len=2 --ssmnist_two_classes --use_ewc --ewc_lambda=1000000000.0 --n_fisher=200

The following run with **Online EWC** and **m = 3** leads to around 78% accuracy:

.. code-block:: console
    
    $ python3 train_seq_smnist.py --n_iter=10000 --multi_head --num_tasks=5 --lr=0.001 --use_cuda --ssmnist_seq_len=3 --ssmnist_two_classes --use_ewc --ewc_lambda=10000000000.0 --n_fisher=200

The following run with **Online EWC** and **m = 4** leads to around 74% accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --n_iter=20000 --multi_head --num_tasks=5 --lr=0.001 --use_cuda --ssmnist_seq_len=4 --ssmnist_two_classes --use_ewc --ewc_lambda=100000000.0 --n_fisher=200


Hypernetwork (HNET)
^^^^^^^^^^^^^^^^^^^

The following run with **HNET** and **m = 1** leads to around 99% accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --hyper_chunks=8000 --temb_size=64 --emb_size=64 --n_iter=2000 --beta=0.01 --multi_head --num_tasks=5 --lr=0.001 --clip_grad_norm=1 --hnet_arch="32,32,32" --hnet_act=relu --use_cuda --hnet_all --ssmnist_seq_len=1 --ssmnist_two_classes

The following run with **HNET** and **m = 2** leads to around 98% accuracy:

.. code-block:: console
 
    $ python3 train_seq_smnist.py --hyper_chunks=8000 --temb_size=64 --emb_size=64 --n_iter=5000 --beta=1.0 --multi_head --num_tasks=5 --lr=0.001 --clip_grad_norm=100 --hnet_arch="32,32,32" --hnet_act=relu --use_cuda --hnet_all --ssmnist_seq_len=2 --ssmnist_two_classes

The following run with **HNET** and **m = 3** leads to around 95% accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --hyper_chunks=8000 --temb_size=64 --emb_size=64 --n_iter=10000 --beta=1.0 --multi_head --num_tasks=5 --lr=0.0001 --clip_grad_norm=1 --hnet_arch="32,32,32" --hnet_act=relu --use_cuda --hnet_all --ssmnist_seq_len=3 --ssmnist_two_classes

The following run with **HNET** and **m = 4** leads to around 93% accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --hyper_chunks=8000 --temb_size=64 --emb_size=64 --n_iter=20000 --beta=0.01 --multi_head --num_tasks=5 --lr=0.0001 --clip_grad_norm=1 --hnet_arch="32,32" --hnet_act=relu --use_cuda --hnet_all --ssmnist_seq_len=4 --ssmnist_two_classes


Main network fine-tuning
^^^^^^^^^^^^^^^^^^^^^^^^

The following run with **main network fine-tuning** and **m = 1** leads to around 93% accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --n_iter=2000 --multi_head --num_tasks=5 --lr=0.001 --use_cuda --ssmnist_seq_len=1 --ssmnist_two_classes

The following run with **main network fine-tuning** and **m = 2** leads to around 60% accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --n_iter=5000 --multi_head --num_tasks=5 --lr=0.01 --use_cuda --ssmnist_seq_len=2 --ssmnist_two_classes

The following run with **main network fine-tuning** and **m = 3** leads to around 69% accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --n_iter=10000 --multi_head --num_tasks=5 --lr=0.0001 --use_cuda --ssmnist_seq_len=3 --ssmnist_two_classes

The following run with **main network fine-tuning** and **m = 4** leads to around 59% accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --n_iter=20000 --multi_head --num_tasks=5 --lr=0.001 --use_cuda --ssmnist_seq_len=4 --ssmnist_two_classes


Synaptic Intelligence (SI)
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with **SI** and **m = 1** leads to around 98% accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --n_iter=2000 --multi_head --num_tasks=5 --lr=0.01 --use_cuda --ssmnist_seq_len=1 --ssmnist_two_classes --use_si --si_lambda=0.1 --si_task_loss_only

The following run with **SI** and **m = 2** leads to around 83% accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --n_iter=5000 --multi_head --num_tasks=5 --lr=0.001 --use_cuda --ssmnist_seq_len=2 --ssmnist_two_classes --use_si --si_lambda=0.1 --si_task_loss_only

The following run with **SI** and **m = 3** leads to around 76% accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --n_iter=10000 --multi_head --num_tasks=5 --lr=0.0001 --use_cuda --ssmnist_seq_len=3 --ssmnist_two_classes --use_si --si_lambda=10.0 --si_task_loss_only

The following run with **SI** and **m = 4** leads to around 70% accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --n_iter=20000 --multi_head --num_tasks=5 --lr=0.001 --use_cuda --ssmnist_seq_len=4 --ssmnist_two_classes --use_si --si_lambda=0.01 --si_task_loss_only


Masking + SI
^^^^^^^^^^^^

The following run with **masking + SI** and **m = 1** leads to around 99% accuracy:

.. code-block:: console
 
    $ python3 train_seq_smnist.py --no_context_mod_outputs --dont_softplus_gains --n_iter=2000 --multi_head --num_tasks=5 --lr=0.01 --use_cuda --use_masks --ssmnist_seq_len=1 --ssmnist_two_classes --use_si --si_lambda=100.0

The following run with **masking + SI** and **m = 2** leads to around 98% accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --no_context_mod_outputs --dont_softplus_gains --multi_head --num_tasks=5 --n_iter=8000 --lr=0.01 --use_cuda --use_masks --ssmnist_seq_len=2 --ssmnist_two_classes --use_si --si_lambda=1000.0

The following run with **masking + SI** and **m = 3** leads to around 97% accuracy:

.. code-block:: console
 
    $ python3 train_seq_smnist.py --no_context_mod_outputs --dont_softplus_gains --n_iter=10000 --multi_head --num_tasks=5 --lr=0.01 --use_cuda --use_masks --ssmnist_seq_len=3 --ssmnist_two_classes --use_si --si_lambda=10.0

The following run with **masking + SI** and **m = 4** leads to around 72% accuracy:

.. code-block:: console

    $ python3 train_seq_smnist.py --no_context_mod_outputs --dont_softplus_gains --n_iter=20000 --multi_head --num_tasks=5 --lr=0.0001 --use_cuda --use_masks --ssmnist_seq_len=4 --ssmnist_two_classes --use_si --si_lambda=0.1
