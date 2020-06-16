Continual learning in Recurrent Neural Networks
===============================================

.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.
.. content-inclusion-marker-do-not-remove

This subpackage of the repository implements a series of experiments to assess different approaches for continual learning on recurrent neural networks (RNN).

Remark on available hypernetworks
---------------------------------

The code in this subpackage uses both, the deprecated implementations of our old hypernetworks as well as the new hypernetworks that implement the interface :class:`hnet.hnet_interface.HyperNetInterface`. The reason for that is simply that the project was developed in parallel with the new hypernets. There are two separate argument groups for old and new hypernets and all arguments related to new hypernetworks start with the prefix ``nh_``.

The new hypernetworks are used when option ``--use_new_hnet`` is given. Note, in principle, if a run is specified using an old hypernetwork, one should be able to translate the arguments to use the corresponding new hypernetwork. However, this has not been thorougly tested yet.

Continual Learning with Generative Models
-----------------------------------------

Generative models allow to replay data from old tasks. Mixing the replayed data with current data allows training in an `approximate` multitask fashion. Hence, if the quality of the generative model is high, this continual learning algorithm has all the benefits of multitask training, i.e., no-forgetting by construction and transfer between all tasks (forward and backward).

However, generative models are usually more complex than discriminative ones, which is why we consider the construction of a separate generative model per task (as done, for instance, in `Cossu et al. <https://openreview.net/forum?id=HklliySFDS>`__) not a viable solution for continual learning. Instead, the problem of continual learning shifts from the discriminative to the generative model when training is done as described above (cmp., `Shin et al. <https://arxiv.org/abs/1705.08690>`__).

We consider two methods of preventing catastrophic forgetting in the generative model.

Replay through Feedback
^^^^^^^^^^^^^^^^^^^^^^^

This method was proposed by `van de Ven et al. <https://arxiv.org/pdf/1809.10635.pdf>`__ for non-sequential data. The idea is to train a variational autoencoder (VAE), where the encoder is also the target discriminative model we are interested in. Thus, the discriminator has two outputs, the target output (e.g., an input classification) and the VAE latent distribution. The decoder is checkpointed at the beginning of training a new task. This checkpointed decoder can be used to replay data from old tasks by sampling latent codes from the prior. This replayed data is then processed by a checkpointed encoder to produce `soft targets` (cmp. `Hinton et al. <https://arxiv.org/abs/1503.02531>`__).

The VAE is then trained on a mixture of replayed and current (input) data. The encoder is additionally trained on current data to perform on this data as well as on replayed data, where the `soft targets` are distilled from the checkpointed encoder.

In general, it is important to distinguish between single-head and multi-head. In the multi-head case, we need to be able to select the correct output head during test time. Therefore, only in this case, the decoder gets an additional 1-hot encoding as input, that tells it the task for which samples should be produced. This allows the selection of the correct output head when training the encoder on `soft targets`.

For classification, the learning scenarios proposed in `van de Ven et al. <https://arxiv.org/abs/1904.07734>`__ can be realized as follows.

- **CL1** - Train a multi-head classifier using replayed data: ``--use_replay --multi_head``.
- **CL2** - Train a single-head classifier using replayed data (task identity not needed during inference): ``--use_replay``.
- **CL3** - Train a softmax across all tasks using replayed data (task identity inferred by encoder): ``--use_replay --all_task_softmax``.

Hypernetwork-protected decoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method was proposed by `von Oswald et al. <https://arxiv.org/abs/1906.00695>`__ for non-sequential data (termed **HNET+R**). The idea is, that the decoder is not trained on replayed data, as errors would accumulate over time and the replayed distribution would shift. Instead, there is a decoder per task, protected by a single hypernetwork.

Most of what has been described above also applies to the **HNET+R** case. A checkpointed encoder/decoder is used to replay data from old tasks and compute the corresponding soft-targets for training the encoder. Additionally, the VAE is trained on the new task data only (given a new task embedding for the hypernetworks). The old decoders are protected by the hypernetwork regularizer.

In contrast to the method described above, the multi-head system does not require an additional 1-hot encoding as decoder input, as the decoders are anyway task-conditioned.

The CL scenarios from above can be realized by appending the additional option ``--hnet_all`` (**note**, the hypernetwork will protect the decoder if ``--use_replay`` is used and not the encoder).
