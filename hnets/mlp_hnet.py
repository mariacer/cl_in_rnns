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
# @title          :hnets/mlp_hnet.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :04/14/2020
# @version        :1.0
# @python_version :3.6.10
"""
MLP - Hypernetwork
------------------

The module :mod:`hnets.mlp_hnet` contains a fully-connected hypernetwork
(also termed `full hypernet`).

This type of hypernetwork represents one of the most simplistic architectural
choices to realize a weight generator. An embedding input, which may consists of
conditional and unconditional parts (for instance, in the case of
`task-conditioned hypernetwork <https://arxiv.org/abs/1906.00695>`__ the
conditional input will be a task embedding) is mapped via a series of fully-
connected layers onto a final hidden representation. Then a linear
fully-connected output layer per is used to produce the target weights, output
tensors with shapes specified via the target shapes (see
:attr:`hnets.hnet_interface.HyperNetInterface.target_shapes`).

If no hidden layers are used, then this resembles a simplistic linear
hypernetwork, where the input embeddings are linearly mapped onto target
weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from warnings import warn

from hnets.hnet_interface import HyperNetInterface

class HMLP(nn.Module, HyperNetInterface):
    """Implementation of a `full hypernet`.

    The network will consist of several hidden layers and a final linear output
    layer that produces all weight matrices/bias-vectors the network has to
    produce.

    The network allows to maintain a set of embeddings internally that can be
    used as conditional input.

    Args:
        target_shapes (list): List of lists of intergers, i.e., a list of tensor
            shapes. Those will be the shapes of the output weights produced by
            the hypernetwork. For each entry in this list, a separate output
            layer will be instantiated.
        uncond_in_size (int): The size of unconditional inputs (for instance,
            noise).
        cond_in_size (int): The size of conditional input embeddings.

            Note, if ``no_cond_weights`` is ``False``, those embeddings will be
            maintained internally.
        layers (list or tuple): List of integers denoteing the sizes of each
            hidden layer. If empty, no hidden layers will be produced.
        verbose (bool): Whether network information should be printed during
            network creation.
        activation_fn (func): The activation function to be used for hidden
            activations. For instance, an instance of class
            :class:`torch.nn.ReLU`.
        use_bias (bool): Whether the fully-connected layers that make up this
            network should have bias vectors.
        no_uncond_weights (bool): If ``True``, unconditional weights are not
            maintained internally and instead expected to be produced
            externally and passed to the :meth:`forward`.
        no_cond_weights (bool): If ``True``, conditional embeddings are assumed
            to be maintained externally. Otherwise, option ``num_cond_embs``
            has to be properly set, which will determine the number of
            embeddings that are internally maintained.
        num_cond_embs (int): Number of conditional embeddings to be internally
            maintained. Only used if option ``no_cond_weights`` is ``False``.

            Note:
                Embeddings will be initialized with a normal distribution using
                zero mean and unit variance.
        dropout_rate (float): If ``-1``, no dropout will be applied. Otherwise a
            number between 0 and 1 is expected, denoting the dropout rate of
            hidden layers.
        use_spectral_norm (bool): Use spectral normalization for training.
        use_batch_norm (bool): Whether batch normalization should be used. Will
            be applied before the activation function in all hidden layers.

            Note:
                Batch norm only makes sense if the hypernetwork is envoked with
                batch sizes greater than 1 during training.
    """
    def __init__(self, target_shapes, uncond_in_size=0, cond_in_size=8,
                 layers=(100, 100), verbose=True, activation_fn=torch.nn.ReLU(),
                 use_bias=True, no_uncond_weights=False, no_cond_weights=False,
                 num_cond_embs=1, dropout_rate=-1, use_spectral_norm=False,
                 use_batch_norm=False):
        # FIXME find a way using super to handle multiple inheritance.
        nn.Module.__init__(self)
        HyperNetInterface.__init__(self)

        if use_spectral_norm:
            raise NotImplementedError('Spectral normalization not yet ' +
                                      'implemented for this hypernetwork type.')

        assert len(target_shapes) > 0
        if cond_in_size == 0 and num_cond_embs > 0:
            warn('Requested that conditional weights are managed, but ' +
                 'conditional input size is zero! Setting "num_cond_embs" to ' +
                 'zero.')
            num_cond_embs = 0
        elif not no_cond_weights and num_cond_embs == 0 and cond_in_size > 0:
            warn('Requested that conditional weights are internally ' +
                 'maintained, but "num_cond_embs" is zero.')
        # Do we maintain conditional weights internally?
        has_int_cond_weights = not no_cond_weights and num_cond_embs > 0
        # Do we expect external conditional weights?
        has_ext_cond_weights = no_cond_weights and num_cond_embs > 0

        ### Make constructor arguments internally available ###
        self._uncond_in_size = uncond_in_size
        self._cond_in_size = cond_in_size
        self._layers = layers
        self._act_fn = activation_fn
        self._no_uncond_weights = no_uncond_weights
        self._no_cond_weights = no_cond_weights
        self._num_cond_embs = num_cond_embs
        self._dropout_rate = dropout_rate
        self._use_spectral_norm = use_spectral_norm
        self._use_batch_norm = use_batch_norm

        ### Setup attributes required by interface ###
        self._target_shapes = target_shapes
        self._num_known_conds = self._num_cond_embs
        self._unconditional_param_shapes_ref = []

        self._has_bias = use_bias
        self._has_fc_out = True
        self._mask_fc_out = True
        self._has_linear_out = True

        self._param_shapes = []
        self._param_shapes_meta = []
        self._internal_params = None if no_uncond_weights and \
            has_int_cond_weights else nn.ParameterList()
        self._hyper_shapes_learned = None \
            if not no_uncond_weights and has_ext_cond_weights else []
        self._hyper_shapes_learned_ref = None if self._hyper_shapes_learned \
            is None else []
        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        self._dropout = None
        if dropout_rate != -1:
            assert dropout_rate > 0 and dropout_rate < 1
            self._dropout = nn.Dropout(dropout_rate)

        ### Create conditional weights ###
        for _ in range(num_cond_embs):
            assert cond_in_size > 0
            if not no_cond_weights:
                self._internal_params.append(nn.Parameter( \
                    data=torch.Tensor(cond_in_size), requires_grad=True))
                torch.nn.init.normal_(self._internal_params[-1], mean=0.,
                                      std=1.)
            else:
                self._hyper_shapes_learned.append([cond_in_size])
                self._hyper_shapes_learned_ref.append(len(self.param_shapes))

            self._param_shapes.append([cond_in_size])
            # Embeddings belong to the input, so we just assign them all to
            # "layer" 0.
            self._param_shapes_meta.append({
                'name': 'embedding',
                'index': -1 if no_cond_weights else \
                    len(self._internal_params)-1,
                'layer': 0
            })

        ### Create batch-norm layers ###
        # We just use even numbers starting from 2 as layer indices for
        # batchnorm layers.
        if use_batch_norm:
            self._add_batchnorm_layers(layers, no_uncond_weights,
                bn_layers=list(range(2, 2*len(layers)+1, 2)),
                distill_bn_stats=False, bn_track_stats=True)

        ### Create fully-connected hidden-layers ###
        in_size = uncond_in_size + cond_in_size
        if len(layers) > 0:
            # We use odd numbers starting at 1 as layer indices for hidden
            # layers.
            self._add_fc_layers([in_size, *layers[:-1]], layers,
                no_uncond_weights, fc_layers=list(range(1, 2*len(layers), 2)))
            hidden_size = layers[-1]
        else:
            hidden_size = in_size

        ### Create fully-connected output-layers ###
        # Note, technically there is no difference between having a separate
        # fully-connected layer per target shape or a single fully-connected
        # layer producing all weights at once (in any case, each output is
        # connceted to all hidden units).
        # I guess it is more computationally efficient to have one output layer
        # and then split the output according to the target shapes.
        self._add_fc_layers([hidden_size], [self.num_outputs],
                            no_uncond_weights, fc_layers=[2*len(layers)+1])

        ### Finalize construction ###
        # All parameters are unconditional except the embeddings created at the
        # very beginning.
        self._unconditional_param_shapes_ref = \
            list(range(num_cond_embs, len(self.param_shapes)))

        self._is_properly_setup()

        if verbose:
            print('Created MLP Hypernet.')
            print(self)

    def forward(self, uncond_input=None, cond_input=None, cond_id=None,
                weights=None, distilled_params=None, condition=None,
                ret_format='squeezed', ext_inputs=None, task_emb=None,
                task_id=None, theta=None, dTheta=None):
        """Compute the weights of a target network.

        Args:
            (....): See docstring of method
                :meth:`hnets.hnet_interface.HyperNetInterface.forward`.
            condition (int):This argument will be passed as argument
                ``stats_id`` to the method
                :meth:`utils.batchnorm_layer.BatchNormLayer.forward` if batch
                normalization is used.

        Returns:
            (list or torch.Tensor): See docstring of method
            :meth:`hnets.hnet_interface.HyperNetInterface.forward`.
        """
        uncond_input, cond_input, uncond_weights, _ = \
            self._preprocess_forward_args(uncond_input=uncond_input,
                cond_input=cond_input, cond_id=cond_id, weights=weights,
                distilled_params=distilled_params, condition=condition,
                ret_format=ret_format, ext_inputs=ext_inputs, task_emb=task_emb,
                task_id=task_id, theta=theta, dTheta=dTheta)

        ### Prepare hypernet input ###
        assert self._uncond_in_size == 0 or uncond_input is not None
        assert self._cond_in_size == 0 or cond_input is not None
        if uncond_input is not None:
            assert len(uncond_input.shape) == 2 and \
                   uncond_input.shape[1] == self._uncond_in_size
            h = uncond_input
        if cond_input is not None:
            assert len(cond_input.shape) == 2 and \
                   cond_input.shape[1] == self._cond_in_size
            h = cond_input
        if uncond_input is not None and cond_input is not None:
            h = torch.cat([uncond_input, cond_input], dim=1)

        ### Extract layer weights ###
        bn_scales = []
        bn_shifts = []
        fc_weights = []
        fc_biases = []

        assert len(uncond_weights) == len(self.unconditional_param_shapes_ref)
        for i, idx in enumerate(self.unconditional_param_shapes_ref):
            meta = self.param_shapes_meta[idx]

            if meta['name'] == 'bn_scale':
                bn_scales.append(uncond_weights[i])
            elif meta['name'] == 'bn_shift':
                bn_shifts.append(uncond_weights[i])
            elif meta['name'] == 'weight':
                fc_weights.append(uncond_weights[i])
            else:
                assert meta['name'] == 'bias'
                fc_biases.append(uncond_weights[i])

        if not self.has_bias:
            assert len(fc_biases) == 0
            fc_biases = [None] * len(fc_weights)

        if self._use_batch_norm:
            assert len(bn_scales) == len(fc_weights) - 1

        ### Process inputs through network ###
        for i in range(len(fc_weights)):
            last_layer = i == (len(fc_weights) - 1)

            h = F.linear(h, fc_weights[i], bias=fc_biases[i])

            if not last_layer:
                # Batch-norm
                if self._use_batch_norm:
                    h = self.batchnorm_layers[i].forward(h, running_mean=None,
                        running_var=None, weight=bn_scales[i],
                        bias=bn_shifts[i], stats_id=condition)

                # Dropout
                if self._dropout_rate != -1:
                    h = self._dropout(h)

                # Non-linearity
                if self._act_fn is not None:
                    h = self._act_fn(h)

        ### Split output into target shapes ###
        ret = self._flat_to_ret_format(h, ret_format)

        return ret

    def distillation_targets(self):
        """Targets to be distilled after training.

        See docstring of abstract super method
        :meth:`mnets.mnet_interface.MainNetInterface.distillation_targets`.

        This network does not have any distillation targets.

        Returns:
            ``None``
        """
        return None

    def apply_hyperfan_init(self, method='in', use_xavier=False,
                            uncond_var=1., cond_var=1.):
        """Not implemented yet!"""
        # TODO Translate from old hypernet implementation and take meta
        # information of generated parameters into account.
        raise NotImplementedError()

    def get_cond_in_emb(self, cond_id):
        """Get the ``cond_id``-th (conditional) input embedding.

        Args:
            cond_id (int): Determines which input embedding should be returned
                (the ID has to be between ``0`` and ``num_cond_embs-1``, where
                ``num_cond_embs`` denotes the corresponding constructor
                argument).

        Returns:
            (torch.nn.Parameter)
        """
        if self.conditional_params is None:
            raise RuntimeError('Input embeddings are not internally ' +
                               'maintained!')
        if not isinstance(cond_id, int) or cond_id < 0 or \
                cond_id >= len(self.conditional_params):
            raise RuntimeError('Option "cond_id" must be between 0 and %d!' \
                               % (len(self.conditional_params)-1))
        return self.conditional_params[cond_id]

if __name__ == '__main__':
    pass