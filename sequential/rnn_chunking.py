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
# @title          :sequential/rnn_chunking.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :05/15/2020
# @version        :1.0
# @python_version :3.6.10
"""
Helper Functions for using Structured Chunked MLP - Hypernetworks with RNNs
---------------------------------------------------------------------------

The module :mod:`sequential.rnn_chunking` provides helpers to create
instantiations of :class:`hnets.structured_mlp_hnet.StructuredHMLP` for target
networks of class :class:`mnets.simple_rnn.SimpleRNN`.

In principle, this module can be seen as an extension of module
:mod:`hnets.structured_hmlp_examples`.
"""
import numpy as np
import torch
from warnings import warn

from mnets.simple_rnn import SimpleRNN

def simple_rnn_chunking(net, chunk_size=None, fc_chunking=False):
    """Design a structured chunking for a :class:`mnets.simple_rnn.SimpleRNN`.

    By default, this function chunks all recurrent layers according to argument
    ``chunk_size`` as described below. The weights of all fully-connected layers
    will be considered separate chunks (except if option ``shared_fc_chunking``
    is used).

    Consider the simple case of an LSTM layer of size ``r`` (omitting biases)
    with input-to-hidden shape ``[4r, in]``, where ``in`` denotes the input
    dimension, and hidden-to-hidden shape ``[4r, r]``. Let `c`` be the
    specified ``chunk_size`` (see argument description). In addition, let
    ``[out, r]`` denote the size of the fully-connected output layer (if any).
    Then, the following 2 chunks will be returned as ``chunk_shapes``:
    ``[[[c, in], [c, r]], [[out, r]]]``. The return value ``num_per_chunk``
    would be: ``[4r // c, 1]``, assuming that ``4r % c == 0``.


    Args:
        net (mnets.simple_rnn.SimpleRNN): The network to be chunked.
        chunk_size (int or list, optional): If not specified, the chunk size
            will simply be the size of each recurrent layer. Otherwise, either a
            list of chunk sizes has to be given or all recurrent layer sizes
            must be dividable by the given ``chunk_size``.
        fc_chunking (bool): If ``True``, all fully-connected layers
            are also effected by ``chunk_size``. Hence, all layers would be
            chunked.

    Returns:
        (tuple): See return value of function
        :func:`hnets.structured_hmlp_examples.resnet_chunking`.
    """
    if not isinstance(net, SimpleRNN):
        raise ValueError('Function only applies to networks of class ' +
                         '"SimpleRNN".')
    for meta in net.param_shapes_meta:
        if meta['name'] not in ('weight', 'bias'):
            raise ValueError('Function does not support networks with ' +
                             'weights of type "%s".' % meta['name'])
    # That's what we assume here; producing all weights with a hnet.
    assert len(net.hyper_shapes_learned) == len(net.param_shapes)

    assert hasattr(net, '_rnn_layers') and hasattr(net, '_fc_layers_pre') and \
           hasattr(net, '_fc_layers')

    n_rec = len(net._rnn_layers)
    n_pre = len(net._fc_layers_pre)
    n_post = len(net._fc_layers)

    if fc_chunking:
        chunked_layers = list(net._fc_layers_pre) + list(net._rnn_layers) + \
            list(net._fc_layers)
    else:
        chunked_layers = list(net._rnn_layers)

    if chunk_size is None:
        chunk_size = chunked_layers
    elif isinstance(chunk_size, int):
        chunk_size = [chunk_size] * len(chunked_layers)
    else:
        if len(chunked_layers) != len(chunk_size):
            warn('Given "chunk_size"s list contains %d elements, but is ' \
                 % (len(chunk_size)) + 'expected to contain %d elements. ' \
                 % (len(chunked_layers)) + 'Non-existing elements will be ' +
                 'replaced by corresponding layer sizes.')
            if len(chunked_layers) > len(chunk_size):
                d = len(chunked_layers) - len(chunk_size)
                chunk_size.extend(chunked_layers[-d:])
            else:
                chunk_size = chunk_size[:len(chunked_layers)]
            assert len(chunked_layers) == len(chunk_size)

    for i, s in enumerate(chunked_layers):
        if s % chunk_size[i] != 0 or not 0 < chunk_size[i] <= s:
            warn('%d is not a valid chunk size for layer of size %d. ' \
                 % (chunk_size[i], s) + 'Using chunk size %d instead.' \
                 % (chunked_layers[i]))
            chunk_size[i] = chunked_layers[i]

    # Collect shapes of pre-fc, recurrent and post-fc layers.
    pre_shapes = []
    rec_shapes = []
    post_shapes = []
    # Note, odd numbers are layer inds.
    layer_ind = 1
    for i, meta in enumerate(net.param_shapes_meta):
        assert meta['layer'] == layer_ind

        is_rec_layer = 'info' in meta.keys()

        if not is_rec_layer and len(rec_shapes) == 0:
            if meta['name'] == 'weight':
                pre_shapes.append([])
            pre_shapes[-1].append((net.param_shapes[i], i))
        elif is_rec_layer:
            if meta['name'] == 'weight' and meta['info'] == 'ih':
                rec_shapes.append([])
            rec_shapes[-1].append((net.param_shapes[i], i))
        else:
            if meta['name'] == 'weight':
                post_shapes.append([])
            post_shapes[-1].append((net.param_shapes[i], i))

        if is_rec_layer:
            is_last_rec_weight = net.use_lstm and meta['info'] == 'hh' or \
                not net.use_lstm and meta['info'] == 'ho'
        else:
            is_last_rec_weight = False

        if (not is_rec_layer or is_last_rec_weight) and \
                (not net.has_bias or meta['name'] == 'bias'):
            layer_ind += 2

    chunk_shapes = []
    num_per_chunk = []
    assembly_fct = None

    if not fc_chunking:
        for i in range(n_pre):
            chunk_shapes.append([t[0] for t in pre_shapes[i]])
            num_per_chunk.append(1)

    def _add_chunk(shapes):
        nonlocal ind

        c = chunk_size[ind]
        chunk_shapes.append([[c] + t[0][1:] for t in shapes])
        n = shapes[0][0][0] // c
        num_per_chunk.append(n)
        assert np.all([t[0][0] == n * c for t in shapes])
        ind += 1

    ind = 0
    if fc_chunking:
        for i in range(n_pre):
            _add_chunk(pre_shapes[i])
    for i in range(n_rec):
        _add_chunk(rec_shapes[i])
    if fc_chunking:
        for i in range(n_post):
            _add_chunk(post_shapes[i])

    if not fc_chunking:
        for i in range(n_post):
            chunk_shapes.append([t[0] for t in post_shapes[i]])
            num_per_chunk.append(1)

    assembly_fct = lambda x : _simple_rnn_chunking_afct(x, net, chunk_shapes,
        num_per_chunk)

    return chunk_shapes, num_per_chunk, assembly_fct

def _simple_rnn_chunking_afct(list_of_chunks, net, chunk_shapes, num_per_chunk):
    """The ``assembly_fct`` function required by function
    :func:`simple_rnn_chunking`.
    """
    assert len(list_of_chunks) == np.sum(num_per_chunk)

    target_weights = []

    for i, n in enumerate(num_per_chunk):
        chunks = list_of_chunks[:n]
        list_of_chunks = list_of_chunks[n:]

        for j in range(len(chunk_shapes[i])):
            target_weights.append(torch.cat([c[j] for c in chunks], dim=0))

    return target_weights

if __name__ == '__main__':
    pass


