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
# @title          :sequential/embedding_utils.py
# @author         :be, ch, mc
# @contact        :henningc@ethz.ch
# @created        :08/10/2020
# @version        :1.0
# @python_version :3.6.10
"""
Utilities to ease working with (word) embeddings
------------------------------------------------

This module contains a set of helper functions/classes that should ease the
usage of, for instance, word embeddings.
"""
import torch
import torch.nn as nn
import pickle

def generate_emb_lookups(config, filename=None, padding_idx=0, device=None):
    """Generate a list of models that contain embeddings for different tasks.

    Args:
        config: The configuration.
        filename (str, optional): If provided, the embeddings will be loaded
            from the provided location. Else, they will be initialized randomly.
        padding_idx (int, optional): The value of the indices which correspond
            to padded tokens.
        device: PyTorch device.

    Returns:
        (list): The embedding lookup model (:class:`WordEmbLookup`) for several
        tasks.
    """
    if filename is not None:
        # Load the embeddings.
        embeddings = pickle.load(open(filename, 'rb'), encoding='bytes')
    else:
        raise NotImplementedError # generate randomly

    if len(embeddings) < config.num_tasks:
        raise ValueError('There are not enough available embeddings for all ' +
                         'tasks.')

    emb_lookups = []
    for t in range(config.num_tasks):
        lookup = WordEmbLookup(embeddings[t], padding_idx=padding_idx).to( \
            device)
        emb_lookups.append(lookup)

    return emb_lookups


class WordEmbLookup(nn.Module):
    """A wrapper class for word embeddings.

    This class will instantiate and initialize a set of word embeddings. In
    addition, it will provide a :meth:`forward` method that can be used to
    translate a batch of vocabulary indices into word embeddings.
    
    Attributes:
        embeddings (nn.Embedding): The embeddings.
    """
    def __init__(self, initial_embeddings, padding_idx=0):
        nn.Module.__init__(self)

        vocab_size, embedding_dim = initial_embeddings.shape
        self._embeddings = nn.Embedding(vocab_size, embedding_dim,
            padding_idx=padding_idx)
        self._embeddings.weight.data = torch.tensor(initial_embeddings)

    @property
    def embeddings(self):
        """Getter for read-only attribute :attr:`embeddings`."""
        return self._embeddings

    def forward(self, x):
        """Translate vocabulary indices into word embeddings.

        Args:
            x (torch.Tensor): Batch of vocabulary indices.
                The tensor is of shape ``[T, B]`` or ``[T, B, 1]`` with ``T``
                denoting the number of timesteps and ``B`` denoting the batch
                size.

        Returns:
            (torch.Tensor): A batch of word embeddings. The output tensor is of
            shape ``[T, B, K]``, where ``K`` is the dimensionality of individual
            word embeddings.
        """
        assert len(x.shape) == 2 or len(x.shape) == 3 and x.shape[2] == 1

        embedded = self.embeddings(x)
        if len(embedded.shape) > 3:
            embedded = torch.squeeze(embedded, dim=2)

        return embedded

if __name__ == '__main__':
    pass


