Hypernetworks
*************

.. contents::

A `hypernetwork <https://arxiv.org/abs/1609.09106>`__ is a neural network that produces the weights of another network. As such, it can be seen as a specific type of main network (aka neural network). Therefore, each hypernetwork has a specific interface :class:`hnets.hnet_interface.HyperNetInterface` which is derived from the main network interface :class:`mnets.mnet_interface.MainNetInterface`. 

.. note::
    Currently, hypernetworks are positioned in arbitrary subpackages within this repository. In the foreseeable future all these networks will be deprecated and migrated into this subpackage :mod:`hnets`.

.. note::
    All hypernetworks in this subpackage implement the abstract interface :class:`hnets.hnet_interface.HyperNetInterface` to provide a consistent interface for users.

API
===

.. automodule:: hnets.hnet_interface
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: hnets.chunked_deconv_hnet
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: hnets.chunked_mlp_hnet
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: hnets.deconv_hnet
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: hnets.mlp_hnet
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: hnets.structured_mlp_hnet
    :members:
    :undoc-members:
    :show-inheritance:
