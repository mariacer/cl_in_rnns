# Continual Learning in Recurrent Neural Networks with Hypernetworks

A continual learning approach for recurrent neural networks that has the flexibility to learn a dedicated set of parameters, fine-tuned for every task, that doesn't require an increase in the number of trainable weights and is robust against catastrophic forgetting.

For details on this approach please read [our paper](TODO).


## Copy Task Experiments

You can find instructions on how to reproduce our experiments on all Copy Task variants and on how to use the corresponding code in the subfolder [sequential.copy](sequential/copy).


## Stroke MNIST Experiments

You can find instructions on how to reproduce our Stroke MNIST experiments and on how to use the corresponding code in the subfolder [sequential.smnist](sequential/smnist).

## Sequential Stroke MNIST Experiments

You can find instructions on how to reproduce our Sequential Stroke MNIST experiments and on how to use the corresponding code in the subfolder [sequential.seq_smnist](sequential/seq_smnist).

## Audioset Experiments

You can find instructions on how to reproduce our Audioset experiments and on how to use the corresponding code in the subfolder [sequential.audioset](sequential/audioset).


## Documentation

Please refer to the [README](docs/README.md) in the subfolder [docs](docs) for instructions on how to compile and open the documentation.

## Setup Python Environment

We use [conda](https://www.anaconda.com/) to manage Python environments. To create an environment that already fulfills all package requirements of this repository, simply execute

```console
$ conda env create -f environment.yml
$ conda activate cl_rnn_env
```

## Citation
Please cite our paper if you use this code in your research project.

```
TODO
```
