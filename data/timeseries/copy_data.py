#!/usr/bin/env python3
# Copyright 2019 Maria Cervera
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
# title           :data/timeseries/copy_data.py
# author          :mc
# contact         :mariacer@ethz.ch
# created         :20/03/2020
# version         :1.0
# python_version  :3.7
"""
Dataset for the sequential copy task
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A data handler for the copy task as described in: 

    https://arxiv.org/pdf/1410.5401.pdf

A typical usecase of this dataset is in an incremental learning setting. For
instance, a sequence of tasks with increasing lengths can be used in curriculum
learning or continual learning.
"""
import copy
import matplotlib.pyplot as plt
import numpy as np
from warnings import warn
import torch

from data.sequential_dataset import SequentialDataset

class CopyTask(SequentialDataset):
    """Data handler for the sequential copy task.

    In this task, a binary vector is presented as input, and the network has
    to learn to copy it. Such that the network cannot rely on intermediate 
    information, there is a delay between the end of the input presentation and
    the output generation. The end of the sequence is delimited by a binary 
    bit, which is always zero except when the sequence finishes. This flag
    should not be copied.

    An instance of this class will represent copy task patterns of random
    length but fixed width. The length of input patterns will be sampled 
    uniformly from the interval ``[min_input_len, max_input_len]``. Note that 
    the actual length of the patterns ``pat_len`` might be smaller in the case 
    where there are a certain number of zero-valued timesteps within the input.
    As such, every sequence is characterised by the following values:

    - ``pat_len``: the actual length of the binary pattern to be copied.
      Across this duration, half the pixels have value of 1 and the other
      half have value 0.
    - ``input_len``: the length of input presentation up until the stop 
      flag. It is equal to the pattern length plus the number of zero-valued
      timesteps.
    - ``seq_len``: the length of the entire sequences, including input
      presentation, stop flag and output generation. Therefore it is equal
      to the input length, plus one (stop flag), plus the pattern length 
      (since during output reconstruction we don't care about reconstructing
      the zero-valued part of the input).

    Caution:
        Manipulations such as permutations or scattering/masking will be applied
        online in :meth:`output_to_torch_tensor`.

    Args:
        min_input_len (int): The minimum length of an input sequence.

            Note:
                The input length is the length of the presented input before the
                stop flag. It might include both a pattern to be copied and a 
                set of zero-valued timesteps that do not need to be 
                reconstructed.
        max_input_len (int): The maximum length of a pattern.
        seq_width (int): The width if each pattern.

            Note:
                Each pattern will have a certain length (across time) and
                a certain width.
        num_train (int): Number of training samples.
        num_test (int): Number of test samples.
        num_val (int, optional): Number of validation samples.
        pat_len (int, optional): The actual length of the pattern within the
            input sequence (excluding zero-valued timested). By default, the 
            value is `-1` meaning that the pattern length is identical to the
            input length, and there are no zeroed timesteps. For other values, 
            the input sequences will be set to zero for the last :math:`t` 
            timesteps, where :math:`t > pat\_len`.
            Therefore, the input sequence lengths remain the same, but the
            actual duration of the patterns is reduced. This manipulation is 
            useful to decouple sequence length and memory requirement for 
            analysis.

            Note:
                We define the number of timesteps that are not zero, and
                therefore for values different than ``-1`` with the current
                implementation we will obtain patterns of identical length
                (but different input sequence length).
        scatter_pattern (bool): Option only compatible with ``pat_len != -1``.
            If activated, the pattern is not concentrated at the beginning of
            the input sequence. Instead, the whole input sequence will be filled
            with a random pattern (i.e., no padding is used) but only a fixed
            and random (see option ``rseed_scatter``)  number of timesteps from
            the input sequence are considered to create an output sequence of
            length ``pat_len``.
        permute_width (boolean, optional): If enabled, the generated pattern
            will be permuted along the width axis.
        permute_time (boolean, optional): If enabled, the generated pattern
            will be permuted along the temporal axis.
        permute_xor (bool): Only applicable if ``permute_width`` or
            ``permute_time`` is ``True``. If ``True``, the permuted and
            unpermuted output pattern will be combined to a new output pattern
            via a logical xor operation.
        permute_xor_iter (int): Only applicable if ``permute_xor`` is set.
            If ``True``, the internal permutation is applied iteratively and
            XOR-ed with the previous target output to obtain a final target
            output.
        permute_xor_separate (bool): Only applicable if ``permute_xor`` is set
            and ``permute_xor_iter > 1``. If ``True``, a separate permutation
            matrix is used per iteration described by ``permute_xor_iter``.
            In this case, we the input pattern is ``permute_xor_iter`` times
            permuted via a separate permutation matrix and the resulting
            patterns are sequentially XOR-ed with the original input pattern.

            Hence, this can be viewed as follows: ``permute_xor_iter`` random
            input pixels are assigned to each output pattern pixel. This
            output pattern pixel will be ``1`` if and only if the number of ones
            in those input pixels is odd.
        rseed (int, optional): If ``None``, the current random state of numpy 
            is used to generate the data. Otherwise, a new random state with the
            given seed is generated.
        rseed_permute (int, optional): Random seed for performing permutations 
            of the copy patterns. Only used if option ``permute_width`` or 
            ``permute_time`` are activated. If ``None``, the current random 
            state of numpy is used to generate the data. Otherwise, a new random
            state with the given seed is generated.
        rseed_scatter (int, optional): See option ``rseed``. Random seed for
            determining which timesteps of the input sequence to use for the
            output pattern if option ``scatter_pattern`` is activated.
        random_pad (boolean, optional): If activated, the truncated part of the
            input will be left as a random pattern, and not padded to zeros.
            Note that the loss computation is unaffected by this option.
    """
    def __init__(self, min_input_len, max_input_len, seq_width=7, num_train=100,
                 num_test=100, num_val=None, pat_len=-1, scatter_pattern=False,
                 permute_width=False, permute_time=False, permute_xor=False,
                 permute_xor_iter=1, permute_xor_separate=False, rseed=None,
                 rseed_permute=None, rseed_scatter=None, random_pad=False):
        super().__init__()

        # set random state
        if rseed is not None:
            self._rstate = np.random.RandomState(rseed)
        else:
            self._rstate = np.random

        self._permute_width = permute_width
        self._permute_time = permute_time
        self._permute_xor = permute_xor
        self._permute_xor_iter = permute_xor_iter
        self._permute_xor_separate = permute_xor_separate
        self._rseed_permute = rseed_permute
        self._random_pad = random_pad

        if permute_width or permute_time:
            # set random state for permutations
            if rseed_permute is not None:
                rstate_permute = np.random.RandomState(rseed_permute)
            else:
                rstate_permute = np.random

            # Permute not implemented for sequences of varying length.
            assert min_input_len == max_input_len
        elif permute_xor:
            warn('Option "permute_xor" has no effect if no ' +
                 'permutations are applied.')

        # We define the permutations in terms of a permutation vector X,
        # which is applied to flattened arrays, which are then returned to 
        # their original shapes.
        # Note that right now the code is not optimal, as we will loop over
        # all samples to apply permutations, even in the case when no actual
        # permutations are occurring. This is implemented like this for sanity
        # checks but should be fixed in the future.
        if pat_len == -1:
            pat_len_perm = min_input_len
        else:
            pat_len_perm = pat_len
            # Note we don't overwrite pat_len since for the non permutation
            # cases, we want it to be different for each sample.

        self.permutation_ = None
        if (permute_width or permute_time) and permute_xor and \
                permute_xor_separate:
            self.permutation_ = []
            for _ in range(permute_xor_iter):
                self.permutation_.append(CopyTask.create_permutation_matrix( \
                    permute_time, permute_width, pat_len_perm, seq_width,
                    rstate_permute))
        elif permute_width or permute_time:
            self.permutation_ = CopyTask.create_permutation_matrix( \
                permute_time, permute_width, pat_len_perm, seq_width,
                rstate_permute)

        out_pat_steps = None
        if scatter_pattern:
            if pat_len == -1 or pat_len > min_input_len:
                raise ValueError('Option "pat_len=%d" invalid when ' % pat_len +
                    'activating "scatter_pattern".')
            if min_input_len != max_input_len:
                warn('Option "scatter_pattern" only considers timesteps ' +
                     'below "min_input_len".')
            rstate_scatter = np.random
            if rseed_scatter is not None:
                rstate_scatter = np.random.RandomState(rseed_scatter)

            # Select timesteps to be used from the input to create the output
            # pattern.
            out_pat_steps = np.sort(rstate_scatter.choice(\
                np.arange(min_input_len), pat_len, replace=False))

        # Specify internal data structure.
        # Note, it's not a typical classification dataset, since it consits of
        # `seq_width` many independent binary classification decisions per
        # timestep.
        self._data['classification'] = False
        self._data['is_one_hot'] = False
        self._data['sequence'] = True
        # Note, there will be an extra channel for the stop bit in each input
        # sample.
        self._data['in_shape'] = [seq_width + 1]
        self._data['out_shape'] = [seq_width]
        self._data['train_inds'] = np.arange(num_train)
        self._data['test_inds'] = np.arange(num_train, num_train + num_test)

        self._data['avg_input_length'] = (min_input_len + max_input_len) / 2
        self._data['min_input_len'] = min_input_len
        self._data['max_input_len'] = max_input_len
        self._data['pat_len'] = pat_len
        self._data['scatter_pattern'] = scatter_pattern
        self._data['scatter_steps'] = out_pat_steps

        # Note that the number of timesteps in x is two times the maximum 
        # input length (one for presenting it, and one for producing it
        # as an output) plus a delay of 1 timestep between the two.
        # If a certain number of timesteps is zeroed out in the input pattern,
        # then the target pattern will only correspond to the non-zeroed part
        # Note that this step is important when computing the accuracy and loss 
        # on the last timestep, since otherwise it will be computed after a long 
        # silence.
        if pat_len == -1:
            self._data['seq_len'] = self._data['max_input_len'] * 2 + 1
        else:
            self._data['seq_len'] = self._data['max_input_len'] + 1 + \
                pat_len

        if num_val is not None:
            n_start = num_train + num_test
            self._data['val_inds'] = np.arange(n_start, n_start + num_val)

        # get train and test data
        train_x, train_y, train_l, train_nz = \
            self._generate_trial_samples(num_train)
        test_x, test_y, test_l, test_nz = self._generate_trial_samples(num_test)

        # Create validation data if requested.
        if num_val is not None:
            val_x, val_y, val_l, val_nz = self._generate_trial_samples(num_val)

            in_data = np.vstack([train_x, test_x, val_x])
            out_data = np.vstack([train_y, test_y, val_y])
            seq_lengths = np.concatenate([train_l, test_l, val_l])
            zeroed_ts = np.concatenate([train_nz, test_nz, val_nz])
        else:
            in_data = np.vstack([train_x, test_x])
            out_data = np.vstack([train_y, test_y])
            seq_lengths = np.concatenate([train_l, test_l])
            zeroed_ts = np.concatenate([train_nz, test_nz])

        self._data['in_data'] = in_data
        self._data['out_data'] = out_data

        # Note, inputs and outputs have the same lengths in this dataset.
        self._data['in_seq_lengths'] = seq_lengths
        self._data['out_seq_lengths'] = seq_lengths
        self._data['zeroed_ts'] = zeroed_ts

    def _generate_trial_samples(self, n_samples):
        """Generate a certain number of trials.

        Args:
            n_samples (int): The number of desired samples.

        Returns:
            (tuple): Tuple containing:

            - **x**: Matrix of trial inputs of shape
              ``[n_samples, in_size*time_steps]``.
            - **y**: Matrix of trial targets of shape
              ``[n_samples, in_size*time_steps]``.
            - **lengths**: Vector of trial lengths.
            - **zeroed_ts**: Vector of number of zeroed out timesteps.
        """
        lengths = []
        zeroed_ts = []

        in_size = self._data['in_shape'][0] - 1
        seq_len = self._data['max_input_len'] * 2 + 1
        scatter_pattern = self._data['scatter_pattern']

        # Randomly create a binary matrix.
        x = self._rstate.rand(self._data['seq_len'], n_samples, in_size)
        x = np.where(x>0.5, 1., 0)

        # Define y as the copy of x without the delimiter flag, delayed by the
        # maximum input length + 1.
        # Note that x and y have different dimensions in this step. Because
        # the output pattern might need to be manipulated differently for
        # different tasks in a way that requires having the entire input
        # sequence (i.e. when scattering patterns), we define y to be twice the
        # length of the maximum sample length plus one, and we will cut it if
        # necessary in :meth:`output_to_torch_tensor`.
        y = np.zeros((seq_len, n_samples, in_size))

        # Cut each sample to a specific length randomly sampled around average
        # input length for this task.
        flag = np.zeros((self._data['seq_len'], n_samples, 1))
        for i in range(n_samples):
            if self._data['min_input_len'] != self._data['max_input_len']:
                sample_input_len = self._rstate.randint(\
                    self._data['min_input_len'], self._data['max_input_len'])
            else:
                sample_input_len = self._data['min_input_len']

            # If needed, zero-out the last timesteps of the input, as 
            # specified by the value `pat_len`.
            pat_len = self._data['pat_len']
            if pat_len == -1 or pat_len > sample_input_len:
                pat_len = sample_input_len
            num_zeroed_ts = sample_input_len-pat_len
            if scatter_pattern:
                x[sample_input_len:, i, :] = 0
            else:
                if not self._random_pad:
                    # Only pad the rest of the pattern to zero if the option
                    # "random_pad" has not been activated.
                    x[pat_len:, i, :] *= 0

            # Copy the content of x into y, after a delay of 1 timestep.
            y[sample_input_len + 1: sample_input_len*2 + 1, i, :] = \
                x[:sample_input_len, i, :]

            # Add flag indicator to easily recover sequence lengths afterwards.
            y[sample_input_len, i, :] = -1 # FIXME it's ugly

            # Add the sequence stop flag.
            flag[sample_input_len, i, :] = 1.

            lengths.append(sample_input_len + 1 + pat_len)
            zeroed_ts.append(num_zeroed_ts)

        # Concatenate the flag to the inputs.
        x = np.concatenate((x, flag), axis=2)

        # Reshape x and y to fit the dataset class.
        x = self._flatten_array(x, ts_dim_first=True)
        y = self._flatten_array(y, ts_dim_first=True)

        return x, y, np.array(lengths), np.array(zeroed_ts)

    def output_to_torch_tensor(self, *args, **kwargs):
        """Similar to method :meth:`input_to_torch_tensor`, just for dataset
        outputs.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset.output_to_torch_tensor`.

        Returns:
            (torch.Tensor): The given input ``y`` as PyTorch tensor. It has
            dimensions ``[T, B, *out_shape]``, where ``T`` is the number of time
            steps (see attribute :attr:`max_num_ts_out`), ``B`` is the batch
            size and ``out_shape`` refers to the output feature shape, see
            :attr:`data.dataset.Dataset.out_shape`.
        """
        y = super().output_to_torch_tensor(*args, **kwargs)
        n_samples = y.shape[1]

        scatter_pattern = self._data['scatter_pattern']
        scatter_steps = self._data['scatter_steps']
        input_len = self._data['min_input_len']

        # Scatter the contents of x if needed.
        if scatter_pattern:
            for i in range(n_samples):
                # Get the correct sample length.
                sample_input_len = y[:,i,:][:,0].tolist().index(-1)

                # Get the sample pattern length.
                pat_len = self._data['pat_len']
                if pat_len == -1 or pat_len > sample_input_len:
                    pat_len = sample_input_len

                # That's just the complete input pattern.
                y_pattern_original = y[sample_input_len + 1:, i, :]

                # Note, the rest is cut away below.
                y[sample_input_len + 1:sample_input_len + 1 + pat_len, i, :] = \
                    y_pattern_original[scatter_steps]

        # Delete the stop flag indicator.
        y[y==-1] = 0.

        # Cut irrelevant timesteps for the loss from the output sequences.
        y = y[:self._data['seq_len'], :, :]
        seq_len, n_samples, in_size = y.shape
        pat_len = self._data['pat_len']

        if self._permute_xor and self.permutation is not None:
            if pat_len == -1:
                pat_len = input_len
            assert seq_len == input_len + 1 + pat_len

            ### Apply permutation & xor iteratively
            y_pattern = y[input_len+1:, :, :].cpu().numpy()
            y_pattern_orig = np.array(y_pattern)
            y_pattern_perm = np.zeros_like(y_pattern)
            for p in range(self._permute_xor_iter):
                curr_perm = self.permutation
                if self._permute_xor_separate:
                    curr_perm = self.permutation[p]

                # get permutation for all samples
                for i in range(n_samples):
                    # What pattern to permute:
                    if self._permute_xor_separate: # The original pattern.
                        sample_pattern = np.array(y_pattern_orig[:, i, :])
                    else: # The result of the last XOR operation.
                        sample_pattern = copy.deepcopy(y_pattern[:, i, :])
                    sample_pattern = sample_pattern.flatten()[curr_perm]

                    y_pattern_perm[:, i, :] = sample_pattern.reshape(\
                        (pat_len, in_size))
                # update y_pattern by applying xor operation 
                y_pattern = np.logical_xor( \
                     y_pattern_perm, y_pattern).astype(float)
            y[input_len+1:, :, :] = torch.tensor(y_pattern)

        elif self.permutation is not None:
            if pat_len == -1:
                pat_len = input_len
            assert seq_len == input_len + 1 + pat_len

            ### Apply the permutation.
            # Note that the permutation can be applied either in the width or 
            # time dimensions, or both. For the first case, at a given timestep 
            # :math:`t` the binary vector of activations is permuted for the 
            # targets. For the second case, given a certain input unit :math:`i` 
            # the binary vector of activations is permuted across time, only for 
            # the timesteps where the pattern is being reconstructed (not during
            # input). If both are active, the permutation is performed 
            # independently for each point in both dimensions.
            y_pattern = y[input_len+1:, :, :]
            y_pattern_perm = np.zeros_like(y_pattern.cpu())
            for i in range(n_samples):
                sample_pattern = copy.deepcopy(y_pattern[:, i, :].cpu())
                sample_pattern = sample_pattern.flatten()[self.permutation]
                y_pattern_perm[:, i, :] = sample_pattern.reshape(\
                    (pat_len, in_size))

            # Add the copied or permuted pattern to the target sequences.
            y[input_len+1:, :, :] = torch.tensor(y_pattern_perm)

        return y

    def get_out_pattern_bounds(self, sample_ids):
        """Get the start time step and length of the output pattern within
        the sequence.

        Note, input sequences may have varying length (even though they are 
        padded to the same length). Assume we are considering a input of 
        length 7, meaning that the total sequence would have the length 
        15 = 7 + 1 + 7 (input pattern presentation, stop bit, output pattern 
        copying). In addition, assume that the maximum input length is 10 
        (hence, the maximum input length is 21 = 10 + 1 + 10).
        In this case, all sequences are padded to have length 21. For the sample
        in consideration (with input length 7), the output pattern sequence
        starts at index 8 and has a length of 7, or less, if the input contains
        some zeroed values. Hence, these two number would be returned for this 
        sample.

        Args:
            (....): See docstring of method
                :meth:`data.sequential_data.SequentialDataset.\
get_in_seq_lengths`.

        Returns:
            (tuple): Tuple containing:

            - **start_inds** (numpy.ndarray): 1D array with the same length as
              ``sample_ids``, which contains the start index for output pattern
              in a given sample.
            - **lengths** (numpy.ndarray): 1D array containing the lengths of
              the pattern per given sample.
        """
        seq_lengths = self.get_out_seq_lengths(sample_ids)

        # Get the actual zeroeing out per sample (since some might be shorter
        # than the number of timesteps to be masked out, it might be that the
        # actual number of timesteps that have been zeroed is smaller).
        zeroed_ts = self.get_zeroed_ts(sample_ids)

        # To compute the start index of target pattern.
        out_pat_start_inds = (seq_lengths + zeroed_ts)// 2 + 1

        # The actual pattern length after zeroing part of the pattern.
        pat_len = (seq_lengths + zeroed_ts)// 2 - zeroed_ts
        assert np.all(np.equal(seq_lengths, out_pat_start_inds + pat_len))

        return out_pat_start_inds, pat_len

    def get_zeroed_ts(self, sample_ids):
        """Get the number of zeroed timesteps in each input pattern.

        Note, if ``scatter_pattern`` was activated in the constructor, then this
        number does not refer to the number of padded steps in the input
        sequence but rather to the number of unused steps in the input sequence.
        However, those unused steps will still contain random patterns.

        Args:
            (....): See docstring of method :meth:`get_in_seq_lengths`.

        Returns:
            (numpy.ndarray): A 1D numpy array.
        """
        return self._data['zeroed_ts'][sample_ids]

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'Copy'

    def _plot_config(self, inputs, outputs=None, predictions=None):
        """Defines properties, used by the method :meth:`plot_samples`.

        This method can be overwritten, if these configs need to be different
        for a certain dataset.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset._plot_config`.

        Returns:
            (dict): A dictionary with the plot configs.
        """
        plot_configs = dict()
        plot_configs['outer_wspace'] = 0.4
        plot_configs['outer_hspace'] = 0.5
        plot_configs['inner_hspace'] = 0.2
        plot_configs['inner_wspace'] = 0.8
        plot_configs['num_inner_rows'] = 1
        if outputs is not None:
            plot_configs['num_inner_rows'] += 1
        if predictions is not None:
            plot_configs['num_inner_rows'] += 1
        plot_configs['num_inner_cols'] = 1
        plot_configs['num_inner_plots'] = plot_configs['num_inner_rows']

        return plot_configs

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None, equalize_size=False,
                     mask_predictions=False, sample_ids=None):
        """Add a custom sample plot to the given Axes object.
        Note, this method is called by the :meth:`plot_samples` method.

        Note, that the number of inner subplots is configured via the method:
        :meth:`_plot_config`.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset._plot_sample`.
            inputs: 2D Array with dimensions 
                [1, self._data['in_shape']*self._data['num_timesteps'].
            outputs: 1D Array with dimensions 
                [(self._data['in_shape']-1)*self._data['num_timesteps'].
            equalize_size (bool): Equalize the size of the input and output 
                tensors by adding an empty row to the outputs (as it has one
                less unit).
            mask_predictions (bool): Mask given predictions to only show
                predicted values during output pattern presentation time.
            sample_ids (numpy.ndarray): See option ``sample_ids`` of method
                :meth:`get_out_pattern_bounds`. Only required if
                ``mask_predictions`` is ``True``.
        """
        # Reshape the values to uncouple inputs and time.
        x = self._flatten_array(inputs, ts_dim_first=True, reverse=True,
                                feature_shape=self.in_shape)
        pdata = [x]
        plabel = ['inputs']
        if outputs is not None:
            t = self._flatten_array(outputs, ts_dim_first=True, reverse=True,
                                    feature_shape=self.out_shape)
            if equalize_size:
                t = np.concatenate([t, np.zeros((t.shape[0], t.shape[1], 1))],
                                    axis=2)
            pdata.append(t)
            plabel.append('outputs')
        if predictions is not None:
            y = self._flatten_array(predictions, ts_dim_first=True,
                                    reverse=True, feature_shape=self.out_shape)
            if equalize_size:
                y = np.concatenate([y, np.zeros((y.shape[0], y.shape[1], 1))],
                                    axis=2)

            if mask_predictions:
                assert sample_ids is not None
                ss, sl = self.get_out_pattern_bounds(sample_ids[[ind]])
                for sind in range(y.shape[1]):
                    y[:ss[sind], sind, :] = 0
                    y[(ss[sind]+sl[sind]):, sind, :] = 0

            pdata.append(y)
            plabel.append('predictions')

        for i, d in enumerate(pdata):
            ax = plt.Subplot(fig, inner_grid[i])
            # Note, we can't use `set_axis_off`, if we wanna keep the y-label.
            ax.set_ylabel(plabel[i])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(d.squeeze().transpose())
            fig.add_subplot(ax)

    def __str__(self):
        """Print major characteristics of the current dataset."""
        return 'Data handler for the copy task dataset.\n' + \
               'Sequence width: %i \n' % self._data['in_shape'][0] + \
               'Average input length: %.1f \n' % \
                    self._data['avg_input_length'] + \
               'Max. total number of timesteps: %i. \n' % \
               self._data['seq_len'] + \
               'Dataset contains %d training, %d validation and %d test ' % \
               (self.num_train_samples, self.num_val_samples,
                self.num_test_samples) + 'samples.'

    @property
    def permutation(self):
        """Getter for attribute :attr:`permutation_`"""
        return self.permutation_

    @staticmethod
    def create_permutation_matrix(permute_time, permute_width, pat_len_perm,
                                  seq_width, rstate_permute):
        """Create a permutation matrix."""
        P = None
        if permute_width or permute_time:
            P = np.arange(pat_len_perm*seq_width)
            if permute_width and permute_time:
                P = rstate_permute.permutation(P)
            elif permute_time:
                P = P.reshape((pat_len_perm, seq_width))
                P = rstate_permute.permutation(P)
                P = P.flatten()
            elif permute_width:
                P = P.reshape((pat_len_perm, seq_width))
                P = rstate_permute.permutation(P.T).T
                P = P.flatten()

        return P

if __name__ == '__main__':
    pass


