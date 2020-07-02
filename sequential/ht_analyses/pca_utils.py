#!/usr/bin/env python3
# Copyright 2020 Maria Cervera

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
# @title           :sequential/ht_analyses/pca_utils.py
# @author          :mc
# @contact         :mariacer@ethz.ch
# @created         :26/06/2020
# @version         :1.0
# @python_version  :3.6.8
"""
Utils for making PCA analyses on the hidden state of RNNs
---------------------------------------------------------
"""
import numpy as np
import torch
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA

def compute_pca(x, n_pcs=None, n_samples=-1):
    """Compute PCA components and explained variance ratios.

    Args:
        x (torch.Tensor): The data. It has dimensions
            ``[num_samples, num_features]``.
        n_pcs (int): The number of components to use.
        n_samples (int): The number of samples to be used.

    Returns:
        (tuple): Tuple containing:

        - **pca**: The PCA object.
        - **expl_var** (np.array): The explained variance ratios for the PCs.
    """

    # Randomly chose a subset of the samples if needed.
    if n_samples != -1:
        order = torch.randperm(x.shape[0])
        x = x[order, :]
        x = x[:n_samples, :]

    # Make sure that we have more samples than dimensions.
    assert x.shape[0] > x.shape[1]

    # Compute principal components
    pca = PCA(n_components=n_pcs)
    pca.fit_transform(x)
    expl_var = pca.explained_variance_ratio_

    return pca, expl_var


def compute_kpca(x,  n_pcs=None, n_samples=-1, max_num_samples=2000):
    """Compute Kernel PCA components and get eigenvalues.

    Args:
        x (torch.Tensor): The data. It has dimensions
            ``[num_samples, num_features]``.
        n_pcs (int): The number of components to use.
        n_samples (int): The number of samples to be used.
        max_num_samples (int): The maximum number of samples to be used. This
        	is used to limit the duration of the computation.

    Returns:
        (tuple): Tuple containing:

        - **pca**: The PCA object.
        - **eigenvalues** (np.array): The normalized eigenvalues (such that 
           they sum up to one).
    """

    # Randomly chose a subset of the samples if needed.
    if n_samples != -1:
        order = torch.randperm(x.shape[0])
        x = x[order, :]
        x = x[:n_samples, :]

    # Make sure that we have more samples than dimensions.
    assert x.shape[0] > x.shape[1]

    # Compute kernel principal components using radial basis functions.
    pca = KernelPCA(kernel='rbf', gamma=15, n_components=n_pcs, \
        fit_inverse_transform=True)

    # Doing Kernel PCA on the entire dataset is too computationally expensive,
    # so we randomly select some samples.
    indices = np.random.permutation(x.shape[0])[:max_num_samples]
    pca.fit_transform(x[indices, :])

    # We only look at as many eigenvalues as number of features (hidden neurons)
    eigenvalues = pca.lambdas_[:x.shape[1]]

    # Normalize their values.
    eigenvalues /= np.sum(eigenvalues)

    return pca, eigenvalues


def get_expl_var(x, do_kernel_pca=False, n_samples=-1, timesteps=None,
        stop_bit=None):
    """Get the dimensionality of the hidden activations on the entire sequence.

    Given a set of hidden activations, this function computes the principal 
    components, where different sequences and different timesteps are all 
    treated as different samples, and the dimensions space is determined by 
    the number of hidden neurons. We define the number of dimensions of the
    hidden space as the number of principal components needed to explain a
    certain variance :math:`p_{var}` of the data.

    Note: 
        For LSTMs, for the moment we perform the analysis on the external hidden 
        states (`h_t` and not `c_t`). For vanilla RNNs, we perform the analysis 
        on the internal hidden states.

    Args:
        x (torch.Tensor): The test hidden activations. It has dimensions 
            ``[seq_length, batch_size*num_tasks, num_hidden]`` or 
            ``[batch_size*num_tasks, num_hidden]``.
        do_kernel_pca (bool, optional): If True, kernel PCA will also be used
            to copmute the number of hidden dimensions.
        n_samples (int): The number of samples to be used.
        timestep (int, optional): The timestep to use for the analyses. If None,
            data from all timesteps is pulled together.
        stop_bit (int, optional): The timestep of the stop bit presentation.
        timesteps (str, optional): The description of which timesteps to use.
            If None, all timesteps are used, if `input` only the timesteps
            before the stop bit are used, if `output` only the timesteps after
            the stop bit are used, if `stop` only the stop bit timestep is used.
        stop_bit (int, optional): The timestep of the stop bit presentation.

    Returns:
        (tuple): Tuple containing:

        - **cum_expl_vars** (list): The percentage explained variance by the 
            principal components.
        - **cum_kexpl_vars** (list): The percentage explained variance by the 
            principal components.when doing kernel PCA.

    """
    x = select_timesteps(x, timesteps=timesteps, stop_bit=stop_bit)

    # Compute the cumulative explained variances of the PCs.
    _, expl_vars = compute_pca(x, n_samples=n_samples)
    cum_expl_vars = np.cumsum(expl_vars)

    # Results on kernel PCA.
    cum_kexpl_vars = None
    if do_kernel_pca:
        _, kexpl_vars = compute_kpca(x, n_samples=n_samples)
        cum_kexpl_vars = np.cumsum(kexpl_vars)

    return cum_expl_vars, cum_kexpl_vars


def get_expl_var_per_ts(x, n_samples=-1, do_kernel_pca=False):
    """Get the dimensionality of the hidden activations per timestep.

    Here we apply the function :func:`get_expl_var` separately for each
    timestep.

    Args:
        x (torch.Tensor): The test hidden activations on all tasks. It has
            dimensions ``[seq_length, batch_size*num_tasks, num_hidden]``.
        n_samples (int): The number of samples to be used.
        do_kernel_pca (bool, optional): If True, kernel PCA will also be used
            to copmute the number of hidden dimensions.

    Returns:
        (tuple): Tuple containing:

        - **expl_vars** (list): The number of principal components needed to 
            explain :math:`p_{var}` % variance of the data in each timestep.
        - **kexpl_vars** (list): The number of principal components needed to 
            explain :math:`p_{var}` % variance of the data in each timestep
            doing kernel PCA.
    
    """
    seq_length = x.shape[0]
    expl_vars = []
    kexpl_vars = []
    for t in range(seq_length):
        expl_var, kexpl_var = get_expl_var(x[t, :, :], 
            do_kernel_pca=do_kernel_pca, n_samples=n_samples)
        expl_vars.append(expl_var)
        kexpl_vars.append(kexpl_var)

    if not do_kernel_pca:
        kexpl_vars = None

    return expl_vars, kexpl_vars


def get_num_dimensions(expl_vars, p_var=0.75):
    """Get the number of dimensions needed to explain some ratio of variance.

    Args:
        expl_vars (list): The list of explained variance ratios.
        p_var (float): The percentage of variance to be explained.

    Returns:
        (int): The number of dimensions.
    """
    if expl_vars is None:
        return None
    else:
        return next(x[0] for x in enumerate(expl_vars) if x[1] >= p_var)

def select_timesteps(x, timesteps=None, stop_bit=None, max_num_samples=-1):
    """Select the data to be given for the PCA analysis.

    Args:
        x (torch.Tensor): The data. It has dimensions 
            ``[seq_length, batch_size*num_tasks, num_hidden]`` or 
            ``[batch_size*num_tasks, num_hidden]``.
        timesteps (str, optional): The description of which timesteps to use.
            If None, all timesteps are used, if `input` only the timesteps
            before the stop bit are used, if `output` only the timesteps after
            the stop bit are used, if `stop` only the stop bit timestep is used.
        stop_bit (int, optional): The timestep of the stop bit presentation.
        max_num_samples (int, optional): The maximum number of samples to use.

    Returns:
        (torch.Tensor): The selected data. It has dimensions 
            ``[num_timesteps*batch_size*num_tasks, num_hidden]``

    """
    # If `x` is only two-dimensional, then it already consists of a single ts.
    if not len(x.shape) == 2:
        if timesteps is None or timesteps=='all':
            # Treat dimensions other than hidden neurons as different samples.
            x = x.view(-1, x.shape[-1])
        elif timesteps == 'stop':
            x = x[stop_bit, :, :]
        elif timesteps == 'input':
            x = x[:stop_bit, :, :].view(-1, x.shape[-1])
        elif timesteps == 'output':
            x = x[stop_bit+1:, :, :].view(-1, x.shape[-1])

    if max_num_samples == -1:
        max_num_samples = x.shape[0]

    order = torch.randperm(x.shape[0])
    x = x[order, :]
    x = x[:max_num_samples, :]

    return x

def get_expl_var_across_tasks(all_act, n_samples=-1, do_kernel_pca=False,
        n_pcs_step=10, timesteps=None, stop_bit=None): 
    """Study the subspace overlap between tasks.

    Project the hidden activations of tasks 2, 3 ... onto the principal 
    components for task 1, and return the explained variance.
    Note that, currently, all timesteps are pulled together as different 
    samples for the analysis.

    Args:
        all_act (list): The list of hidden activations in each task.
        n_samples (int): The number of samples to be used.
        do_kernel_pca (bool, optional): If True, kernel PCA will also be used
            to copmute the number of hidden dimensions.
        n_pcs_step (int): The step in number of pcs to be considered.
        timesteps (str, optional): The description of which timesteps to use.
            If None, all timesteps are used, if `input` only the timesteps
            before the stop bit are used, if `output` only the timesteps after
            the stop bit are used, if `stop` only the stop bit timestep is used.
        stop_bit (int, optional): The timestep of the stop bit presentation.

    Returns:
        (tuple): Tuple containing:

        - **expl_var** (list): The variance explained by each principal 
            component of task 1 when projecting each of the subsequent tasks. 
            It has length number of tasks.
        - **kexpl_var** (list): Same as `expl_var` but for kernel PCA.
        - **n_pcs_considered** (list): The number of pcs considered. We can't
            afford exploring the addition of 1 pc at a time, because this leads
            to a lot of computation.
    """
    if do_kernel_pca == True:
        raise ValueError('Implemented, but causing trouble.')

    num_tasks = len(all_act)
    num_hidden = all_act[0].shape[-1]
    expl_var = []
    kexpl_var = []
    n_pcs_considered = np.arange(0, num_hidden, n_pcs_step)

    ### Perform PCA on the hidden state of the first task.
    first_task_act = all_act[0] 
    first_task_act = select_timesteps(first_task_act, timesteps=timesteps,
        stop_bit=stop_bit)

    # Compute pca.
    pca, expl_var = compute_pca(first_task_act, n_samples=n_samples) 
    expl_var = [np.cumsum(expl_var).tolist()]

    # Compute kernel pca.
    kexpl_var = None
    if do_kernel_pca:
        kpca, kexpl_var = compute_kpca(first_task_act, n_samples=n_samples) 
        kexpl_var = [np.cumsum(kexpl_var).tolist()]

    # Return only values for the considered pcs.
    expl_var = [[expl_var[0][i] for i in n_pcs_considered]]
    if do_kernel_pca:
        kexpl_var = [[kexpl_var[0][i] for i in n_pcs_considered]]

    if len(all_act) == 1:
        return expl_var, kexpl_var, n_pcs_considered

    # Because we are accumulating samples from different tasks, we set a common
    # maximum number of samples so that the PCA results are comparable.
    max_num_samples = np.min([5000, first_task_act.shape[0]])

    ### Concatenate the hidden states for all subsequent tasks.
    subsequent_tasks_act = torch.tensor(())
    for task_id, act in zip(range(1, num_tasks), all_act[1:]):

        acts = select_timesteps(act, timesteps=timesteps, stop_bit=stop_bit)

        # Accumulate hidden activations of all but first task.
        subsequent_tasks_act = torch.cat((subsequent_tasks_act, acts), dim=0)

        # Select a random subset of the data to have comparable results.
        order = torch.randperm(subsequent_tasks_act.shape[0])[:max_num_samples]
        subsequent_tasks_act_random = subsequent_tasks_act[order, :]

        # Normalize the data.
        subsequent_tasks_norm = torch.tensor(StandardScaler(with_std=False \
            ).fit_transform(subsequent_tasks_act_random), dtype=torch.float32)

        expl_var_aux = []
        kexpl_var_aux = []
        # Make the analysis with a varying number of pcs used.
        for n_pcs in n_pcs_considered[1:]:

            ### Compute pca.
            # We recompute pca every time instead of just chopping the results
            # above for the desired number of components because when doing
            # kernel PCA, we cannot access its components.
            pca, _ = compute_pca(first_task_act, n_samples=n_samples, \
                n_pcs=n_pcs)

            # Compute kernel pca.
            kexpl_var = None
            if do_kernel_pca:
                kpca, _ = compute_kpca(first_task_act, n_samples=n_samples, 
                    n_pcs=n_pcs)

            ### Project onto the pcs of the first task.
            # Equivalent to 
            # `torch.matmul(subsequent_tasks_norm, pcs.T[:, :n_pcs])`.
            # Note that pca.components_ has dimensions (n_components,n_features) 
            # so we would need to transpose it. 
            subsequent_tasks_projected = pca.transform(subsequent_tasks_norm)
            if do_kernel_pca:
                ksubsequent_tasks_projected = kpca.transform(\
                    subsequent_tasks_norm)

            ### Project back to the original space. 
            # Equivalent to 
            # `torch.matmul(subsequent_tasks_projected, pcs[:n_pcs, :])`
            subsequent_tasks_projected_back = pca.inverse_transform(\
                subsequent_tasks_projected)
            if do_kernel_pca:
                ksubsequent_tasks_projected_back = kpca.inverse_transform(\
                    subsequent_tasks_projected)

            ### Obtain the explained variances.
            # Compute the residual of the reconstruction.
            residual = subsequent_tasks_norm - subsequent_tasks_projected_back
            if do_kernel_pca:
                kresidual = subsequent_tasks_norm - \
                    ksubsequent_tasks_projected_back
            # Compute the explained variance.
            ve = 1 - LA.norm(np.array(residual), 'fro')**2 / \
                LA.norm(np.array(subsequent_tasks_norm), 'fro')**2
            if do_kernel_pca:
                kve = 1 - LA.norm(np.array(kresidual), 'fro')**2 / \
                    LA.norm(np.array(subsequent_tasks_norm), 'fro')**2
            expl_var_aux.append(ve)
            if do_kernel_pca:
                kexpl_var_aux.append(kve)
        expl_var_aux.append(1.)
        if do_kernel_pca:
            kexpl_var_aux.append(1.)

        expl_var.append(expl_var_aux)
        if do_kernel_pca:
            kexpl_var.append(kexpl_var_aux)

    return expl_var, kexpl_var, n_pcs_considered


def get_expl_var_across_tasks_cum(all_act, n_samples=-1, do_kernel_pca=False,
        n_pcs_step=10, timesteps=None, stop_bit=None): 
    """Study the cumulative subspace overlap between tasks.

    Project the hidden activations of tasks 2, 3 ... onto the principal 
    components for task 1, and return the explained variance.
    Note that, currently, all timesteps are pulled together as different 
    samples for the analysis.

    Args:
        all_act (list): The list of hidden activations in each task.
        n_samples (int): The number of samples to be used.
        do_kernel_pca (bool, optional): If True, kernel PCA will also be used
            to copmute the number of hidden dimensions.
        n_pcs_step (int): The step in number of pcs to be considered.
        timesteps (str, optional): The description of which timesteps to use.
            If None, all timesteps are used, if `input` only the timesteps
            before the stop bit are used, if `output` only the timesteps after
            the stop bit are used, if `stop` only the stop bit timestep is used.
        stop_bit (int, optional): The timestep of the stop bit presentation.

    Returns:
        (tuple): Tuple containing:

        - **expl_var** (list): The variance explained by each principal 
            component of task 1 when projecting each of the subsequent tasks. 
            It has length number of tasks.
        - **kexpl_var** (list): Same as `expl_var` but for kernel PCA.
        - **n_pcs_considered** (list): The number of pcs considered. We can't
            afford exploring the addition of 1 pc at a time, because this leads
            to a lot of computation.
    """
    if do_kernel_pca == True:
        raise ValueError('Implemented, but causing trouble.')

    num_tasks = len(all_act)
    num_hidden = all_act[0].shape[-1]
    expl_var = []
    kexpl_var = []
    n_pcs_considered = np.arange(0, num_hidden, n_pcs_step)

    ### Concatenate the hidden states for all subsequent tasks.
    subsequent_tasks_act = torch.tensor(())
    expl_var = []
    kexpl_var = None
    for task_id, act in zip(range(num_tasks), all_act):

        if task_id == 0:
            # Because we are accumulating samples from many tasks, we set common
            # maximum number of samples so that the PCA results are comparable.
            max_num_samples = np.min([5000, act.shape[1]])

        acts = select_timesteps(act, timesteps=timesteps, stop_bit=stop_bit)

        # Accumulate hidden activations of all but first task.
        subsequent_tasks_act = torch.cat((subsequent_tasks_act, acts), dim=0)

        # Select a random subset of the data to have comparable results.
        order = torch.randperm(subsequent_tasks_act.shape[0])[:max_num_samples]
        X = subsequent_tasks_act[order, :]

        if task_id == 0:
            # Compute pca.
            pca, expl_var_aux = compute_pca(X, n_samples=n_samples) 
            expl_var_aux = np.cumsum(expl_var_aux).tolist()
            expl_var_aux = [expl_var_aux[i] for i in n_pcs_considered]
            expl_var.append(expl_var_aux)
            pca_old = pca

            # Compute kernel pca.
            if do_kernel_pca:
                kpca, kexpl_var = compute_kpca(X, n_samples=n_samples) 
                kexpl_var_aux = np.cumsum(kexpl_var_aux).tolist()
                kexpl_var_aux = [kexpl_var_aux[i] for i in n_pcs_considered]
                kexpl_var.append(kexpl_var_aux)
                kpca_old = kpca

        else:
            # Normalize the data.
            subsequent_tasks_norm = torch.tensor(StandardScaler(with_std=False \
                ).fit_transform(X), dtype=torch.float32)

            expl_var_aux = []
            kexpl_var_aux = []
            # Make the analysis with a varying number of pcs used.
            for n_pcs in n_pcs_considered[1:]:

                ### Compute pca.
                # We recompute pca every time instead of just chopping the results
                # above for the desired number of components because when doing
                # kernel PCA, we cannot access its components.
                pca, _ = compute_pca(X, n_samples=n_samples, n_pcs=n_pcs)

                # Compute kernel pca.
                kexpl_var = None
                if do_kernel_pca:
                    kpca, _ = compute_kpca(X, n_samples=n_samples, n_pcs=n_pcs)

                ### Project onto the pcs of the first task.
                # Equivalent to 
                # `torch.matmul(subsequent_tasks_norm, pcs.T[:, :n_pcs])`.
                # Note that pca.components_ has dimensions (n_components,n_features) 
                # so we would need to transpose it. 
                subsequent_tasks_projected = pca.transform(subsequent_tasks_norm)
                if do_kernel_pca:
                    ksubsequent_tasks_projected = kpca.transform(\
                        subsequent_tasks_norm)

                ### Project back to the original space. 
                # Equivalent to 
                # `torch.matmul(subsequent_tasks_projected, pcs[:n_pcs, :])`
                subsequent_tasks_projected_back = pca.inverse_transform(\
                    subsequent_tasks_projected)
                if do_kernel_pca:
                    ksubsequent_tasks_projected_back = kpca.inverse_transform(\
                        subsequent_tasks_projected)

                ### Obtain the explained variances.
                # Compute the residual of the reconstruction.
                residual = subsequent_tasks_norm - subsequent_tasks_projected_back
                if do_kernel_pca:
                    kresidual = subsequent_tasks_norm - \
                        ksubsequent_tasks_projected_back
                # Compute the explained variance.
                ve = 1 - LA.norm(np.array(residual), 'fro')**2 / \
                    LA.norm(np.array(subsequent_tasks_norm), 'fro')**2
                if do_kernel_pca:
                    kve = 1 - LA.norm(np.array(kresidual), 'fro')**2 / \
                        LA.norm(np.array(subsequent_tasks_norm), 'fro')**2
                expl_var_aux.append(ve)
                if do_kernel_pca:
                    kexpl_var_aux.append(kve)
            expl_var_aux.append(1.)
            if do_kernel_pca:
                kexpl_var_aux.append(1.)

            expl_var.append(expl_var_aux)
            if do_kernel_pca:
                kexpl_var.append(kexpl_var_aux)

        pca_old = pca
        if do_kernel_pca:
            kpca_old = kpca

    return expl_var, kexpl_var, n_pcs_considered

