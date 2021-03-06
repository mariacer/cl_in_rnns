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
# @title           :sequential/audioset/gather_seeds_audioset.py
# @author          :mc
# @contact         :mariacer@ethz.ch
# @created         :05/05/2020
# @version         :1.0
# @python_version  :3.6.8
"""
Gather results of SMNIST for different random seeds.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

from sequential import gather_random_seeds
from sequential.audioset.train_args_audioset import parse_cmd_arguments

if __name__=='__main__':

    config = None
    argv = {}
    ## To design a specific config without loading a store one do as follows: 
    # config = parse_cmd_arguments(mode='split_audioset', default=True)
    # argv = {'multi_head':True, 'num_tasks':10}
    for key, value in argv.items():
       setattr(config, key, value)

    # Define items of the config file that might need to be ignored when 
    # launching the hpsearch code to gather random seeds.
    ignore_kwds = ['mode']
    forced_params = {
        'during_acc_criterion': '-1'
    }

    gather_random_seeds.run('sequential.audioset.hp_search_audioset', \
        ignore_kwds=ignore_kwds, forced_params=forced_params, config=config)
