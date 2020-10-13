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
# @title           :sequential/copy/gather_seeds_copy.py
# @author          :mc
# @contact         :mariacer@ethz.ch
# @created         :05/05/2020
# @version         :1.0
# @python_version  :3.6.8
"""
Gather results of a copy task for different random seeds.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

from sequential import gather_random_seeds

if __name__=='__main__':

    # Define the dictionary with key-value pairs of config to be overwritten.
    forced_params ={
        'during_acc_criterion': '-1'
    }

    gather_random_seeds.run('sequential.copy.hp_search_copy',
        forced_params=forced_params)
