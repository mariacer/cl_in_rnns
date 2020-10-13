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
# @title          :sequential/student_teacher/st_state_analysis.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :07/28/2020
# @version        :1.0
# @python_version :3.6.10
"""
Study dimensionality of hidden states in Student-Teacher experiments
--------------------------------------------------------------------
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import train_utils_st as ctu
import train_args_st
from sequential.ht_analyses.state_space_analysis import run

if __name__ == '__main__':
    ### Get default config for current dataset.
    config = train_args_st.parse_cmd_arguments(default=True)

    ### Extract important functions.
    task_loss_func = ctu.get_loss_func
    generate_tasks_func = ctu.generate_tasks

    ### Create dictionary explaining important settings for the current analysis
    analysis_kwd = {'complexity_measure': 'input_feature_dim',
                    'complexity_measure_name': 'input feature dim',
                    'forced_params': [],
                    'fixed_params': ['num_tasks']}

    run(config, analysis_kwd, task_loss_func, None, \
        generate_tasks_func, dataset='student_teacher')


