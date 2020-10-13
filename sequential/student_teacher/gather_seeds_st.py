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
# @title          :sequential/student_teacher/gather_seeds_st.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :07/28/2020
# @version        :1.0
# @python_version :3.6.10
"""
Gather results of a student-teacher experiment for different random seeds.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

from sequential import gather_random_seeds

if __name__=='__main__':

    # Define the dictionary with key-value pairs of config to be overwritten.
    forced_params ={
    }

    gather_random_seeds.run('sequential.student_teacher.hpconfig_st',
        forced_params=forced_params)

