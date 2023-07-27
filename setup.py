# coding=utf-8
# Copyright 2024 The Language Tale Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup."""

from distutils import core
import os

from setuptools import find_namespace_packages


here = os.path.abspath(os.path.dirname(__file__))
try:
  README = open(os.path.join(here, 'README.md'), encoding='utf-8').read()
except IOError:
  README = ''


install_requires = [
    'clu',
    'dm-reverb-nightly>=0.9.0.dev20221205',
    'gym<=0.23.0',
    'matplotlib',
    'numpy',
    'opencv-python',
    'protobuf',
    'pybullet',
    'rlds>=0.1.7',
    'scipy',
    'six',
    'tf-nightly>=2.12.0.dev20230201',
    'tensorflow_datasets>=4.7.0',
    'tf_agents>=0.14.0',
]


core.setup(
    name='language_table',
    version='0.1',
    description=(
        'Language-Table is a suite of human-collected datasets and a multi-task'
        ' continuous control benchmark for open vocabulary visuolinguomotor'
        ' learning.'
    ),
    long_description='\n\n'.join([README]),
    long_description_content_type='text/markdown',
    author='Language Table Team',
    author_email='language-table-team@google.com',
    url='https://github.com/google-research/language-table',
    packages=find_namespace_packages(),
    install_requires=install_requires,
    include_package_data=True,
)
