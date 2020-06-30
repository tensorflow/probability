# Copyright 2020 The TensorFlow Probability Authors.
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
# ============================================================================
"""Install oryx."""
import os
import sys
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'dataclasses;python_version<"3.7"',
    'jax==0.1.71',
    'jaxlib',
    # Pin a TF version while TFP-on-JAX still depends on TF
    'tensorflow==2.2.0',
    # Pin a TFP version until a new release
    'tfp-nightly==0.11.0.dev20200629',
]


# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(
    os.path.dirname(__file__), 'oryx')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

with open('README.md', 'r') as fh:
  oryx_long_description = fh.read()

setup(
    name='oryx',
    python_requires='>=3.6',
    version=__version__,
    description='Probabilistic programming and deep learning in JAX',
    long_description=oryx_long_description,
    long_description_content_type='text/markdown',
    author='Google LLC',
    author_email='no-reply@google.com',
    url='http://github.com/tensorflow/probability/spinoffs/oryx',
    license='Apache 2.0',
    packages=find_packages('.'),
    install_requires=REQUIRED_PACKAGES,
    # Add in any packaged data.
    exclude_package_data={'': ['BUILD']},
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='jax probability statistics bayesian machine learning',
)
