# Copyright 2018 The TensorFlow Probability Authors.
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
"""Install tensorflow_probability."""
import datetime
import os
import sys

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(
    os.path.dirname(__file__), 'tensorflow_probability', 'python')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

REQUIRED_PACKAGES = [
    'six >= 1.10.0',
    'numpy >= 1.13.3',
]

REQUIRED_TENSORFLOW_VERSION = '1.10.0'

if '--gpu' in sys.argv:
  use_gpu = True
  sys.argv.remove('--gpu')
else:
  use_gpu = False

if '--release' in sys.argv:
  release = True
  sys.argv.remove('--release')
else:
  # Build a nightly package by default.
  release = False

maybe_gpu_suffix = '-gpu' if use_gpu else ''

if release:
  project_name = 'tensorflow-probability' + maybe_gpu_suffix
  tensorflow_package_name = 'tensorflow{}>={}'.format(
      maybe_gpu_suffix, REQUIRED_TENSORFLOW_VERSION)
else:
  # Nightly releases use date-based versioning of the form
  # '0.0.1.dev20180305'
  project_name = 'tfp-nightly' + maybe_gpu_suffix
  datestring = datetime.datetime.now().strftime('%Y%m%d')
  __version__ += datestring
  tensorflow_package_name = 'tf-nightly{}'.format(
      maybe_gpu_suffix)

REQUIRED_PACKAGES.append(tensorflow_package_name)


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return False

description_kwargs = {}
if use_gpu:
  description_kwargs['description'] = (
      'DEPRECATED, please use tensorflow-probability or tfp-nightly')
  description_kwargs['long_description'] = (
      'To select a GPU build, use a GPU flavor of TensorFlow, '
      'i.e. tensorflow-gpu or tf-nightly-gpu')
else:
  description_kwargs['description'] = (
      'Probabilistic modeling and statistical inference in TensorFlow')
 
setup(
    name=project_name,
    version=__version__,
    author='Google LLC',
    author_email='no-reply@google.com',
    url='http://github.com/tensorflow/probability',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    # Add in any packaged data.
    include_package_data=True,
    package_data={'': ['*.so']},
    exclude_package_data={'': ['BUILD', '*.h', '*.cc']},
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={
        'pip_pkg': InstallCommandBase,
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='tensorflow probability statistics bayesian machine learning',
    **description_kwargs
)
