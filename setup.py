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
import sys

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution

VERSION = '0.0.1'

REQUIRED_PACKAGES = [
    'six >= 1.10.0',
    'numpy >= 1.11.1',
]
# TODO(b/76094057): Once we support releases, enable the following:
# REQUIRED_TENSORFLOW_VERSION = '1.6.0'

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
  raise NotImplementedError('TensorFlow Probability team does not yet '
                            'support releases.')
  # TODO(b/76094057): Once we support releases, enable the following:
  # tensorflow_package_name = 'tensorflow{}>={}'.format(
  #     maybe_gpu_suffix, REQUIRED_TENSORFLOW_VERSION)
else:
  # Nightly releases use date-based versioning of the form
  # '0.0.1.dev20180305'
  project_name = 'tfp-nightly' + maybe_gpu_suffix
  datestring = datetime.datetime.now().strftime('%Y%m%d')
  VERSION += '.dev' + datestring
  tensorflow_package_name = 'tf-nightly{}'.format(
      maybe_gpu_suffix)

REQUIRED_PACKAGES.append(tensorflow_package_name)


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return False

setup(
    name=project_name,
    version=VERSION,
    description='Probabilistic modeling and statistical '
                'inference in TensorFlow',
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
)
