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
    'decorator',
    'cloudpickle == 1.3',  # TODO(b/155109696): Unpin cloudpickle version.
    'gast >= 0.3.2'  # For autobatching
]

if '--release' in sys.argv:
  release = True
  sys.argv.remove('--release')
else:
  # Build a nightly package by default.
  release = False

if release:
  project_name = 'tensorflow-probability'
else:
  project_name = 'tfp-nightly'

if release:
  TFDS_PACKAGE = 'tensorflow-datasets >= 2.2.0'
else:
  TFDS_PACKAGE = 'tfds-nightly'


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return False

with open('README.md', 'r') as fh:
  long_description = fh.read()

setup(
    name=project_name,
    version=__version__,
    description='Probabilistic modeling and statistical '
                'inference in TensorFlow',
    long_description=long_description,
    long_description_content_type='text/markdown',
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
    keywords='tensorflow probability statistics bayesian machine learning',
    extras_require={  # e.g. `pip install tfp-nightly[jax]`
        'jax': ['jax', 'jaxlib'],
        'tfds': [TFDS_PACKAGE],
    }
)
