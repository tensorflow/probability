#!/usr/bin/env python

from setuptools import setup, find_packages
import sys

setup(
    name="tensorflow-probability",
    version="conditionalmaf",
    description='Probabilistic modeling and statistical '
                'inference in TensorFlow',
    author='Google LLC',
    author_email='no-reply@google.com',
    url='http://github.com/tomcharnock/probability',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=[
      'six >= 1.10.0',
      'numpy >= 1.13.3',
      'decorator',
      'cloudpickle == 1.1.1',
      'gast >= 0.2, < 0.3'], 
    include_package_data=True,
    package_data={'': ['*.so']},
    exclude_package_data={'': ['BUILD', '*.h', '*.cc']},
    zip_safe=False,
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
