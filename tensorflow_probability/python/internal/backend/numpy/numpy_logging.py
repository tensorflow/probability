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
"""Experimental Numpy backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python.internal.backend.numpy import _utils as utils


__all__ = [
    'TaskLevelStatusMessage',
    'debug',
    'error',
    'fatal',
    'flush',
    'get_verbosity',
    'info',
    'log',
    'log_every_n',
    'log_first_n',
    'log_if',
    'set_verbosity',
    'vlog',
    'warn',
    'warning',
    'DEBUG',
    'ERROR',
    'FATAL',
    'INFO',
    'WARN',
]


def _TaskLevelStatusMessage(*_, **__):  # pylint: disable=invalid-name,unused-argument
  pass


def _debug(*_, **__):  # pylint: disable=unused-argument
  pass


def _error(*_, **__):  # pylint: disable=unused-argument
  pass


def _fatal(*_, **__):  # pylint: disable=unused-argument
  pass


def _flush(*_, **__):  # pylint: disable=unused-argument
  pass


def _get_verbosity(*_, **__):  # pylint: disable=unused-argument
  pass


def _info(*_, **__):  # pylint: disable=unused-argument
  pass


def _log(*_, **__):  # pylint: disable=unused-argument
  pass


def _log_every_n(*_, **__):  # pylint: disable=unused-argument
  pass


def _log_first_n(*_, **__):  # pylint: disable=unused-argument
  pass


def _log_if(*_, **__):  # pylint: disable=unused-argument
  pass


def _set_verbosity(*_, **__):  # pylint: disable=unused-argument
  pass


def _vlog(*_, **__):  # pylint: disable=unused-argument
  pass


def _warn(*_, **__):  # pylint: disable=unused-argument
  pass


def _warning(*_, **__):  # pylint: disable=unused-argument
  pass


# --- Begin Public Functions --------------------------------------------------

TaskLevelStatusMessage = utils.copy_docstring(  # pylint: disable=invalid-name
    'tf1.logging.TaskLevelStatusMessage',
    _TaskLevelStatusMessage)

debug = utils.copy_docstring(
    'tf1.logging.debug',
    _debug)

error = utils.copy_docstring(
    'tf1.logging.error',
    _error)

fatal = utils.copy_docstring(
    'tf1.logging.fatal',
    _fatal)

flush = utils.copy_docstring(
    'tf1.logging.flush',
    _flush)

get_verbosity = utils.copy_docstring(
    'tf1.logging.get_verbosity',
    _get_verbosity)

info = utils.copy_docstring(
    'tf1.logging.info',
    _info)

log = utils.copy_docstring(
    'tf1.logging.log',
    _log)

log_every_n = utils.copy_docstring(
    'tf1.logging.log_every_n',
    _log_every_n)

log_first_n = utils.copy_docstring(
    'tf1.logging.log_first_n',
    _log_first_n)

log_if = utils.copy_docstring(
    'tf1.logging.log_if',
    _log_if)

set_verbosity = utils.copy_docstring(
    'tf1.logging.set_verbosity',
    _set_verbosity)

vlog = utils.copy_docstring(
    'tf1.logging.vlog',
    _vlog)

warn = utils.copy_docstring(
    'tf1.logging.warn',
    _warn)

warning = utils.copy_docstring(
    'tf1.logging.warning',
    _warning)

DEBUG = 'DEBUG'
ERROR = 'ERROR'
FATAL = 'FATAL'
INFO = 'INFO'
WARN = 'WARN'
