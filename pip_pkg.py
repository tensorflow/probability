#!/usr/bin/python
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
# ==============================================================================
import argparse
import os
import sys
import subprocess

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Build TF probability pip package.')
  parser.add_argument('dest', metavar='D', type=str, help='destination directory')
  parser.add_argument('args', metavar='A', type=str, nargs="*", help='additional arguments to setup.py')
  args = parser.parse_args()
  if not os.path.exists(args.dest):
    os.makedirs(args.dest)
  
  shargs = [sys.executable,"setup.py","bdist_wheel","--universal","--dist-dir",args.dest] + args.args
  with open(os.devnull,'w') as devnull:
    proc = subprocess.run(shargs, cwd=os.path.dirname(__file__), stdout=devnull)

  if proc.returncode != 0:
    print("Error running setup.py, return code {}".format(proc.returncode))
  else:
    print("Build complete. Wheel files are in {}".format(args.dest))
