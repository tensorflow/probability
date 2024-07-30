# Copyright 2024 The TensorFlow Probability Authors.
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
import functools
import os

from discussion.robust_inverse_graphics import saving
from discussion.robust_inverse_graphics.util import test_util
from fun_mc import using_jax as fun_mc


class SavingTest(test_util.TestCase):

  def test_interruptible_trace_state_saving(self):
    def fun(x, y):
      x = x + 1.0
      y = y + 2.0
      return (x, y), (x, y)

    state, _ = fun_mc.trace(
        state=fun_mc.interruptible_trace_init((0.0, 0.0), fn=fun, num_steps=5),
        fn=functools.partial(fun_mc.interruptible_trace_step, fn=fun),
        num_steps=4,
    )

    out_dir = self.create_tempdir()
    path = os.path.join(out_dir, 'test.tree2')

    saving.save(state, path)
    state = saving.load(path)

    # This line would only work if the right type got loaded.
    x_trace, y_trace = state.trace()

    self.assertAllEqual([1.0, 2.0, 3.0, 4.0], x_trace)
    self.assertAllEqual([2.0, 4.0, 6.0, 8.0], y_trace)


if __name__ == '__main__':
  test_util.main()
