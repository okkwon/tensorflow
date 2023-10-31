"""Runs the JAX primitives test suite through TFLite's converter and runtime."""
from absl import logging
from absl.testing import absltest
import jax
from jax.experimental.jax2tf.tests import jax2tf_limitations
from jax.experimental.jax2tf.tests import primitive_harness

import jax2tf_test_util

class PrimitivesTest(jax2tf_test_util.JaxToTfliteTestCase):

  @jax2tf_test_util.primitives_parameterized(
      primitive_harness.all_harnesses,
  )
  @jax2tf_test_util.ignore_warning(
      category=UserWarning, message="Using reduced precision for gradient.*"
  )
  def test_tflite_prim(self, harness: primitive_harness.Harness):
    device = jax2tf_test_util.device_under_test()

    def _filter_limitation(limitation):
      return limitation.filter(
          device=device, dtype=harness.dtype, mode="compiled"
      )

    limitations = tuple(
        filter(
            _filter_limitation,
            jax2tf_limitations.Jax2TfLimitation.limitations_for_harness(
                harness
            ),
        )
    )

    func_jax = harness.dyn_fun
    args = harness.dyn_args_maker(self.rng())

    try:
      with jax.jax2tf_associative_scan_reductions(True):
        self.convert_and_compare_tflite(
            func_jax,
            *args,
            limitations=limitations,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
      if "does not work with custom calls" in str(e):
        logging.warning("Suppressing error %s", e)
      else:
        raise e


if __name__ == "__main__":
  absltest.main()
