# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Util functions/classes for jax primitive test harnesses."""

import contextlib
import functools
from typing import Optional
import warnings
import zlib
from absl import logging
from absl.testing import parameterized
import jax
from jax.experimental import jax2tf
from jax.experimental.jax2tf.tests import primitive_harness
import ml_dtypes
import numpy as np
import numpy.random as npr
import tensorflow as tf

_SUPPORTED_DTYPES = [
  np.float16,
  np.float32,
  np.float64,
  np.int8,
  np.int16,
  np.int32,
  np.int64,
  np.uint8,
  np.uint16,
  np.uint32,
  np.uint64,
  np.complex64,
  np.complex128
]

def _harness_matches(harness, group_name, dtype, params):
  if harness.group_name != group_name:
    return False
  if dtype is not None and harness.dtype != dtype:
    return False
  for key, value in params.items():
    if harness.params.get(key, None) != value:
      return False
  return True


# The following tests are currently crashing the TFLite converter/runtime.

_CRASH_LIST_PARMS = [
    # TODO(b/292571578) Fix the following crashes
    {
        "group_name": "eq",
        "dtype": np.uint8,
        "params": {"op_name": "eq", "lhs_shape": (), "rhs_shape": ()},
    },
    # TODO(b/292571295) Fix the following crashes
    {
        "group_name": "dot_general",
        "dtype": np.float32,
        "params": {
            "rhs_dtype": np.float32,
            "lhs_shape": (4,),
            "rhs_shape": (4,),
            "dimension_numbers": (((0,), (0,)), ((), ())),
            "precision": None,
            "preferred_element_type": None,
        },
    },
    {
        "group_name": "dot_general",
        "dtype": np.float16,
        "params": {
            "rhs_dtype": np.float16,
            "lhs_shape": (3,),
            "rhs_shape": (3,),
            "dimension_numbers": (((0,), (0,)), ((), ())),
            "precision": None,
            "preferred_element_type": np.float32,
        },
    },
    {
        "group_name": "dot_general",
        "dtype": np.float32,
        "params": {
            "rhs_dtype": np.float32,
            "lhs_shape": (3,),
            "rhs_shape": (3,),
            "dimension_numbers": (((0,), (0,)), ((), ())),
            "precision": None,
            "preferred_element_type": np.float32,
        },
    },
    {
        "group_name": "dot_general",
        "dtype": np.int8,
        "params": {
            "rhs_dtype": np.int8,
            "lhs_shape": (3,),
            "rhs_shape": (3,),
            "dimension_numbers": (((0,), (0,)), ((), ())),
            "precision": None,
            "preferred_element_type": np.int32,
        },
    },
    {
        "group_name": "dot_general",
        "dtype": None,  # match all types
        "params": {
            # "rhs_dtype": np.float32,  # match all
            "lhs_shape": (3,),
            "rhs_shape": (3,),
            "dimension_numbers": (((0,), (0,)), ((), ())),
            "precision": None,
            "preferred_element_type": None,
        },
    },
    # TODO(b/292570541) Fix the following crashes
    {
        "group_name": "conv_general_dilated",
        "dtype": np.float32,
        "params": {
            "lhs_shape": (2, 3, 10),
            "rhs_shape": (3, 3, 5),
            "window_strides": (1,),
            "padding": ((0, 0),),
            "lhs_dilation": (1,),
            "rhs_dilation": (1,),
            "dimension_numbers": ("NCH", "OIH", "NCH"),
            "feature_group_count": 1,
            "batch_group_count": 1,
            "precision": None,
            "preferred_element_type": None,
            "enable_xla": True,
        },
    },
    {
        "group_name": "conv_general_dilated",
        "dtype": np.float32,
        "params": {
            "lhs_shape": (2, 3, 9),
            "rhs_shape": (12, 1, 3),
            "window_strides": (1,),
            "padding": ((0, 0),),
            "lhs_dilation": (1,),
            "rhs_dilation": (1,),
            "dimension_numbers": ("NCH", "OIH", "NCH"),
            "feature_group_count": 3,
            "batch_group_count": 1,
            "precision": None,
            "preferred_element_type": None,
            "enable_xla": True,
        },
    },
    {
        "group_name": "conv_general_dilated",
        "dtype": np.float32,
        "params": {
            "lhs_shape": (2, 3, 9),
            "rhs_shape": (12, 1, 3),
            "window_strides": (1,),
            "padding": ((0, 0),),
            "lhs_dilation": (1,),
            "rhs_dilation": (2,),
            "dimension_numbers": ("NCH", "OIH", "NCH"),
            "feature_group_count": 3,
            "batch_group_count": 1,
            "precision": None,
            "preferred_element_type": None,
            "enable_xla": True,
        },
    },
    {
        "group_name": "conv_general_dilated",
        "dtype": np.float32,
        "params": {
            "lhs_shape": (1, 28, 1),
            "rhs_shape": (3, 1, 16),
            "window_strides": (1,),
            "padding": "VALID",
            "lhs_dilation": (1,),
            "rhs_dilation": (1,),
            "dimension_numbers": ("NWC", "WIO", "NWC"),
            "feature_group_count": 1,
            "batch_group_count": 1,
            "precision": None,
            "preferred_element_type": None,
            "enable_xla": True,
        },
    },
    {
        "group_name": "conv_general_dilated",
        "dtype": np.float32,
        "params": {
            "lhs_shape": (1, 28, 1),
            "rhs_shape": (3, 1, 16),
            "window_strides": (1,),
            "padding": "SAME",
            "lhs_dilation": (1,),
            "rhs_dilation": (1,),
            "dimension_numbers": ("NWC", "WIO", "NWC"),
            "feature_group_count": 1,
            "batch_group_count": 1,
            "precision": None,
            "preferred_element_type": None,
            "enable_xla": True,
        },
    },
    {
        "group_name": "conv_general_dilated",
        "dtype": np.float32,
        "params": {
            "lhs_shape": (1, 28, 1),
            "rhs_shape": (3, 1, 16),
            "window_strides": (1,),
            "padding": "VALID",
            "lhs_dilation": (1,),
            "rhs_dilation": (2,),
            "dimension_numbers": ("NWC", "WIO", "NWC"),
            "feature_group_count": 1,
            "batch_group_count": 1,
            "precision": None,
            "preferred_element_type": None,
            "enable_xla": True,
        },
    },
    {
        "group_name": "conv_general_dilated",
        "dtype": np.float32,
        "params": {
            "lhs_shape": (1, 28, 1),
            "rhs_shape": (3, 1, 16),
            "window_strides": (1,),
            "padding": "SAME",
            "lhs_dilation": (1,),
            "rhs_dilation": (2,),
            "dimension_numbers": ("NWC", "WIO", "NWC"),
            "feature_group_count": 1,
            "batch_group_count": 1,
            "precision": None,
            "preferred_element_type": None,
            "enable_xla": True,
        },
    },
]

_DEFAULT_TOLERANCE = {
    jax.dtypes.float0: 0,
    np.dtype(np.bool_): 0,
    np.dtype(ml_dtypes.int4): 0,
    np.dtype(np.int8): 0,
    np.dtype(np.int16): 0,
    np.dtype(np.int32): 0,
    np.dtype(np.int64): 0,
    np.dtype(ml_dtypes.uint4): 0,
    np.dtype(np.uint8): 0,
    np.dtype(np.uint16): 0,
    np.dtype(np.uint32): 0,
    np.dtype(np.uint64): 0,
    np.dtype(ml_dtypes.float8_e4m3b11fnuz): 1e-1,
    np.dtype(ml_dtypes.float8_e4m3fn): 1e-1,
    np.dtype(ml_dtypes.float8_e5m2): 1e-1,
    np.dtype(ml_dtypes.bfloat16): 1e-2,
    np.dtype(np.float16): 1e-3,
    np.dtype(np.float32): 1e-6,
    np.dtype(np.float64): 1e-15,
    np.dtype(np.complex64): 1e-6,
    np.dtype(np.complex128): 1e-15,
}


def device_under_test():
  return "cpu"


def _dtype(x):
  if hasattr(x, "dtype"):
    return x.dtype
  elif type(x) in jax.dtypes.python_scalar_dtypes:
    return np.dtype(jax.dtypes.python_scalar_dtypes[type(x)])
  else:
    return np.asarray(x).dtype


def tolerance(dtype, tol=None):
  tol = {} if tol is None else tol
  if not isinstance(tol, dict):
    return tol
  tol = {np.dtype(key): value for key, value in tol.items()}
  dtype = jax.dtypes.canonicalize_dtype(np.dtype(dtype))
  return tol.get(dtype, _DEFAULT_TOLERANCE[dtype])


def _assert_numpy_allclose(a, b, atol=None, rtol=None, err_msg=""):
  """Checks if two numpy arrays are all close given tolerances.

  Args:
    a: The array to check.
    b: The expected array.
    atol: Absolute tolerance.
    rtol: Relative tolerance.
    err_msg: The error message to print in case of failure.
  """
  if a.dtype == b.dtype == jax.dtypes.float0:
    np.testing.assert_array_equal(a, b, err_msg=err_msg)
    return
  custom_dtypes = [
      ml_dtypes.float8_e4m3b11fnuz,
      ml_dtypes.float8_e4m3fn,
      ml_dtypes.float8_e5m2,
      ml_dtypes.bfloat16,
  ]
  a = a.astype(np.float32) if a.dtype in custom_dtypes else a
  b = b.astype(np.float32) if b.dtype in custom_dtypes else b
  kw = {}
  if atol:
    kw["atol"] = atol
  if rtol:
    kw["rtol"] = rtol
  with np.errstate(invalid="ignore"):
    # TODO(phawkins): surprisingly, assert_allclose sometimes reports invalid
    # value errors. It should not do that.
    np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)


@contextlib.contextmanager
def ignore_warning(**kw):
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", **kw)
    yield


def _make_tf_input_signature(*tf_args) -> list[tf.TensorSpec]:
  # tf_args can be PyTrees
  def _make_one_array_signature(tf_arg):
    return tf.TensorSpec(np.shape(tf_arg), jax2tf.dtype_of_val(tf_arg))

  return tf.nest.map_structure(_make_one_array_signature, list(tf_args))


def _has_only_supported_dtypes(harness):
  if harness.dtype not in _SUPPORTED_DTYPES:
    return False

  for key, value in harness.params.items():
    if "dtype" in key and value not in _SUPPORTED_DTYPES:
      return False

  return True


def primitives_parameterized(
    harnesses, *, one_containing: Optional[str] = None
):
  """Decorator for tests. This is used to filter the tests.

  Args:
    harnesses: List of Harness objects to be filtered.
    one_containing: If set, only creates one test case for the provided name.

  Returns:
    A parameterized version of the test function with filtered set of harnesses.
  """

  def _filter_harness(harness):
    # TODO(b/295369536) Put a limitations system in place so what's not covered
    # is explicit.
    if not harness.params.get("enable_xla", True):
      return False

    if one_containing is not None and one_containing not in harness.fullname:
      return False

    if not _has_only_supported_dtypes(harness):
      return False

    if harness.group_name != "add":
      return False

    for crash_item in _CRASH_LIST_PARMS:
      if _harness_matches(
          harness,
          crash_item["group_name"],
          crash_item["dtype"],
          crash_item["params"],
      ):
        return False

    return True

  harnesses = filter(_filter_harness, harnesses)

  return primitive_harness.parameterized(harnesses, include_jax_unimpl=False)


def _get_jax2tflite_result(func_jax, *tf_args):
  """Runs a JAX function using TFLite.

  Args:
    func_jax: Jax Callable.
    *tf_args: Arguments to be passed.

  Returns:
    The result returned by the TFLite runtime.
  """

  # JAX -> TF
  input_signature = _make_tf_input_signature(*tf_args)
  tf_fn = tf.function(
      jax2tf.convert(func_jax, native_serialization=True),
      input_signature=input_signature,
      autograph=False,
  )
  apply_tf = tf_fn.get_concrete_function()

  # TF -> TFLite
  converter = tf.lite.TFLiteConverter.from_concrete_functions([apply_tf], tf_fn)
  tflite_model = converter.convert()

  # Run the TFLite model.
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()
  inputs = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  outputs = tuple(interpreter.tensor(out["index"]) for out in output_details)

  for i, x in enumerate(tf_args):
    interpreter.set_tensor(inputs[i]["index"], x)
  interpreter.invoke()
  result = tuple(o() for o in outputs)
  if len(result) != 1:
    raise AssertionError(f"Expecting exactly 1 result, got {len(result)}.")

  return result[0]


def is_sequence(x):
  try:
    iter(x)
  except TypeError:
    return False
  else:
    return True


class JaxToTfliteTestCase(parameterized.TestCase):
  """A test case for JAX to TFLite conversions."""

  # We want most tests to use the maximum available version, from the locally
  # installed tfxla module.
  use_max_serialization_version = True

  def setUp(self):
    super().setUp()

    # We use the adler32 hash for two reasons.
    # a) it is deterministic run to run, unlike hash() which is randomized.
    # b) it returns values in int32 range, which RandomState requires.
    self._rng = npr.RandomState(zlib.adler32(self._testMethodName.encode()))

    # We run the tests using the maximum version supported, even though
    # the default serialization version may be held back for a while to
    # ensure compatibility
    version = jax.config.jax_serialization_version
    self.addCleanup(
        functools.partial(
            jax.config.update, "jax_serialization_version", version
        )
    )
    if self.use_max_serialization_version:
      # The largest version we support is 7
      max_version = 7 # min(7, tfxla.call_module_maximum_supported_version())
      self.assertLessEqual(version, max_version)
      version = max_version
      jax.config.update("jax_serialization_version", max_version)
    logging.info(
        "Using JAX serialization version %s%s",
        version,
        " (max_version)" if self.use_max_serialization_version else "",
    )

  def rng(self):
    return self._rng

  def assert_all_close(
      self,
      x,
      y,
      *,
      check_dtypes=True,
      atol=None,
      rtol=None,
      canonicalize_dtypes=True,
      err_msg="",
  ):
    """Assert that x and y, either arrays or nested tuples/lists, are close."""
    if isinstance(x, dict):
      self.assertIsInstance(y, dict)
      self.assertEqual(set(x.keys()), set(y.keys()))
      for k in x.keys():
        self.assert_all_close(
            x[k],
            y[k],
            check_dtypes=check_dtypes,
            atol=atol,
            rtol=rtol,
            canonicalize_dtypes=canonicalize_dtypes,
            err_msg=err_msg,
        )
    elif is_sequence(x) and not hasattr(x, "__array__"):
      self.assertTrue(is_sequence(y) and not hasattr(y, "__array__"))
      self.assertEqual(len(x), len(y))
      for x_elt, y_elt in zip(x, y):
        self.assert_all_close(
            x_elt,
            y_elt,
            check_dtypes=check_dtypes,
            atol=atol,
            rtol=rtol,
            canonicalize_dtypes=canonicalize_dtypes,
            err_msg=err_msg,
        )
    elif hasattr(x, "__array__") or np.isscalar(x):
      self.assertTrue(hasattr(y, "__array__") or np.isscalar(y))
      if check_dtypes:
        self.assert_dtypes_match(x, y, canonicalize_dtypes=canonicalize_dtypes)
      x = np.asarray(x)
      y = np.asarray(y)
      self.assert_arrays_all_close(
          x, y, check_dtypes=False, atol=atol, rtol=rtol, err_msg=err_msg
      )
    elif x == y:
      return
    else:
      raise TypeError((type(x), type(y)))

  def assert_arrays_all_close(
      self, x, y, *, check_dtypes=True, atol=None, rtol=None, err_msg=""
  ):
    """Assert that x and y are close (up to numerical tolerances)."""
    self.assertEqual(x.shape, y.shape)
    atol = max(tolerance(_dtype(x), atol), tolerance(_dtype(y), atol))
    rtol = max(tolerance(_dtype(x), rtol), tolerance(_dtype(y), rtol))

    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)

    if check_dtypes:
      self.assert_dtypes_match(x, y)

  def assert_dtypes_match(self, x, y, *, canonicalize_dtypes=True):
    if not jax.config.x64_enabled and canonicalize_dtypes:
      self.assertEqual(
          jax.dtypes.canonicalize_dtype(_dtype(x), allow_extended_dtype=True),
          jax.dtypes.canonicalize_dtype(_dtype(y), allow_extended_dtype=True),
      )
    else:
      self.assertEqual(_dtype(x), _dtype(y))

  def convert_and_compare_tflite(
      self,
      func_jax,
      *args,
      limitations=(),
  ):
    """Compares jax_func(*args) with the output of a converted TFLite model.

    Args:
      func_jax: the function to invoke (``func_jax(*args)``)
      *args: the arguments.
      limitations: the set of limitations for this harness.
    """

    def log_message(extra):
      return f"[{self._testMethodName}]: {extra}"

    # If any of the limitations have skip_tf_run set to True, skip this test.
    skip_tf_run = [l for l in limitations if l.skip_tf_run]
    if skip_tf_run:
      logging.info(log_message(f"Skip TF run due to limitations {skip_tf_run}"))
      return

    # Run the JAX callable directly.
    result_jax = func_jax(*args)
    result_tf = None

    try:
      result_tf = _get_jax2tflite_result(func_jax, *args)
      tf_exception = None
    except Exception as e:  # pylint: disable=broad-exception-caught
      tf_exception = e

    expect_tf_error = [l for l in limitations if l.expect_tf_error]
    if tf_exception:
      if expect_tf_error:
        logging.info(
            log_message(
                "Found expected TF error with enabled limitations "
                f"{expect_tf_error}; TF error is {tf_exception}"
            )
        )
        return
      else:
        raise tf_exception
    else:
      if expect_tf_error:
        logging.warning(
            log_message(
                f"Unexpected success with known limitations {expect_tf_error}"
            )
        )

    skip_comparison = [l for l in limitations if l.skip_comparison]
    if skip_comparison:
      logging.warning(
          log_message(f"Skip result comparison due to {skip_comparison}")
      )
      return

    max_tol = None
    max_tol_lim = (
        None
        if not limitations
        else limitations[0].get_max_tolerance_limitation(limitations)
    )
    if max_tol_lim is not None:
      max_tol = max_tol_lim.tol
      logging.info(log_message(f"Using tol={max_tol} due to {max_tol_lim}"))

    custom_assert_lim = [l for l in limitations if l.custom_assert]
    assert len(custom_assert_lim) <= 1, (
        "Expecting at most one applicable limitation with custom_assert, found"
        f" {custom_assert_lim}"
    )

    if custom_assert_lim:
      logging.info(
          log_message(
              f"Running custom_assert with tol={max_tol} due to"
              f" {custom_assert_lim[0]}"
          )
      )
      custom_assert_lim[0].custom_assert(
          self, result_jax, result_tf, args=args, tol=max_tol
      )
    else:
      logging.info(log_message(f"Running default assert with tol={max_tol}"))
      self.assert_all_close(result_jax, result_tf, atol=max_tol, rtol=max_tol)
