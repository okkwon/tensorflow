/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "iree_runtime_call.h"

#include <string.h>

#include "tensorflow/lite/c/c_api_types.h"

// #define PRINT_VALUES

#ifdef PRINT_VALUES
#include <stdio.h>
#endif

static iree_hal_element_type_t get_hal_elem_type(TfLiteType t) {
  switch (t) {
    case kTfLiteNoType:
      return IREE_HAL_ELEMENT_TYPE_NONE;
    case kTfLiteFloat32:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
    case kTfLiteInt32:
      return IREE_HAL_ELEMENT_TYPE_INT_32;
    case kTfLiteUInt8:
      return IREE_HAL_ELEMENT_TYPE_UINT_32;
    case kTfLiteInt64:
      return IREE_HAL_ELEMENT_TYPE_INT_64;
    case kTfLiteBool:
      return IREE_HAL_ELEMENT_TYPE_BOOL_8;
    case kTfLiteInt16:
      return IREE_HAL_ELEMENT_TYPE_INT_16;
    case kTfLiteComplex64:
      return IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64;
    case kTfLiteInt8:
      return IREE_HAL_ELEMENT_TYPE_INT_8;
    case kTfLiteFloat16:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_16;
    case kTfLiteFloat64:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_64;
    case kTfLiteComplex128:
      return IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128;
    case kTfLiteUInt64:
      return IREE_HAL_ELEMENT_TYPE_UINT_64;
    case kTfLiteUInt32:
      return IREE_HAL_ELEMENT_TYPE_UINT_32;
    case kTfLiteUInt16:
      return IREE_HAL_ELEMENT_TYPE_UINT_16;
    case kTfLiteInt4:
      return IREE_HAL_ELEMENT_TYPE_INT_4;
    case kTfLiteString:  // unsupported
    case kTfLiteResource:
    case kTfLiteVariant:
    default:
      return IREE_HAL_ELEMENT_TYPE_NONE;
  }
}

// Call a function from the module in the session.
iree_status_t iree_runtime_call_function(iree_runtime_session_t* session,
                                         iree_string_view_t function_name,
                                         TfLiteContext* context,
                                         TfLiteNode* node) {
  // Initialize the call to the function.
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_initialize_by_name(session, function_name, &call));

  // Append the function inputs with the HAL device allocator in use by the
  // session. The buffers will be usable within the session and _may_ be usable
  // in other sessions depending on whether they share a compatible device.
  // iree_hal_device_t* device = iree_runtime_session_device(session);
  iree_hal_allocator_t* device_allocator =
      iree_runtime_session_device_allocator(session);
  iree_allocator_t host_allocator =
      iree_runtime_session_host_allocator(session);

  iree_status_t status = iree_ok_status();

  // handle inputs
  for (int i = 0; i < node->inputs->size; ++i) {
    iree_hal_buffer_view_t* arg = NULL;
    const int tensor_index = node->inputs->data[i];

    if (iree_status_is_ok(status)) {
      iree_hal_dim_t arg_shape[MAX_TENSOR_DIMS] = {
          1,
      };
      const TfLiteTensor* tf_tensor = &context->tensors[tensor_index];

      if (tf_tensor->dims->size > MAX_TENSOR_DIMS) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "only supports up to %lu dims but got %d",
                                sizeof(arg_shape), tf_tensor->dims->size);
      }

      for (int dim = 0; dim < tf_tensor->dims->size; ++dim) {
        arg_shape[dim] = tf_tensor->dims->data[dim];
      }

      // import the buffer
      const iree_hal_buffer_params_t params = {
          .type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_HOST |
                  IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          .access = IREE_HAL_MEMORY_ACCESS_READ,
          .queue_affinity = 0,
      };
      iree_hal_external_buffer_t external_buffer = {
          .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
          .flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE,
          .size = tf_tensor->bytes,
          .handle.host_allocation.ptr = tf_tensor->data.data,
      };
      iree_hal_buffer_release_callback_t null_callback = {
          .fn = NULL,
          .user_data = NULL,
      };
      iree_hal_buffer_t* buffer = NULL;
      iree_status_t status = iree_hal_allocator_import_buffer(
          device_allocator, params, &external_buffer, null_callback, &buffer);

      if (iree_status_is_ok(status)) {
        // create a buffer view of the imported buffer
        status = iree_hal_buffer_view_create(
            buffer, tf_tensor->dims->size, arg_shape,
            get_hal_elem_type(tf_tensor->type),
            IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, host_allocator, &arg);
      }

      if (iree_status_is_ok(status)) {
        // The buffer view retains the buffer.
        iree_hal_buffer_release(buffer);
      }
    }
    if (iree_status_is_ok(status)) {
#ifdef PRINT_VALUES
      fprintf(stdout, "arg %d: ", i);
      IREE_IGNORE_ERROR(iree_hal_buffer_view_fprint(
          stdout, arg, /*max_element_count=*/4096, host_allocator));
#endif  // PRINT_VALUES
      // Add to the call inputs list (which retains the buffer view).
      status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg);
    }
    // Since the call retains the buffer view we can release it here.
    iree_hal_buffer_view_release(arg);
#ifdef PRINT_VALUES
    fprintf(stdout, "\n");
#endif  // PRINT_VALUES
  }
  // Synchronously perform the call.
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  // Process the outputs.
  // FIXME: figure out how to output to the tf output buffer. Would need to use
  // DPS.
  for (int i = 0; i < node->outputs->size; ++i) {
    iree_hal_buffer_view_t* ret_buffer_view = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_runtime_call_outputs_pop_front_buffer_view(
          &call, &ret_buffer_view);
    }

    if (iree_status_is_ok(status)) {
      int tensor_index = node->outputs->data[i];
      TfLiteTensor* tf_tensor = &context->tensors[tensor_index];

      iree_hal_buffer_mapping_t buffer_mapping = {{0}};
      IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
          iree_hal_buffer_view_buffer(ret_buffer_view),
          IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ, 0,
          IREE_WHOLE_BUFFER, &buffer_mapping));
      memcpy(tf_tensor->data.data, buffer_mapping.contents.data,
             buffer_mapping.contents.data_length);
    }
#ifdef PRINT_VALUES
    if (iree_status_is_ok(status)) {
      // This prints the buffer view out but an application could read its
      // contents, pass it to another call, etc.
      fprintf(stdout, "out  : ");
      status = iree_hal_buffer_view_fprint(
          stdout, ret_buffer_view, /*max_element_count=*/4096, host_allocator);
      fprintf(stdout, "\n");
    }
#endif  // PRINT_VALUES
    // The function allocates the output buffer and the buffer view, but when
    // the buffer view gets created it increments the buffer's reference count
    // to 2. To deallocate the buffer along with the buffer view, we need to
    // decrement it to 1.
    iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(ret_buffer_view);
    iree_hal_buffer_release(buffer);
    iree_hal_buffer_view_release(ret_buffer_view);
  }

  iree_runtime_call_deinitialize(&call);
  return status;
}
