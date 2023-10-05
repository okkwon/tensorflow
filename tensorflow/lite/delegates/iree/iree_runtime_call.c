// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree_runtime_call.h"

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
  iree_hal_device_t* device = iree_runtime_session_device(session);
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
                                "only supports up to %d dims but got %d",
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
          .handle.host_allocation.ptr = tf_tensor->data.f,
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
            IREE_HAL_ELEMENT_TYPE_FLOAT_32,
            IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, host_allocator, &arg);
      }

      if (iree_status_is_ok(status)) {
        // The buffer view retains the buffer.
        iree_hal_buffer_release(buffer);
      }
    }
    if (iree_status_is_ok(status)) {
#ifndef NDEBUG
      fprintf(stdout, "arg %zu: ", i);
      IREE_IGNORE_ERROR(iree_hal_buffer_view_fprint(
          stdout, arg, /*max_element_count=*/4096, host_allocator));
#endif  // NDEBUG
      // Add to the call inputs list (which retains the buffer view).
      status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg);
    }
    // Since the call retains the buffer view we can release it here.
    iree_hal_buffer_view_release(arg);
#ifndef NDEBUG
    fprintf(stdout, "\n");
#endif  // NDEBUG
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
      memcpy(tf_tensor->data.f, buffer_mapping.contents.data,
             buffer_mapping.contents.data_length);
    }
#ifndef NDEBUG
    if (iree_status_is_ok(status)) {
      // This prints the buffer view out but an application could read its
      // contents, pass it to another call, etc.
      fprintf(stdout, "out  : ");
      status = iree_hal_buffer_view_fprint(
          stdout, ret_buffer_view, /*max_element_count=*/4096, host_allocator);
      fprintf(stdout, "\n");
    }
#endif  // NDEBUG
    iree_hal_buffer_view_release(ret_buffer_view);
  }

  iree_runtime_call_deinitialize(&call);
  return status;
}
