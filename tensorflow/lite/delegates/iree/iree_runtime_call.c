// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/runtime/api.h"

// Temporary struct to mimic TFLiteTensor for testing.
// FIXME: should be merged when integrating this file into the iree delegate
typedef enum {
  kTfLiteNoType = 0,
  kTfLiteFloat32 = 1,
  kTfLiteInt32 = 2,
  kTfLiteUInt8 = 3,
  kTfLiteInt64 = 4,
  kTfLiteString = 5,
  kTfLiteBool = 6,
  kTfLiteInt16 = 7,
  kTfLiteComplex64 = 8,
  kTfLiteInt8 = 9,
  kTfLiteFloat16 = 10,
  kTfLiteFloat64 = 11,
  kTfLiteComplex128 = 12,
  kTfLiteUInt64 = 13,
  kTfLiteResource = 14,
  kTfLiteVariant = 15,
  kTfLiteUInt32 = 16,
  kTfLiteUInt16 = 17,
  kTfLiteInt4 = 18,
} TfLiteType;

#define TFLITE_MAX_RANK 4  // FIXME

typedef struct TfLiteTensor {
  TfLiteType elemType;
  float* data;  // FIXME: should be a union
  int rank;
  int dims[TFLITE_MAX_RANK];
  int num_dims;
  size_t bytes;
} TfLiteTensor;

typedef struct iree_string_list_t {
  // Total number of values in the list.
  iree_host_size_t count;
  // Value list or NULL if no values.
  const iree_string_view_t* values;
} iree_string_list_t;

static iree_status_t iree_runtime_module_call(
    iree_runtime_instance_t* instance, iree_string_view_t module_path,
    iree_string_view_t function_name, TfLiteTensor* tf_inputs,
    iree_host_size_t num_inputs, iree_const_byte_span_t module_contents,
    int* out_exit_code);

int iree_call(const char* module_path_cstr, const char* function_name_cstr,
              TfLiteTensor* tf_inputs, size_t num_inputs,
              TfLiteTensor* tf_outputs, size_t num_outputs) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Set up the shared runtime instance.
  // An application should usually only have one of these and share it across
  // all of the sessions it has. The instance is thread-safe, while the
  // sessions are only thread-compatible (you need to lock if its required).
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  iree_status_t status = iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &instance);

  // Utility to run the module with the command line flags. This particular
  // method is only useful in these IREE tools that want consistent flags -
  // a real application will need to do what this is doing with its own setup
  // and I/O handling.
  int exit_code = EXIT_SUCCESS;
  if (iree_status_is_ok(status)) {
    iree_const_byte_span_t module_contents = iree_const_byte_span_empty();
    iree_string_view_t module_path = iree_make_cstring_view(module_path_cstr);
    iree_string_view_t function_name =
        iree_make_cstring_view(function_name_cstr);
    status = iree_runtime_module_call(instance, module_path, function_name,
                                      tf_inputs, num_inputs,
                                      module_contents, &exit_code);
  }

  iree_runtime_instance_release(instance);

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}

int test_add_4d_f32() {
  // FIXME: add a logic to select an op from the op type.
  const char* module_path_cstr =
      "/usr/local/google/home/okwan/callable-module/test/simple_add/add.vmfb";
  // FIXME: get the function name from the TF op.
  float in_data0[1] = {1.0f};
  float in_data1[1] = {2.0f};
  const char* function_name_cstr = "module.add_4d_f32";
  TfLiteTensor tf_inputs[2] = {
      {/*.elemType=*/kTfLiteFloat32, /*.data=*/in_data0, /*.rank=*/1,
       /*.dims=*/{1,1,1,1}, /*.num_dims=*/4, /*.bytes=*/4},
      {/*.elemType=*/kTfLiteFloat32, /*.data=*/in_data1, /*.rank=*/1,
       /*.dims=*/{1,1,1,1}, /*.num_dims=*/4, /*.bytes=*/4},
  };
  size_t num_inputs = 2;

  float out_data0[1] = {0.0f};
  TfLiteTensor tf_outputs[1] = {
      {/*.elemType=*/kTfLiteFloat32, /*.data=*/out_data0, /*.rank=*/1,
       /*.dims=*/{1,1,1,1}, /*.num_dims=*/4, /*.bytes=*/4},
  };
  size_t num_outputs = 1;

  return iree_call(module_path_cstr, function_name_cstr, tf_inputs,
                   num_inputs, tf_outputs, num_outputs);
}

/* out_list is a list of iree_buffer_view_ref_t.
   For each input, the given information is
    - raw data pointer and size in byte
    - pointer to dims[] and the size --> shape_rank, iree_host_size_t
      * iree_alloca(rank * sizeof(iree_hal_dim_t))
      * parse(dims, ndims, &shape_rank, shape)
    - element type --> iree_hal_element_type_t
    - encondig -> ROW_MAJOR
    - import the buffer from memory
      * iree_hal_external_buffer_t external_buffer = { ... };
      * iree_hal_allocator_import_buffer(..., &external_buffer, &out_buffer);
      * iree_hal_buffer_view_create(out_buffer, rank, shape, elem_type,
          encoding, host_allocator, &out_buffer_view);

*/
static iree_status_t iree_runtime_call_function(
    iree_runtime_session_t* session, iree_string_view_t function_name,
    TfLiteTensor* tf_inputs, iree_host_size_t num_inputs) {
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
  for (iree_host_size_t i = 0; i < num_inputs; ++i) {
    iree_hal_buffer_view_t* arg = NULL;

    if (iree_status_is_ok(status)) {
      const iree_hal_dim_t arg_shape[1] = {1};  // FIXME:
      const float* arg_data = tf_inputs[i].data;

      status = iree_hal_buffer_view_allocate_buffer_copy(
          device, device_allocator,
          // Shape rank and dimensions:
          IREE_ARRAYSIZE(arg_shape), arg_shape,
          // Element type:
          IREE_HAL_ELEMENT_TYPE_FLOAT_32,
          // Encoding type:
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
          (iree_hal_buffer_params_t){
              // Where to allocate (host or device):
              .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
              // Access to allow to this memory:
              .access = IREE_HAL_MEMORY_ACCESS_ALL,
              // Intended usage of the buffer (transfers, dispatches, etc):
              .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          },
          // The actual heap buffer to wrap or clone and its allocator:
          iree_make_const_byte_span(arg_data, tf_inputs[i].bytes),
          // Buffer view + storage are returned and owned by the caller:
          &arg);
    }
    if (iree_status_is_ok(status)) {
      fprintf(stdout, "arg %zu: ", i);
      IREE_IGNORE_ERROR(iree_hal_buffer_view_fprint(
          stdout, arg, /*max_element_count=*/4096, host_allocator));
      // Add to the call inputs list (which retains the buffer view).
      status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg);
    }
    // Since the call retains the buffer view we can release it here.
    iree_hal_buffer_view_release(arg);
    fprintf(stdout, "\n");
  }
  // Synchronously perform the call.
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  // Dump the function outputs.
  iree_hal_buffer_view_t* ret0 = NULL;
  if (iree_status_is_ok(status)) {
    // Try to get the first call result as a buffer view.
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &ret0);
  }
  if (iree_status_is_ok(status)) {
    // This prints the buffer view out but an application could read its
    // contents, pass it to another call, etc.
    fprintf(stdout, "out = ");
    status = iree_hal_buffer_view_fprint(
        stdout, ret0, /*max_element_count=*/4096, host_allocator);
    fprintf(stdout, "\n");
  }
  iree_hal_buffer_view_release(ret0);

  iree_runtime_call_deinitialize(&call);
  return status;
}

static iree_status_t iree_runtime_module_call(
    iree_runtime_instance_t* instance, iree_string_view_t module_path,
    iree_string_view_t function_name, TfLiteTensor* tf_inputs,
    iree_host_size_t num_inputs, iree_const_byte_span_t module_contents,
    int* out_exit_code) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(#5724): move device selection into the compiled modules.
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view("local-task"), &device));

  // Set up the session to run the module.
  // Sessions are like OS processes and are used to isolate modules from each
  // other and hold runtime state such as the variables used within the module.
  // The same module loaded into two sessions will see their own private state.
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = NULL;
  iree_status_t status = iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session);
  iree_hal_device_release(device);

  // Load the compiled user module in a demo-specific way.
  // Applications could specify files, embed the outputs directly in their
  // binaries, fetch them over the network, etc.
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_bytecode_module_from_file(
        session, module_path.data);
  }

  // Build and issue the call.
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_function(session, function_name, tf_inputs,
                                        num_inputs);
  }

  // Release the session and free all resources.
  iree_runtime_session_release(session);
  return status;
}
