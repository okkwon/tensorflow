
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
#include "tensorflow/lite/delegates/iree/iree_delegate.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/iree/iree_runtime_call.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"

namespace tflite {
namespace iree_test {

// Iree delegate kernel.
class IreeDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit IreeDelegateKernel(const TfLiteIreeDelegateOptions& options)
      : options_(options) {}

  ~IreeDelegateKernel() override {
    iree_runtime_session_release(iree_runtime_session_);
    iree_runtime_instance_release(iree_runtime_instance_);
  }

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    // TODO: move the module loading into the delegate init.

    // Set up the shared runtime instance.
    // An application should usually only have one of these and share it across
    // all of the sessions it has. The instance is thread-safe, while the
    // sessions are only thread-compatible (you need to lock if its required).
    iree_runtime_instance_options_initialize(&iree_runtime_instance_options_);
    iree_runtime_instance_options_use_all_available_drivers(
        &iree_runtime_instance_options_);
    iree_runtime_instance_ = nullptr;
    iree_status_t status = iree_runtime_instance_create(
        &iree_runtime_instance_options_, iree_allocator_system(),
        &iree_runtime_instance_);

    iree_hal_device_t* iree_device = nullptr;

    if (!iree_status_is_ok(status)) {
      LOG(ERROR) << "iree_runtime_instance_create failed: "
                 << iree_status_code_string(iree_status_code(status));
      return kTfLiteError;
    }

    status = iree_runtime_instance_try_create_default_device(
        iree_runtime_instance_, iree_make_cstring_view("local-task"),
        &iree_device);

    if (!iree_status_is_ok(status)) {
      LOG(ERROR) << "iree_runtime_instance_try_create_default_device failed: "
                 << iree_status_code_string(iree_status_code(status));
      return kTfLiteError;
    }

    // Set up the session to run the module.
    // Sessions are like OS processes and are used to isolate modules from
    // each other and hold runtime state such as the variables used within the
    // module. The same module loaded into two sessions will see their own
    // private state.
    iree_runtime_session_options_initialize(&iree_runtime_session_options_);
    /* iree_status_t status = */ iree_runtime_session_create_with_device(
        iree_runtime_instance_, &iree_runtime_session_options_, iree_device,
        iree_runtime_instance_host_allocator(iree_runtime_instance_),
        &iree_runtime_session_);
    // Session keeps device reference internally.
    iree_hal_device_release(iree_device);

    // Load the compiled user module in a demo-specific way.
    // Applications could specify files, embed the outputs directly in their
    // binaries, fetch them over the network, etc.
    // The lifetime of the delegate data in the context can be disjoint from
    // the lifetime of the delegate, so it is safe to copy it here.
    vmfb_content_.resize(context->delegate_data_size);
    memcpy(vmfb_content_.data(), context->delegate_data,
           context->delegate_data_size);
    iree_const_byte_span_t vmfb_data =
        iree_make_const_byte_span(vmfb_content_.data(), vmfb_content_.size());

    status = iree_runtime_session_append_bytecode_module_from_memory(
        iree_runtime_session_, vmfb_data, iree_allocator_null());

    if (!iree_status_is_ok(status)) {
      LOG(ERROR) << "iree_runtime_session_append_bytecode_module_from_memory() "
                 << "failed : "
                 << iree_status_code_string(iree_status_code(status));
      return kTfLiteError;
    }

    // Build and issue the call.

    // Save index to all nodes which are part of this delegate.
    inputs_.resize(params->nodes_to_replace->size);
    outputs_.resize(params->nodes_to_replace->size);
    builtin_code_.resize(params->nodes_to_replace->size);

    // Currently, we only support a single builtin code.
    TF_LITE_ENSURE_EQ(context, builtin_code_.size(), 1);

    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
      const int node_index = params->nodes_to_replace->data[i];
      // Get this node information.
      TfLiteNode* delegated_node = nullptr;
      TfLiteRegistration* delegated_node_registration = nullptr;
      TF_LITE_ENSURE_EQ(
          context,
          context->GetNodeAndRegistration(context, node_index, &delegated_node,
                                          &delegated_node_registration),
          kTfLiteOk);
      inputs_[i].push_back(delegated_node->inputs->data[0]);
      inputs_[i].push_back(delegated_node->inputs->data[1]);
      outputs_[i].push_back(delegated_node->outputs->data[0]);
      builtin_code_[i] = delegated_node_registration->builtin_code;
    }

    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    return !options_.error_during_prepare ? kTfLiteOk : kTfLiteError;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    if (options_.error_during_invoke) return kTfLiteError;

    // Evaluate the delegated graph.
    // Here we loop over all the delegated nodes.
    // We know that all the nodes are either ADD or SUB operations and the
    // number of nodes equals ''inputs_.size()'' and inputs[i] is a list of
    // tensor indices for inputs to node ''i'', while outputs_[i] is the list of
    // outputs for node
    // ''i''. Note, that it is intentional we have simple implementation as this
    // is for demonstration.

    for (int i = 0; i < inputs_.size(); ++i) {
      // FIXME: get the function name from the TF op.
      std::string function_name_str =
          GetFunctionName(context, builtin_code_[i], inputs_[i], outputs_[i]);

      LOG(INFO) << "Eval() function : " << function_name_str;

      iree_string_view_t function_name =
          iree_make_cstring_view(function_name_str.c_str());

      iree_status_t status = iree_runtime_call_function(
          iree_runtime_session_, function_name, context, node);

      if (!iree_status_is_ok(status)) {
        LOG(ERROR) << "iree_runtime_call_function failed: "
                   << iree_status_code_string(iree_status_code(status));
        return kTfLiteError;
      }
      LOG(INFO) << "Eval done()";
    }
    return kTfLiteOk;
  }

 private:
  TfLiteType GetType(TfLiteContext* context, int tensor_index) {
    return context->tensors[tensor_index].type;
  }

  bool HasSameType(TfLiteContext* context, std::vector<int>& indexes) {
    TfLiteType type = kTfLiteNoType;
    bool is_type_unset = true;
    for (int index : indexes) {
      TfLiteType tensor_type = GetType(context, index);
      if (is_type_unset) {
        type = tensor_type;
      } else {
        if (tensor_type != type) {
          return false;
        }
      }
    }
    return true;
  }

  std::string GetTypeSuffix(TfLiteType type) {
    switch (type) {
      case kTfLiteNoType:
        return "notype";
      case kTfLiteFloat32:
        return "f32";
      case kTfLiteInt32:
        return "i32";
      case kTfLiteUInt8:
        return "ui8";
      case kTfLiteInt64:
        return "i64";
      case kTfLiteString:
        return "str";
      case kTfLiteBool:
        return "bool";
      case kTfLiteInt16:
        return "i16";
      case kTfLiteComplex64:
        return "complex64";
      case kTfLiteInt8:
        return "i8";
      case kTfLiteFloat16:
        return "f16";
      case kTfLiteFloat64:
        return "f64";
      case kTfLiteComplex128:
        return "complex128";
      case kTfLiteUInt64:
        return "ui64";
      case kTfLiteResource:
        return "resource";
      case kTfLiteVariant:
        return "variant";
      case kTfLiteUInt32:
        return "ui32";
      case kTfLiteUInt16:
        return "ui16";
      case kTfLiteInt4:
        return "i4";
      default:
        return "";
    }
  }

  std::string GetFunctionName(TfLiteContext* context, int builtin_code,
                              std::vector<int>& inputs,
                              std::vector<int>& outputs) {
    std::string name = "module.";
    switch (builtin_code) {
      case kTfLiteBuiltinAdd:
      case kTfLiteBuiltinStablehloAdd:
        name += "add";
        break;
      default:
        return "";
    }

#if 1
    for (int index : inputs) {
      name += "_";
      TfLiteTensor* tensor = &context->tensors[index];
      TfLiteType type = tensor->type;
      for (int i = 0; i < tensor->dims->size; ++i) {
        name += std::to_string(tensor->dims->data[i]);
        name += "x";
      }
      name += GetTypeSuffix(type);
    }
    // The output types also appear in the function name.
    for (int index : outputs) {
      name += "_";
      TfLiteTensor* tensor = &context->tensors[index];
      TfLiteType type = tensor->type;
      for (int i = 0; i < tensor->dims->size; ++i) {
        name += std::to_string(tensor->dims->data[i]);
        name += "x";
      }
      name += GetTypeSuffix(type);
    }
#else
    // Handle the type. When all the input types are the same, the type is
    // specified once in the function name. Otherwise, all types are
    // specified.
    if (HasSameType(context, inputs)) {
      TfLiteType type = GetType(context, inputs[0]);
      name += GetTypeSuffix(type);
    } else {
      // When there is a different input type, all types are specified.
      for (int index : inputs) {
        TfLiteType type = GetType(context, inputs[index]);
        name += GetTypeSuffix(type);
      }
    }
#endif
    return name;
  }

  // Holds the indices of the input/output tensors.
  // inputs_[i] is list of all input tensors to node at index 'i'.
  // outputs_[i] is list of all output tensors to node at index 'i'.
  std::vector<std::vector<int>> inputs_, outputs_;
  // Holds the builtin code of the ops.
  // builtin_code_[i] is the type of node at index 'i'
  std::vector<int> builtin_code_;

  const TfLiteIreeDelegateOptions options_;
  iree_runtime_instance_options_t iree_runtime_instance_options_;
  iree_runtime_instance_t* iree_runtime_instance_ = nullptr;
  iree_runtime_session_t* iree_runtime_session_ = nullptr;
  iree_runtime_session_options_t iree_runtime_session_options_;
  const char* module_path_cstr_ = nullptr;
  std::vector<unsigned char> vmfb_content_;
};

// IreeDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class IreeDelegate : public SimpleDelegateInterface {
 public:
  explicit IreeDelegate(const TfLiteIreeDelegateOptions& options)
      : options_(options) {}
  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    // TODO: check inputs
    switch (registration->builtin_code) {
      case kTfLiteBuiltinAdd:
        // case kTfLiteBuiltinStablehloAdd:
        {
          if (node->inputs->size != 2) return false;
          if (node->outputs->size != 1) return false;

          return true;
        }
    }
    // TODO: implement
    return false;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override {
    if (context->delegate_data == nullptr || context->delegate_data_size == 0) {
      return kTfLiteError;
    }
    LOG(INFO) << "IreeDelegate::Initialize(): vmfb module size = "
              << context->delegate_data_size << " bytes";

    // TODO: move the module loading here.

    return kTfLiteOk;
  }

  const char* Name() const override {
    static constexpr char kName[] = "IreeDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<IreeDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

 private:
  const TfLiteIreeDelegateOptions options_;
};

}  // namespace iree_test
}  // namespace tflite

TfLiteIreeDelegateOptions TfLiteIreeDelegateOptionsDefault() {
  TfLiteIreeDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this iree test delegate will
  // not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteIreeDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteIreeDelegateCreate(
    const TfLiteIreeDelegateOptions* options) {
  std::unique_ptr<tflite::iree_test::IreeDelegate> iree(
      new tflite::iree_test::IreeDelegate(
          options ? *options : TfLiteIreeDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(iree));
}

// Destroys a delegate created with `TfLiteIreeDelegateCreate` call.
void TfLiteIreeDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
