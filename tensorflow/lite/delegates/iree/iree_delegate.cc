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

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <utility>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/delegates/iree/iree_runtime_call.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"

constexpr char kVMFBPath[] = "VMFB_PATH";

namespace tflite {
namespace iree_test {

// Iree delegate kernel.
class IreeDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit IreeDelegateKernel(const TfLiteIreeDelegateOptions& options)
      : options_(options) {}

  virtual ~IreeDelegateKernel() {
    iree_runtime_session_release(iree_runtime_session_);
    iree_runtime_instance_release(iree_runtime_instance_);
  }

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    module_path_cstr_ = std::getenv(kVMFBPath);
    if (!module_path_cstr_) {
      return kTfLiteError;
    }
    // Set up the shared runtime instance.
    // An application should usually only have one of these and share it across
    // all of the sessions it has. The instance is thread-safe, while the
    // sessions are only thread-compatible (you need to lock if its required).
    iree_runtime_instance_options_initialize(&iree_runtime_instance_options_);
    iree_runtime_instance_options_use_all_available_drivers(
        &iree_runtime_instance_options_);
    iree_runtime_instance_ = NULL;
    iree_status_t status = iree_runtime_instance_create(
        &iree_runtime_instance_options_, iree_allocator_system(),
        &iree_runtime_instance_);

    iree_hal_device_t* iree_device = nullptr;

    if (iree_status_is_ok(status)) {
      status = iree_runtime_instance_try_create_default_device(
          iree_runtime_instance_, iree_make_cstring_view("local-task"),
          &iree_device);
    }

    // Set up the session to run the module.
    // Sessions are like OS processes and are used to isolate modules from
    // each other and hold runtime state such as the variables used within the
    // module. The same module loaded into two sessions will see their own
    // private state.
    if (iree_status_is_ok(status)) {
      iree_runtime_session_options_initialize(&iree_runtime_session_options_);
      iree_status_t status = iree_runtime_session_create_with_device(
          iree_runtime_instance_, &iree_runtime_session_options_, iree_device,
          iree_runtime_instance_host_allocator(iree_runtime_instance_),
          &iree_runtime_session_);
      // Session keeps device reference internally.
      iree_hal_device_release(iree_device);
    }

    // Load the compiled user module in a demo-specific way.
    // Applications could specify files, embed the outputs directly in their
    // binaries, fetch them over the network, etc.
    if (iree_status_is_ok(status)) {
      status = iree_runtime_session_append_bytecode_module_from_file(
          iree_runtime_session_, module_path_cstr_);
    }

    // Build and issue the call.
    if (iree_status_is_ok(status)) {
      return !options_.error_during_init ? kTfLiteOk : kTfLiteError;
    }
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    return !options_.error_during_prepare ? kTfLiteOk : kTfLiteError;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    if (options_.error_during_invoke) return kTfLiteError;

    // FIXME: get the function name from the TF op.
    const char* function_name_cstr = "module.add_4d_f32";
    iree_string_view_t function_name =
        iree_make_cstring_view(function_name_cstr);

    iree_status_t status = iree_runtime_call_function(
        iree_runtime_session_, function_name, context, node);
    return iree_status_is_ok(status) ? kTfLiteOk : kTfLiteError;
  }

 private:
  const TfLiteIreeDelegateOptions options_;
  iree_runtime_instance_options_t iree_runtime_instance_options_;
  iree_runtime_instance_t* iree_runtime_instance_ = nullptr;
  iree_runtime_session_t* iree_runtime_session_ = nullptr;
  iree_runtime_session_options_t iree_runtime_session_options_;
  const char* module_path_cstr_ = nullptr;
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
      case kTfLiteBuiltinAdd: {
        if (node->inputs->size != 2) return false;
        if (node->outputs->size != 1) return false;

        return true;
      }
    }
    // TODO: implement
    return false;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

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
