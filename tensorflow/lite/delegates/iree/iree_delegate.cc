/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <utility>

#include "tensorflow/lite/delegates/utils/simple_delegate.h"

namespace tflite {
namespace iree_test {

// Iree delegate kernel.
class IreeDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit IreeDelegateKernel(const TfLiteIreeDelegateOptions& options)
      : options_(options) {}

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    return !options_.error_during_init ? kTfLiteOk : kTfLiteError;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    return !options_.error_during_prepare ? kTfLiteOk : kTfLiteError;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    return !options_.error_during_invoke ? kTfLiteOk : kTfLiteError;
  }

 private:
  const TfLiteIreeDelegateOptions options_;
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
    return options_.allowed_builtin_code == registration->builtin_code;
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
TfLiteDelegate* TfLiteIreeDelegateCreate(const TfLiteIreeDelegateOptions* options) {
  std::unique_ptr<tflite::iree_test::IreeDelegate> iree(
      new tflite::iree_test::IreeDelegate(
          options ? *options : TfLiteIreeDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(iree));
}

// Destroys a delegate created with `TfLiteIreeDelegateCreate` call.
void TfLiteIreeDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
