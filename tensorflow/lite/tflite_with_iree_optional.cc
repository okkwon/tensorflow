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
#include "tensorflow/lite/tflite_with_iree_optional.h"

#include <memory>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/macros.h"

#ifdef TFLITE_BUILD_WITH_IREE_DELEGATE
#include "tensorflow/lite/delegates/iree/iree_delegate.h"
#endif

namespace tflite {

using TfLiteDelegatePtr =
    std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>;

#ifdef TFLITE_BUILD_WITH_IREE_DELEGATE
TfLiteDelegatePtr MaybeCreateIREEDelegate(TfLiteContext* context) {
  TfLiteIreeDelegateOptions options = TfLiteIreeDelegateOptionsDefault();

  return TfLiteDelegatePtr(TfLiteIreeDelegateCreate(&options),
                           TfLiteIreeDelegateDelete);
}
#else
// Using weak symbols to create a delegate allows automatic injection of the
// delegate simply by adding it as a dependency. See the strong override in
// lite/tflite_with_iree.cc,
TFLITE_ATTRIBUTE_WEAK TfLiteDelegatePtr AcquireIREEDelegate() {
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

TfLiteDelegatePtr MaybeCreateIREEDelegate(TfLiteContext* context) {
  return AcquireIREEDelegate();
}
#endif

}  // namespace tflite
