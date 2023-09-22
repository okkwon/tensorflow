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
#include <string>
#include <utility>

#include "tensorflow/lite/delegates/iree/iree_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace tflite {
namespace tools {

class IreeDelegateProvider : public DelegateProvider {
 public:
  IreeDelegateProvider() {
    default_params_.AddParam("use_iree", ToolParam::Create<bool>(false));
    default_params_.AddParam("iree_lib_path",
                             ToolParam::Create<std::string>("/data/local/tmp"));
    default_params_.AddParam("iree_profiling", ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "Iree"; }
};
REGISTER_DELEGATE_PROVIDER(IreeDelegateProvider);

std::vector<Flag> IreeDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {
      CreateFlag<bool>("use_iree", params, "Use Iree delegate"),
      CreateFlag<std::string>(
          "iree_lib_path", params,
          "The library path for the underlying Iree libraries. The library "
          "path ONLY for the libiree_nn_skel*.so files. For "
          "libiree_interface.so, it needs to be on a system library search "
          "path such as LD_LIBRARY_PATH."),
      CreateFlag<bool>("iree_profiling", params, "Enables Iree profiling")};
  return flags;
}

void IreeDelegateProvider::LogParams(const ToolParams& params,
                                     bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_iree", "Use Iree", verbose);
  LOG_TOOL_PARAM(params, std::string, "iree_lib_path", "Iree lib path",
                 verbose);
  LOG_TOOL_PARAM(params, bool, "iree_profiling", "Iree profiling", verbose);
}

TfLiteDelegatePtr IreeDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  TfLiteIreeDelegateOptions options = {0};
  // options.print_graph_profile = params.Get<bool>("iree_profiling");
  // options.max_delegated_partitions =
  //     params.Get<int>("max_delegated_partitions");
  // options.min_nodes_per_partition =
  TfLiteDelegatePtr delegate = evaluation::CreateIreeDelegate(&options);

  if (!delegate.get()) {
    TFLITE_LOG(WARN)
        << "Could not create Iree delegate: platform may not support "
           "delegate or required libraries are missing";
  }
  return delegate;
}

std::pair<TfLiteDelegatePtr, int>
IreeDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  int rank = 0;
  rank = params.GetPosition<bool>("use_iree");
  return std::make_pair(std::move(ptr), rank);
}

}  // namespace tools
}  // namespace tflite
