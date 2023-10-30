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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_EMIT_IREE_VMFB_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_EMIT_IREE_VMFB_H_

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "iree/compiler/embedding_api.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/OperationSupport.h"
#include "mlir/include/mlir/IR/Value.h"

namespace mlir {
namespace TFL {

/// Check if an op is unsupported by TFLite but supported by IREE.
bool IsUnsupportedOp(mlir::Operation* op);

std::string GetFunctionName(mlir::Operation* op);

struct IREECompiler {
  IREECompiler() = delete;
  ~IREECompiler();
  explicit IREECompiler(std::vector<std::string>&& addl_args);
  void Run(std::vector<mlir::Operation*>& ops);
  void* GetVMFBBuffer();
  uint64_t GetVMFBBufferSize();

 private:
  void CreateVMFBOutputs();
  void CreateFuncForEachOp(std::vector<mlir::Operation*>& ops);
  bool Initialize();
  void InitializeOutputBuffer();
  void FinalizeOutputBuffer();
  void CreateSession();
  void DestroySession();
  std::string CreateFunctionBody(mlir::Operation* op, std::string func_name);
  mlir::Operation* CreateStableHloOp(mlir::OpBuilder& builder,
                                     mlir::Operation* op);
  std::string mlir_module_;
  void* vmfb_buffer_;
  uint64_t vmfb_buffer_size_;
  iree_compiler_source_t* source_;
  iree_compiler_session_t* session_;
  iree_compiler_invocation_t* invocation_;
  iree_compiler_output_t* compiler_output_;
  std::unordered_set<std::string> generated_funcs_;
  std::vector<std::string> args_;
};
}  // namespace TFL
};  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_EMIT_IREE_VMFB_H_
