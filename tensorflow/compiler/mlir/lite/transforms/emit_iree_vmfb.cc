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

#include "tensorflow/compiler/mlir/lite/transforms/emit_iree_vmfb.h"

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "iree/compiler/embedding_api.h"
#include "iree/compiler/loader.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/Block.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/Region.h"
#include "mlir/include/mlir/IR/TypeUtilities.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/ValueRange.h"
#include "mlir/include/mlir/Pass/PassManager.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
// namespace {

static bool IsUnsupportedTypeForBinaryOp(mlir::Type element_type) {
  if (element_type.isF16() || element_type.isF64() ||
      element_type.isInteger(8) || element_type.isInteger(16) ||
      element_type.isInteger(32) || element_type.isInteger(64)) {
    return true;
  }
  if (auto complex_type = element_type.dyn_cast<ComplexType>()) {
    return complex_type.getElementType().isF32() ||
           complex_type.getElementType().isF64();
  }
  return false;
}

bool IsUnsupportedOp(mlir::Operation* op) {
  if (isa<mlir::TFL::AddOp>(op)) {
    auto element_type = getElementTypeOrSelf(op->getResultTypes()[0]);
    return IsUnsupportedTypeForBinaryOp(element_type);
  }
  return false;
}

static std::string GetOpName(mlir::Operation* op) {
  if (isa<mlir::TFL::AddOp>(op)
      // || isa<mlir::stablehlo::AddOp>(op)
  ) {
    return "add";
  }

  return "unsupported_op_" + op->getName().getStringRef().str();
}

static std::string GetTypeStr(mlir::Type type) {
  std::string type_str;
  llvm::raw_string_ostream os(type_str);
  auto tensor_type = cast<RankedTensorType>(type);
  auto element_type = tensor_type.getElementType();

  for (int dim : tensor_type.getShape()) {
    os << dim << "x";
  }

  if (element_type.isIntOrFloat()) {
    element_type.print(os);
  } else if (auto complex_type = element_type.dyn_cast<ComplexType>()) {
    if (complex_type.getElementType().isF32()) {
      os << "complex64";
    } else if (complex_type.getElementType().isF64()) {
      os << "complex128";
    } else {
      // unsupported complex element type
      os << "unsupported_complex_element_type";
    }
  } else {
    os << "unsupported_element_type";
  }
  os.flush();
  return type_str;
}

std::string GetFunctionName(mlir::Operation* op) {
  std::string name = GetOpName(op);

  for (auto operand : op->getOperands()) {
    name += "_" + GetTypeStr(operand.getType());
  }
  for (auto result : op->getResults()) {
    name += "_" + GetTypeStr(result.getType());
  }
  LOG(INFO) << "Function name: " << name;
  return name;
}

IREECompiler::IREECompiler(std::vector<std::string>&& addl_args) {
  LOG(INFO) << "EmitIREEVMFBPass()";
  args_ = std::move(addl_args);
  session_ = nullptr;
  vmfb_buffer_ = nullptr;
  vmfb_buffer_size_ = 0;
  compiler_output_ = nullptr;
  invocation_ = nullptr;
  source_ = nullptr;
}

IREECompiler::~IREECompiler() { DestroySession(); }

void IREECompiler::InitializeOutputBuffer() { mlir_module_ = "module {\n"; }

void IREECompiler::FinalizeOutputBuffer() { mlir_module_ += "}\n"; }

void IREECompiler::Run(std::vector<mlir::Operation*>& ops) {
  LOG(INFO) << "Run()";
  Initialize();
  CreateSession();
  InitializeOutputBuffer();

  CreateFuncForEachOp(ops);

  FinalizeOutputBuffer();
  CreateVMFBOutputs();
}

void IREECompiler::CreateFuncForEachOp(std::vector<mlir::Operation*>& ops) {
  LOG(INFO) << "CreateFuncForEachOp";

  for (auto op : ops) {
    std::string function_name = GetFunctionName(op);
    if (generated_funcs_.find(function_name) != generated_funcs_.end()) {
      continue;
    }
    generated_funcs_.insert(function_name);

    // FIXME: Create a ModuleOp instead of concatenating strings.
    mlir_module_ += CreateFunctionBody(op, function_name);
  }
}

void IREECompiler::CreateVMFBOutputs() {
  LOG(INFO) << "CreateVMFBOutputs()";

  // Cannot compile an empty module.
  if (generated_funcs_.empty()) {
    LOG(INFO) << "No generated functions";
    return;
  }

  if (source_ != nullptr) {
    LOG(ERROR) << "source must be null";
    return;
  }

  if (iree_compiler_error_t* error =
          ireeCompilerSourceWrapBuffer(session_, "", mlir_module_.c_str(),
                                       mlir_module_.size(), false, &source_)) {
    LOG(ERROR) << ireeCompilerErrorGetMessage(error);
    ireeCompilerErrorDestroy(error);
    return;
  }
  invocation_ = ireeCompilerInvocationCreate(session_);
  ireeCompilerInvocationEnableConsoleDiagnostics(invocation_);
  if (!ireeCompilerInvocationParseSource(invocation_, source_)) {
    LOG(ERROR) << "IREE Compiler Invocation ParseSource returned an ERROR!";
    return;
  }

  ireeCompilerInvocationSetVerifyIR(invocation_, true);

  if (!ireeCompilerInvocationPipeline(invocation_,
                                      IREE_COMPILER_PIPELINE_STD)) {
    LOG(ERROR) << "IREE Compiler Invocation Pipeline returned an ERROR!";
    return;
  }

  if (compiler_output_ != nullptr) {
    LOG(ERROR) << "compiler_output must be null";
    return;
  }

  if (iree_compiler_error_t* error =
          ireeCompilerOutputOpenMembuffer(&compiler_output_)) {
    LOG(ERROR) << ireeCompilerErrorGetMessage(error);
    ireeCompilerErrorDestroy(error);
    return;
  }
  if (iree_compiler_error_t* error = ireeCompilerInvocationOutputVMBytecode(
          invocation_, compiler_output_)) {
    LOG(ERROR) << ireeCompilerErrorGetMessage(error);
    ireeCompilerErrorDestroy(error);
    return;
  }

  if (iree_compiler_error_t* error = ireeCompilerOutputMapMemory(
          compiler_output_, &vmfb_buffer_, &vmfb_buffer_size_)) {
    LOG(ERROR) << ireeCompilerErrorGetMessage(error);
    ireeCompilerErrorDestroy(error);
    return;
  }
  LOG(ERROR) << "MLIR Source: " << mlir_module_;
  LOG(ERROR) << "VMFB Bytecode size: " << vmfb_buffer_size_;
}

mlir::Operation* IREECompiler::CreateStableHloOp(mlir::OpBuilder& builder,
                                                 mlir::Operation* op) {
  mlir::Operation* new_op = nullptr;
  if (isa<mlir::TFL::AddOp>(op) || isa<mlir::stablehlo::AddOp>(op)) {
    new_op =
        builder.create<mlir::stablehlo::AddOp>(op->getLoc(), op->getOperands());
  }
  return new_op;
}

void* IREECompiler::GetVMFBBuffer() { return vmfb_buffer_; }

uint64_t IREECompiler::GetVMFBBufferSize() { return vmfb_buffer_size_; }

std::string IREECompiler::CreateFunctionBody(mlir::Operation* op,
                                             std::string func_name) {
  // `inputs` and `outputs` are the inputs and outputs of the given subgraph.
  // We are currently handling a subgraph with a single op, but it can be
  // generalized.
  std::vector<mlir::Value> inputs;
  inputs.insert(inputs.end(), op->getOperands().begin(),
                op->getOperands().end());

  std::vector<mlir::Value> outputs;
  outputs.insert(outputs.end(), op->getResults().begin(),
                 op->getResults().end());

  // The function has operands for the outputs after the operands for the input.
  std::vector<Type> input_types;
  std::vector<Type> output_types;

  for (auto input : inputs) {
    input_types.push_back(input.getType());
  }
  for (auto output : outputs) {
    input_types.push_back(output.getType());
    output_types.push_back(output.getType());
  }

  op->getContext()->loadDialect<func::FuncDialect>();

  mlir::OpBuilder b(op->getContext());

  FunctionType func_type = b.getFunctionType(input_types, output_types);
  mlir::func::FuncOp func_op =
      mlir::func::FuncOp::create(op->getLoc(), func_name, func_type);

  // attach iree.abi.output to the operands for output
  auto* context = func_op.getContext();
  auto iree_abi_output = StringAttr::get(op->getContext(), "iree.abi.output");
  unsigned num_inputs = inputs.size();
  auto index_type = IndexType::get(context);
  for (unsigned int i = 0; i < outputs.size(); ++i) {
    auto output_index = IntegerAttr::get(index_type, i);
    NamedAttribute attr = {iree_abi_output, output_index};
    func_op.setArgAttrs(num_inputs + i, attr);
  }

  Block* entry_block = func_op.addEntryBlock();
  b.setInsertionPointToEnd(entry_block);

  IRMapping mapper;
  for (const auto& arg : llvm::enumerate(inputs)) {
    mapper.map(arg.value(), func_op.getArgument(arg.index()));
  }
  mlir::Operation* cloned_op = op->clone(mapper);
  mlir::Operation* new_op = CreateStableHloOp(b, cloned_op);
  if (new_op == nullptr) {
    new_op = b.clone(*op, mapper);
  }
  std::vector<mlir::Value> cloned_vals;
  cloned_vals.insert(cloned_vals.end(), new_op->result_begin(),
                     new_op->result_end());
  b.create<func::ReturnOp>(op->getLoc(), cloned_vals);
  std::string func_contents;
  llvm::raw_string_ostream func_contents_rso(func_contents);
  func_op.print(func_contents_rso);
  return func_contents;
}

bool IREECompiler::Initialize() {
  static bool is_compiler_lib_loaded = false;

  LOG(INFO) << "Initialize()";
  if (!is_compiler_lib_loaded) {
#if 0  // google3
    std::string loader_path = devtools_build::GetRunfilesDir() +
      "/google3/third_party/py/iree/tools/core/iree-lld";
    setenv("IREE_LLVM_EMBEDDED_LINKER_PATH", loader_path.c_str(),
           /*overwrite=*/0);
    std::string iree_lib_path =
      devtools_build::GetRunfilesDir() +
      "/google3/third_party/iree/lib/libIREECompiler.so";
    const char* compiler_lib_path = iree_lib_path.c_str();
#else
    const char* compiler_lib_path = getenv("LIB_IREE_COMPILER_PATH");
    if (compiler_lib_path == nullptr) {
      LOG(ERROR) << "LIB_IREE_COMPILER_PATH is not set.";
    }
#endif
    if (!ireeCompilerLoadLibrary(compiler_lib_path)) {
      LOG(ERROR) << "Failed to load libIREECompiler lib: " << compiler_lib_path;
      return false;
    }
    is_compiler_lib_loaded = true;
  }
  ireeCompilerGlobalInitialize();
  return true;
}

/* struct iree_compiler_session_t * */
void IREECompiler::CreateSession() {
  LOG(INFO) << "CreateSession()";
  char** args_ptr = new char*[args_.size()];
  for (int i = 0; i < args_.size(); i++) {
    args_ptr[i] = strdup(args_[i].c_str());
  }
  session_ = ireeCompilerSessionCreate();
  if (iree_compiler_error_t* error =
          ireeCompilerSessionSetFlags(session_, args_.size(), args_ptr)) {
    LOG(ERROR) << ireeCompilerErrorGetMessage(error);
    ireeCompilerErrorDestroy(error);
    session_ = nullptr;
  }
}

void IREECompiler::DestroySession() {
  LOG(INFO) << "DestroySession()";
  ireeCompilerInvocationDestroy(invocation_);
  invocation_ = nullptr;
  ireeCompilerSourceDestroy(source_);
  source_ = nullptr;
  ireeCompilerOutputDestroy(compiler_output_);
  compiler_output_ = nullptr;
  ireeCompilerSessionDestroy(session_);
  session_ = nullptr;
}

}  // namespace TFL
}  // namespace mlir
