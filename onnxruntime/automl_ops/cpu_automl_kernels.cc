// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "automl_ops/cpu_automl_kernels.h"
#include "core/graph/constants.h"
#include "core/framework/data_types.h"

namespace onnxruntime {
namespace automl {

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, DateTimeTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, string, StringTransformer);

Status RegisterCpuAutoMLKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
     // add more kernels here
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, DateTimeTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float, StringTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, string, StringTransformer)>
  };

  for (auto& function_table_entry : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
  }
  return Status::OK();
}

}  // namespace automl
}  // namespace onnxruntime
