// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include "core/automl/featurizers/src/FeaturizerPrep/Featurizers/StringFeaturizer.h"

namespace dtf = Microsoft::Featurizer::Featurizers;

namespace onnxruntime {
namespace automl {

template <typename T>
class StringTransformer final : public OpKernel {
 public:
  explicit StringTransformer(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
Status StringTransformer<T>::Compute(OpKernelContext* ctx) const {
 

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    StringTransformer,
    kMSAutoMLDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<uint16_t>())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<std::string>()),
    StringTransformer);
}  // namespace automl
}  // namespace onnxruntime
