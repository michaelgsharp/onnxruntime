// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include "core/automl/featurizers/src/FeaturizerPrep/Featurizers/CatImputerFeaturizer.h"
#include "core/automl/featurizers/src/FeaturizerPrep/Archive.h"

namespace dtf = Microsoft::Featurizer::Featurizers;

namespace onnxruntime {
namespace automl {

template <typename T>
class CategoryImputer final : public OpKernel {
 public:
  explicit CategoryImputer(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
T Execute(typename dtf::CatImputerEstimator<T>::Transformer& transformer, const T& data) {
  return transformer.execute(data);
}

template <>
std::string Execute<std::string>(dtf::CatImputerEstimator<std::string>::Transformer& transformer, const std::string& data) {
  return transformer.execute(data.empty() ? nonstd::optional<std::string>() : data);
}

template <typename T>
Status CategoryImputer<T>::Compute(OpKernelContext* ctx) const {
  auto state_tensor = ctx->Input<Tensor>(0);
  const uint8_t* state_data = state_tensor->Data<uint8_t>();

  Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
  dtf::CatImputerEstimator<T>::Transformer transformer(archive);

  auto input_tensor = ctx->Input<Tensor>(1);
  const T* input_data = input_tensor->Data<T>();

  Tensor* output_tensor = ctx->Output(0, input_tensor->Shape());
  T* output_data = output_tensor->MutableData<T>();

  const int64_t length = input_tensor->Shape().GetDims()[0];

  for (int i = 0; i < length; i++) {

    output_data[i] = Execute(transformer, input_data[i]);
  }

  return Status::OK();
}

#define REG_STRINGFEATURIZER(in_type)                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      CategoryImputer,                                                  \
      kMSAutoMLDomain,                                                  \
      1,                                                                \
      in_type,                                                          \
      kCpuExecutionProvider,                                            \
      KernelDefBuilder()                                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<in_type>()), \
      CategoryImputer<in_type>);

REG_STRINGFEATURIZER(float_t);
REG_STRINGFEATURIZER(double_t);

using namespace std;
REG_STRINGFEATURIZER(string);

}  // namespace automl
}  // namespace onnxruntime
