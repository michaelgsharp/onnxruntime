// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include "core/automl/featurizers/src/Featurizers/MaxAbsScalarFeaturizer.h"
#include "core/automl/featurizers/src/Archive.h"

namespace dtf = Microsoft::Featurizer::Featurizers;

namespace onnxruntime {
namespace automl {

template <typename T>
class MaxAbsScaler final : public OpKernel {
 public:
  explicit MaxAbsScaler(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override;
};

template <typename InputT, typename OutputT>
void Execute(Microsoft::Featurizer::Archive& archive, const InputT* input_data, Tensor* output_tensor, const int64_t& length) {
  dtf::MaxAbsScalarEstimator<InputT, OutputT>::TransformerType transformer(archive);

  OutputT* output_data = output_tensor->MutableData<OutputT>();

  for (int i = 0; i < length; i++) {
    output_data[i] = transformer.execute(input_data[i]);
  }
}

template <typename T>
Status MaxAbsScaler<T>::Compute(OpKernelContext* ctx) const {
  auto state_tensor = ctx->Input<Tensor>(0);
  const uint8_t* state_data = state_tensor->Data<uint8_t>();

  Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);

  auto input_tensor = ctx->Input<Tensor>(1);
  const T* input_data = input_tensor->Data<T>();

  Tensor* output_tensor = ctx->Output(0, input_tensor->Shape());

  const int64_t length = input_tensor->Shape().GetDims()[0];

  if (std::is_same<T, int8_t>::value ||
      std::is_same<T, int16_t>::value ||
      std::is_same<T, uint8_t>::value ||
      std::is_same<T, uint16_t>::value ||
      std::is_same<T, float_t>::value) {
    Execute<T, float>(archive, input_data, output_tensor, length);
  } else {
    Execute<T, double>(archive, input_data, output_tensor, length);
  }

  return Status::OK();
}

#define REG_MAXABSSCALERFEATURIZER(in_type)                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      MaxAbsScaler,                                                     \
      kMSAutoMLDomain,                                                  \
      1,                                                                \
      in_type,                                                          \
      kCpuExecutionProvider,                                            \
      KernelDefBuilder()                                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<in_type>()), \
      MaxAbsScaler<in_type>);

REG_MAXABSSCALERFEATURIZER(int8_t);
REG_MAXABSSCALERFEATURIZER(int16_t);
REG_MAXABSSCALERFEATURIZER(int32_t);
REG_MAXABSSCALERFEATURIZER(int64_t);
REG_MAXABSSCALERFEATURIZER(uint8_t);
REG_MAXABSSCALERFEATURIZER(uint16_t);
REG_MAXABSSCALERFEATURIZER(uint32_t);
REG_MAXABSSCALERFEATURIZER(uint64_t);
REG_MAXABSSCALERFEATURIZER(float_t);
REG_MAXABSSCALERFEATURIZER(double_t);

}  // namespace automl
}  // namespace onnxruntime
