// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include "core/automl/featurizers/src/FeaturizerPrep/Featurizers/CatImputerFeaturizer.h"

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
Status CategoryImputer<T>::Compute(OpKernelContext* ctx) const {
  /*dtf::CatImputerEstimator<T>::Transformer transformer;

  auto input_tensor = ctx->Input<Tensor>(0);
  const T* input_data = input_tensor->Data<T>();

  Tensor* string_tensor = ctx->Output(0, input_tensor->Shape());
  std::string* string_data = string_tensor->MutableData<std::string>();

  const int64_t length = input_tensor->Shape().GetDims()[0];

  for (int i = 0; i < length; i++) {
    string_data[i] = transformer.execute(input_data[i]);
  }
*/
  return Status::OK();
}

#define REG_STRINGFEATURIZER(in_type)                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      CategoryImputer,                                                \
      kMSAutoMLDomain,                                                  \
      1,                                                                \
      in_type,                                                          \
      kCpuExecutionProvider,                                            \
      KernelDefBuilder()                                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<in_type>()), \
      CategoryImputer<in_type>);

REG_STRINGFEATURIZER(int8_t);
REG_STRINGFEATURIZER(int16_t);
REG_STRINGFEATURIZER(int32_t);
REG_STRINGFEATURIZER(int64_t);
REG_STRINGFEATURIZER(uint8_t);
REG_STRINGFEATURIZER(uint16_t);
REG_STRINGFEATURIZER(uint32_t);
REG_STRINGFEATURIZER(uint64_t);
REG_STRINGFEATURIZER(float_t);
REG_STRINGFEATURIZER(double_t);
REG_STRINGFEATURIZER(bool);

using namespace std;
REG_STRINGFEATURIZER(string);




}  // namespace automl
}  // namespace onnxruntime
