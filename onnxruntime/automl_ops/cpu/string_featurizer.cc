// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

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

#define REG_STRINGFEATURIZER(in_type)                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      StringTransformer,                                                \
      kMSAutoMLDomain,                                                  \
      1,                                                                \
      in_type,                                                          \
      kCpuExecutionProvider,                                            \
      KernelDefBuilder()                                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<in_type>()), \
      StringTransformer<in_type>);                                      \

REG_STRINGFEATURIZER(int8_t);
REG_STRINGFEATURIZER(int16_t);
REG_STRINGFEATURIZER(int32_t);
REG_STRINGFEATURIZER(int64_t);
REG_STRINGFEATURIZER(uint8_t);
REG_STRINGFEATURIZER(uint16_t);
REG_STRINGFEATURIZER(uint32_t);
REG_STRINGFEATURIZER(uint64_t);
REG_STRINGFEATURIZER(float);
REG_STRINGFEATURIZER(double);
REG_STRINGFEATURIZER(bool);
using namespace std;
REG_STRINGFEATURIZER(string);



}  // namespace automl
}  // namespace onnxruntime
