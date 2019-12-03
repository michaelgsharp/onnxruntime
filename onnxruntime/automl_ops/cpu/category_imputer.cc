//// Copyright (c) Microsoft Corporation. All rights reserved.
//// Licensed under the MIT License.
//
//#include "core/common/common.h"
//#include "core/framework/data_types.h"
//#include "core/framework/op_kernel.h"
//
//#include "core/automl/featurizers/src/FeaturizerPrep/Featurizers/CatImputerFeaturizer.h"
//#include "core/automl/featurizers/src/FeaturizerPrep/Archive.h"
//
//namespace dtf = Microsoft::Featurizer::Featurizers;
//
//namespace onnxruntime {
//namespace automl {
//
//template <typename T>
//class CategoryImputer final : public OpKernel {
// public:
//  explicit CategoryImputer(const OpKernelInfo& info) : OpKernel(info) {}
//  Status Compute(OpKernelContext* context) const override;
//};
//
//template <typename T>
//T Execute(typename dtf::CatImputerEstimator<T>::Transformer& transformer, const T& data) {
//  return transformer.execute(data);
//}
//
//template <>
//std::string Execute<std::string>(dtf::CatImputerEstimator<std::string>::Transformer& transformer, const std::string& data) {
//  return transformer.execute(data.empty() ? nonstd::optional<std::string>() : data);
//}
//
//template <typename T>
//Status CategoryImputer<T>::Compute(OpKernelContext* ctx) const {
//  auto state_tensor = ctx->Input<Tensor>(0);
//  const uint8_t* state_data = state_tensor->Data<uint8_t>();
//
//  Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
//  dtf::CatImputerEstimator<T>::Transformer transformer(archive);
//
//  auto input_tensor = ctx->Input<Tensor>(1);
//  const T* input_data = input_tensor->Data<T>();
//
//  Tensor* output_tensor = ctx->Output(0, input_tensor->Shape());
//  T* output_data = output_tensor->MutableData<T>();
//
//  const int64_t length = input_tensor->Shape().GetDims()[0];
//
//  for (int i = 0; i < length; i++) {
//
//    output_data[i] = Execute(transformer, input_data[i]);
//  }
//
//  return Status::OK();
//}
//
//#define REG_CATIMPUTERFEATURIZER(in_type)                                   \
//  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
//      CategoryImputer,                                                  \
//      kMSAutoMLDomain,                                                  \
//      1,                                                                \
//      in_type,                                                          \
//      kCpuExecutionProvider,                                            \
//      KernelDefBuilder()                                                \
//          .TypeConstraint("T", DataTypeImpl::GetTensorType<in_type>()), \
//      CategoryImputer<in_type>);
//
//REG_CATIMPUTERFEATURIZER(float_t);
//REG_CATIMPUTERFEATURIZER(double_t);
//
//using namespace std;
//REG_CATIMPUTERFEATURIZER(string);
//
//}  // namespace automl
//}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include "core/automl/featurizers/src/Featurizers/CatImputerFeaturizer.h"
#include "core/automl/featurizers/src/Traits.h"

using namespace std;
namespace featurizers = Microsoft::Featurizer::Featurizers;

namespace onnxruntime {
namespace automl {

template <typename T>
struct OutputTypeMapper {};
template <>
struct OutputTypeMapper<float_t> { using type = float_t; };
template <>
struct OutputTypeMapper<double_t> { using type = double_t; };
template <>
struct OutputTypeMapper<string> { using type = string; };

inline float_t const& PreprocessOptional(float_t const& value) { return value; }
inline double_t const& PreprocessOptional(double_t const& value) { return value; }
inline nonstd::optional<string> PreprocessOptional(string value) { return value.empty() ? nonstd::optional<string>() : nonstd::optional<string>(std::move(value)); }

template <typename T>
class CatImputerTransformer final : public OpKernel {
 public:
  explicit CatImputerTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    // Create the transformer

    auto state_tensor(ctx->Input<Tensor>(0));
    uint8_t const* const state_data(state_tensor->Data<uint8_t>());
    Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);

    featurizers::CatImputerTransformer<Microsoft::Featurizer::Traits<T>::nullable_type, typename OutputTypeMapper<T>::type> transformer(archive);

    // Get the input
    auto input_tensor(ctx->Input<Tensor>(1));
    const T* input_data(input_tensor->Data<T>());

    // Prepare the output
    Tensor* output_tensor(ctx->Output(0, input_tensor->Shape()));
    typename OutputTypeMapper<T>::type* output_data(output_tensor->MutableData<typename OutputTypeMapper<T>::type>());

    // Execute
    int64_t const length(input_tensor->Shape().GetDims()[0]);

    for (int64_t i = 0; i < length; ++i) {
      output_data[i] = transformer.execute(PreprocessOptional(input_data[i]));
    }

    return Status::OK();
  }
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    CatImputerTransformer,
    kMSAutoMLDomain,
    1,
    float_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float_t>()),
    CatImputerTransformer<float_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    CatImputerTransformer,
    kMSAutoMLDomain,
    1,
    double_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<double_t>()),
    CatImputerTransformer<double_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    CatImputerTransformer,
    kMSAutoMLDomain,
    1,
    string,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<string>()),
    CatImputerTransformer<string>);

}  // namespace automl
}  // namespace onnxruntime
