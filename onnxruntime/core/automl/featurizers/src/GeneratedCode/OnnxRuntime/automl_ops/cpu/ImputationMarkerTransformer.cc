// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include "core/automl/featurizers/src/FeaturizerPrep/Featurizers/ImputationMarkerFeaturizer.h"

using namespace std;
namespace featurizers = Microsoft::Featurizer;

namespace onnxruntime {
namespace automl {

inline float_t const & PreprocessOptional(float_t const &value) { return value; }
inline double_t const & PreprocessOptional(double_t const &value) { return value; }
inline nonstd::optional<string> PreprocessOptional(string value) { return value.empty() ? nonstd::optional<string>() : nonstd::optional<string>(std::move(value)); }

template <typename InputT>
class ImputationMarkerTransformer final : public OpKernel {
public:
  explicit ImputationMarkerTransformer(const OpKernelInfo &info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext *ctx) const override {
    // Create the transformer
    featurizers::ImputationMarkerTransformer<InputT> transformer(
      [ctx](void) {
        auto state_tensor(ctx->Input<Tensor>(0));
        uint8_t const * const state_data(state_tensor->Data<uint8_t>());

        Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
        return featurizers::ImputationMarkerTransformer<InputT>(archive);
      }()
    );

    // Get the input
    auto input_tensor(ctx->Input<Tensor>(1));
    InputT * const input_data(input_tensor->Data<InputT>());

    // Prepare the output
    Tensor * output_tensor(ctx->Output(0, input_tensor->Shape()));
    bool * output_data(output_tensor->MutableData<bool>());

    // Execute
    int64_t const length(input_tensor->Shape().GetDims()[0]);

    for(int64_t i = 0; i < length; ++i) {
      output_data[i] = transformer.execute(PreprocessOptional(input_data[i]));
    }

    return Status::OK();
  }
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    ImputationMarkerTransformer,
    kMSAutoMLDomain,
    1,
    float_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<float_t>()),
    ImputationMarkerTransformer<float_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    ImputationMarkerTransformer,
    kMSAutoMLDomain,
    1,
    double_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<double_t>()),
    ImputationMarkerTransformer<double_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    ImputationMarkerTransformer,
    kMSAutoMLDomain,
    1,
    string,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<string>()),
    ImputationMarkerTransformer<string>
);

} // namespace automl
} // namespace onnxruntime
