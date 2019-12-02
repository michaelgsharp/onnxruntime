// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include "core/automl/featurizers/src/FeaturizerPrep/Featurizers/OneHotEncoderFeaturizer.h"

using namespace std;
namespace featurizers = Microsoft::Featurizer;

namespace onnxruntime {
namespace automl {

template <typename InputT>
class OneHotEncoderTransformer final : public OpKernel {
public:
  explicit OneHotEncoderTransformer(const OpKernelInfo &info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext *ctx) const override {
    // Create the transformer
    featurizers::OneHotEncoderTransformer<InputT> transformer(
      [ctx](void) {
        auto state_tensor(ctx->Input<Tensor>(0));
        uint8_t const * const state_data(state_tensor->Data<uint8_t>());

        Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
        return featurizers::OneHotEncoderTransformer<InputT>(archive);
      }()
    );

    // Get the input
    auto input_tensor(ctx->Input<Tensor>(1));
    InputT * const input_data(input_tensor->Data<InputT>());

    // Prepare the output
    Tensor * index_tensor(ctr->Output(0, input_tensor->Shape()));
    Tensor * size_tensor(ctr->Output(1, input_tensor->Shape()));
    Tensor * appearances_tensor(ctr->Output(2, input_tensor->Shape()));

    uint32_t * index_data(index_tensor->MutableData<uint32_t>());
    uint32_t * size_data(size_tensor->MutableData<uint32_t>());
    uint32_t * appearances_data(appearances_tensor->MutableData<uint32_t>());

    // Execute
    int64_t const length(input_tensor->Shape().GetDims()[0]);

    for(int64_t i = 0; i < length; ++i) {
      auto result(transformer.execute(input_data[i]));

      index_data[i] = std::move(result.index);
      size_data[i] = std::move(result.size);
      appearances_data[i] = std::move(result.appearances);
    }

    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    OneHotEncoderTransformer,
    kMSAutoMLDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("OutputT0", DataTypeImpl::GetTensorType<uint32_t>())
);

} // namespace automl
} // namespace onnxruntime
