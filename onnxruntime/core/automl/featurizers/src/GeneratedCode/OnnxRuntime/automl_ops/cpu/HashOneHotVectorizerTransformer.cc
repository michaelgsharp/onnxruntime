// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include "core/automl/featurizers/src/FeaturizerPrep/Featurizers/HashOneHotVectorizerFeaturizer.h"

using namespace std;
namespace featurizers = Microsoft::Featurizer;

namespace onnxruntime {
namespace automl {

template <typename InputT>
class HashOneHotVectorizerTransformer final : public OpKernel {
public:
  explicit HashOneHotVectorizerTransformer(const OpKernelInfo &info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext *ctx) const override {
    // Create the transformer
    featurizers::HashOneHotVectorizerTransformer<InputT> transformer(
      [ctx](void) {
        auto state_tensor(ctx->Input<Tensor>(0));
        uint8_t const * const state_data(state_tensor->Data<uint8_t>());

        Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
        return featurizers::HashOneHotVectorizerTransformer<InputT>(archive);
      }()
    );

    // Get the input
    auto input_tensor(ctx->Input<Tensor>(1));
    InputT * const input_data(input_tensor->Data<InputT>());

    // Prepare the output
    Tensor * ColIndex_tensor(ctr->Output(0, input_tensor->Shape()));
    Tensor * NumCols_tensor(ctr->Output(1, input_tensor->Shape()));
    Tensor * Val_tensor(ctr->Output(2, input_tensor->Shape()));

    uint32_t * ColIndex_data(ColIndex_tensor->MutableData<uint32_t>());
    uint32_t * NumCols_data(NumCols_tensor->MutableData<uint32_t>());
    bool * Val_data(Val_tensor->MutableData<bool>());

    // Execute
    int64_t const length(input_tensor->Shape().GetDims()[0]);

    for(int64_t i = 0; i < length; ++i) {
      auto result(transformer.execute(input_data[i]));

      ColIndex_data[i] = std::move(result.ColIndex);
      NumCols_data[i] = std::move(result.NumCols);
      Val_data[i] = std::move(result.Val);
    }

    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    HashOneHotVectorizerTransformer,
    kMSAutoMLDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("OutputT0", DataTypeImpl::GetTensorType<uint32_t>())
        .TypeConstraint("OutputT1", DataTypeImpl::GetTensorType<bool>())
);

} // namespace automl
} // namespace onnxruntime
