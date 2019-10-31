// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include "core/automl/featurizers/src/FeaturizerPrep/Featurizers/DateTimeFeaturizer.h"

namespace dtf = Microsoft::Featurizer::Featurizers;

namespace onnxruntime {
namespace automl {

class DateTimeTransformer final : public OpKernel {
 public:
  explicit DateTimeTransformer(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override;
};

Status DateTimeTransformer::Compute(OpKernelContext* ctx) const {
  Status s;
  auto input_tensor = ctx->Input<Tensor>(0);
  auto* year_tensor = ctx->Output<Tensor>(0);
  auto* month_tensor = ctx->Output<Tensor>(1);
  auto* day_tensor = ctx->Output<Tensor>(2);
  auto* hour_tensor = ctx->Output<Tensor>(3);
  auto* minute_tensor = ctx->Output<Tensor>(4);
  auto* second_tensor = ctx->Output<Tensor>(5);
  auto* amPm_tensor = ctx->Output<Tensor>(6);
  auto* hour12_tensor = ctx->Output<Tensor>(7);
  auto* dayOfWeek_tensor = ctx->Output<Tensor>(8);
  auto* dayOfQuarter_tensor = ctx->Output<Tensor>(9);
  auto* dayOfYear_tensor = ctx->Output<Tensor>(10);
  auto* weekOfMonth_tensor = ctx->Output<Tensor>(11);
  auto* quarterOfYear_tensor = ctx->Output<Tensor>(12);
  auto* halfOfYear_tensor = ctx->Output<Tensor>(13);
  auto* weekIso_tensor = ctx->Output<Tensor>(14);
  auto* yearIso_tensor = ctx->Output<Tensor>(15);
  auto* monthLabel_tensor = ctx->Output<Tensor>(16);
  auto* amPmLabel_tensor = ctx->Output<Tensor>(17);
  auto* dayOfWeekLabel_tensor = ctx->Output<Tensor>(18);
  auto* holidayName_tensor = ctx->Output<Tensor>(19);
  auto* isPaidTimeOff_tensor = ctx->Output<Tensor>(20);

  int64_t tp = *input_tensor->Data<int64_t>();

  dtf::DateTimeTransformer transformer("", "");
  auto time_point = transformer.execute(tp);

  return s;
}

ONNX_OPERATOR_KERNEL_EX(
    DateTimeTransformer,
    kMSAutoMLDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<uint16_t>())
        .TypeConstraint("T5", DataTypeImpl::GetTensorType<std::string>()),
    DateTimeTransformer);
}  // namespace automl
}  // namespace onnxruntime
