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
  auto state_tensor = ctx->Input<Tensor>(0);
  const uint8_t* state_data = state_tensor->Data<uint8_t>();

  Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
  dtf::DateTimeTransformer transformer(archive);

  auto input_tensor = ctx->Input<Tensor>(1);
  Tensor* year_tensor = ctx->Output(0, input_tensor->Shape());
  Tensor* month_tensor = ctx->Output(1, input_tensor->Shape());
  Tensor* day_tensor = ctx->Output(2, input_tensor->Shape());
  Tensor* hour_tensor = ctx->Output(3, input_tensor->Shape());
  Tensor* minute_tensor = ctx->Output(4, input_tensor->Shape());
  Tensor* second_tensor = ctx->Output(5, input_tensor->Shape());
  Tensor* amPm_tensor = ctx->Output(6, input_tensor->Shape());
  Tensor* hour12_tensor = ctx->Output(7, input_tensor->Shape());
  Tensor* dayOfWeek_tensor = ctx->Output(8, input_tensor->Shape());
  Tensor* dayOfQuarter_tensor = ctx->Output(9, input_tensor->Shape());
  Tensor* dayOfYear_tensor = ctx->Output(10, input_tensor->Shape());
  Tensor* weekOfMonth_tensor = ctx->Output(11, input_tensor->Shape());
  Tensor* quarterOfYear_tensor = ctx->Output(12, input_tensor->Shape());
  Tensor* halfOfYear_tensor = ctx->Output(13, input_tensor->Shape());
  Tensor* weekIso_tensor = ctx->Output(14, input_tensor->Shape());
  Tensor* yearIso_tensor = ctx->Output(15, input_tensor->Shape());
  Tensor* monthLabel_tensor = ctx->Output(16, input_tensor->Shape());
  Tensor* amPmLabel_tensor = ctx->Output(17, input_tensor->Shape());
  Tensor* dayOfWeekLabel_tensor = ctx->Output(18, input_tensor->Shape());
  Tensor* holidayName_tensor = ctx->Output(19, input_tensor->Shape());
  Tensor* isPaidTimeOff_tensor = ctx->Output(20, input_tensor->Shape());

  const int64_t* tp = input_tensor->Data<int64_t>();

  int32_t* year_data = year_tensor->MutableData<int32_t>();
  uint8_t* month_data = month_tensor->MutableData<uint8_t>();
  uint8_t* day_data = day_tensor->MutableData<uint8_t>();
  uint8_t* hour_data = hour_tensor->MutableData<uint8_t>();
  uint8_t* minute_data = minute_tensor->MutableData<uint8_t>();
  uint8_t* second_data = second_tensor->MutableData<uint8_t>();
  uint8_t* amPm_data = amPm_tensor->MutableData<uint8_t>();
  uint8_t* hour12_data = hour12_tensor->MutableData<uint8_t>();
  uint8_t* dayOfWeek_data = dayOfWeek_tensor->MutableData<uint8_t>();
  uint8_t* dayOfQuarter_data = dayOfQuarter_tensor->MutableData<uint8_t>();
  uint16_t* dayOfYear_data = dayOfYear_tensor->MutableData<uint16_t>();
  uint16_t* weekOfMonth_data = weekOfMonth_tensor->MutableData<uint16_t>();
  uint8_t* quarterOfYear_data = quarterOfYear_tensor->MutableData<uint8_t>();
  uint8_t* halfOfYear_data = halfOfYear_tensor->MutableData<uint8_t>();
  uint8_t* weekIso_data = weekIso_tensor->MutableData<uint8_t>();
  int32_t* yearIso_data = yearIso_tensor->MutableData<int32_t>();
  std::string* monthLabel_data = monthLabel_tensor->MutableData<std::string>();
  std::string* amPmLabel_data = amPmLabel_tensor->MutableData<std::string>();
  std::string* dayOfWeekLabel_data = dayOfWeekLabel_tensor->MutableData<std::string>();
  std::string* holidayName_data = holidayName_tensor->MutableData<std::string>();
  uint8_t* isPaidTimeOff_data = isPaidTimeOff_tensor->MutableData<uint8_t>();

  const int64_t length = input_tensor->Shape().GetDims()[0];

  for (int i = 0; i < length; i++) {
    auto time_point = transformer.execute(tp[i]);

    year_data[i] = time_point.year;
    month_data[i] = time_point.month;
    day_data[i] = time_point.day;
    hour_data[i] = time_point.hour;
    minute_data[i] = time_point.minute;
    second_data[i] = time_point.second;
    amPm_data[i] = time_point.amPm;
    hour12_data[i] = time_point.hour12;
    dayOfWeek_data[i] = time_point.dayOfWeek;
    dayOfQuarter_data[i] = time_point.dayOfQuarter;
    dayOfYear_data[i] = time_point.dayOfYear;
    weekOfMonth_data[i] = time_point.weekOfMonth;
    quarterOfYear_data[i] = time_point.quarterOfYear;
    halfOfYear_data[i] = time_point.halfOfYear;
    weekIso_data[i] = time_point.weekIso;
    yearIso_data[i] = time_point.yearIso;
    monthLabel_data[i] = time_point.monthLabel;
    amPmLabel_data[i] = time_point.amPmLabel;
    dayOfWeekLabel_data[i] = time_point.dayOfWeekLabel;
    holidayName_data[i] = time_point.holidayName;
    isPaidTimeOff_data[i] = time_point.isPaidTimeOff;
  }

  return Status::OK();
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
