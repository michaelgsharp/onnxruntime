// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"
#include "core/session/automl_data_containers.h"

#include "automl_ops/automl_types.h"
#include "automl_ops/automl_featurizers.h"

namespace dtf = Microsoft::Featurizer::Featurizers;

namespace onnxruntime {

// This temporary to register custom types so ORT is aware of it
// although it still can not serialize such a type.
// These character arrays must be extern so the resulting instantiated template
// is globally unique

extern const char kMsAutoMLDomain[] = "com.microsoft.automl";
extern const char kTimepointName[] = "DateTimeFeaturizer_TimePoint";

// Specialize for our type so we can convert to external struct
// 
template <>
struct NonTensorTypeConverter<dtf::TimePoint> {
  static void FromContainer(MLDataType dtype, const void* data, size_t data_size, OrtValue& output) {
    ORT_ENFORCE(sizeof(DateTimeFeaturizerTimePointData) == data_size, "Expecting an instance of ExternalTimePoint");
    const DateTimeFeaturizerTimePointData* dc = reinterpret_cast<const DateTimeFeaturizerTimePointData*>(data);
    std::unique_ptr<dtf::TimePoint> tp(new dtf::TimePoint());
    tp->year = dc->year;
    tp->month = dc->month;
    tp->day = dc->day;
    tp->hour = dc->hour;
    tp->minute = dc->minute;
    tp->second = dc->second;
    tp->amPm = dc->amPm;
    tp->hour12 = dc->hour12;
    tp->dayOfWeek = dc->dayOfWeek;
    tp->dayOfQuarter = dc->dayOfQuarter;
    tp->dayOfYear = dc->dayOfYear;
    tp->weekOfMonth = dc->weekOfMonth;
    tp->quarterOfYear = dc->quarterOfYear;
    tp->halfOfYear = dc->halfOfYear;
    tp->weekIso = dc->weekIso;
    tp->yearIso = dc->yearIso;
    tp->monthLabel = dc->monthLabel;
    tp->amPmLabel = dc->amPmLabel;
    tp->dayOfWeekLabel = dc->dayOfWeekLabel;
    tp->holidayName = dc->holidayName;
    tp->isPaidTimeOff = dc->isPaidTimeOff;
    output.Init(tp.get(),
                dtype,
                dtype->GetDeleteFunc());
    tp.release();
  }
  static void ToContainer(const OrtValue& input, size_t data_size, void* data) {
    ORT_ENFORCE(sizeof(DateTimeFeaturizerTimePointData) == data_size, "Expecting an instance of ExternalTimePoint");
    DateTimeFeaturizerTimePointData* dc = reinterpret_cast<DateTimeFeaturizerTimePointData*>(data);
    const dtf::TimePoint& tp = input.Get<dtf::TimePoint>();
    dc->year = tp.year;
    dc->month = tp.month;
    dc->day = tp.day;
    dc->hour = tp.hour;
    dc->minute = tp.minute;
    dc->second = tp.second;
    dc->amPm = tp.amPm;
    dc->hour12 = tp.hour12;
    dc->dayOfWeek = tp.dayOfWeek;
    dc->dayOfQuarter = tp.dayOfQuarter;
    dc->dayOfYear = tp.dayOfYear;
    dc->weekOfMonth = tp.weekOfMonth;
    dc->quarterOfYear = tp.quarterOfYear;
    dc->halfOfYear = tp.halfOfYear;
    dc->weekIso = tp.weekIso;
    dc->yearIso = tp.yearIso;
    dc->monthLabel = tp.monthLabel;
    dc->amPmLabel = tp.amPmLabel;
    dc->dayOfWeekLabel = tp.dayOfWeekLabel;
    dc->holidayName = tp.holidayName;
    dc->isPaidTimeOff = tp.isPaidTimeOff;
  }
};


// This has to be under onnxruntime to properly specialize a function template
ORT_REGISTER_OPAQUE_TYPE(dtf::TimePoint, kMsAutoMLDomain, kTimepointName);

namespace automl {

#define REGISTER_CUSTOM_PROTO(TYPE, reg_fn)            \
  {                                                    \
    MLDataType mltype = DataTypeImpl::GetType<TYPE>(); \
    reg_fn(mltype);                                    \
  }

void RegisterAutoMLTypes(const std::function<void(MLDataType)>& reg_fn) {
  REGISTER_CUSTOM_PROTO(dtf::TimePoint, reg_fn);
}
#undef REGISTER_CUSTOM_PROTO
} // namespace automl
} // namespace onnxruntime
