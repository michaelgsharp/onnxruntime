// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "core/graph/automl_ops/automl_defs.h"
#include "core/graph/op.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace onnxruntime {
namespace automl {
using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL;

void RegisterAutoMLSchemas() {
  static const char* DateTimeTransformer_ver1_doc = R"DOC(
    DateTimeTransformer accepts a single scalar int64 tensor, constructs
    an instance of std::chrono::system_clock::time_point and passes it as an argument
    to Microsoft::DateTimeFeaturizer which is a part of a shared library.
    It returns an instance of TimePoint class.
  )DOC";

  MS_AUTOML_OPERATOR_SCHEMA(DateTimeTransformer)
      .SinceVersion(1)
      .SetDomain(kMSAutoMLDomain)
      .SetDoc(DateTimeTransformer_ver1_doc)
      .Input(0, "X",
             "The input represents a number of seconds passed since the epoch, suitable to properly construct"
             "an instance of std::chrono::system_clock::time_point",
             "T1")
      .Output(0, "Year", "calendar year, std::int32_t", "T2")
      .Output(1, "Month", "calendar month, 1 through 1, std::uint8_t", "T3")
      .Output(2, "Day", "calendar day of month, 1 through 31, std::uint8_t", "T3")
      .Output(3, "Hour", "hour of day, 0 through 23, std::uint8_t", "T3")
      .Output(4, "Minute", "minute of day, 0 through 59, std::uint8_t", "T3")
      .Output(5, "Second", "second of day, 0 through 59, std::uint8_t", "T3")
      .Output(6, "AmPm", "0 if hour is before noon (12 pm), 1 otherwise, std::uint8_t", "T3")
      .Output(7, "Hour12", "hour of day on a 12 basis, without the AM/PM piece, std::uint8_t", "T3")
      .Output(8, "DayOfWeek", "day of week, 0 (Monday) through 6 (Sunday), std::uint8_t", "T3")
      .Output(9, "DayOfQuarter", "day of quarter, 1 through 92, std::uint8_t", "T3")
      .Output(10, "DayOfYear", "day of year, 1 through 366, std::uint16_t", "T4")
      .Output(11, "WeekOfMonth", "week of the month, 0 - 4, std::uint16_t", "T4")
      .Output(12, "QuarterOfYear", "calendar quarter, 1 through 4, std::uint8_t", "T3")
      .Output(13, "HalfOfYear", "1 if date is prior to July 1, 2 otherwise, std::uint8_t", "T3")
      .Output(14, "WeekIso", "ISO week, see below for details, std::uint8_t", "T3")
      .Output(15, "YearIso", "ISO year, see details later, std::int32_t", "T2")
      .Output(16, "MonthLabel", "calendar month as string, 'January' through 'December', std::string", "T5")
      .Output(17, "AmPmLabel", "'am' if hour is before noon (12 pm), 'pm' otherwise, std::string", "T5")
      .Output(18, "DayOfWeekLabel", "day of week as string, std::string", "T5")
      .Output(19, "HolidayName", "If a country is provided, we check if the date is a holiday, std::string", "T5")
      .Output(20, "IsPaidTimeOff", "If its a holiday, is it PTO, std::uint8_t", "T3")
      .TypeConstraint(
          "T1",
          {"tensor(int64)"},
          "Constrain input type to int64 scalar tensor.")
      .TypeConstraint(
          "T2",
          {"tensor(int32)"},
          "Constrain output type to int32 scalar tensor")
      .TypeConstraint(
          "T3",
          {"tensor(uint8)"},
          "Constrain output type to uint8 scalar tensor")
      .TypeConstraint(
          "T4",
          {"tensor(uint16)"},
          "Constrain output type to uint16")
      .TypeConstraint(
          "T5",
          {"tensor(string)"},
          "Constrain output type to string")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
            ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
            ctx.getOutputType(1)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(2)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(3)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(4)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(5)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(6)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(7)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(8)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(9)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(10)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT16);
            ctx.getOutputType(12)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT16);
            ctx.getOutputType(12)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(13)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(14)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(15)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT32);
            ctx.getOutputType(16)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
            ctx.getOutputType(17)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
            ctx.getOutputType(18)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
            ctx.getOutputType(19)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
            ctx.getOutputType(20)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);

            for (int i = 0; i < ctx.getNumOutputs(); i++) {
              *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() =
                  ctx.getInputType(0)->tensor_type().shape();
            }
      });

  static const char* StringTransformer_ver1_doc = R"DOC(
    StringTransformer accepts a single scalar tensor and converts it
    to its appropriate string representation by passing it to
    Microsoft::DateTimeFeaturizer which is a part of a shared library.
  )DOC";

  MS_AUTOML_OPERATOR_SCHEMA(StringTransformer)
      .SinceVersion(1)
      .SetDomain(kMSAutoMLDomain)
      .SetDoc(StringTransformer_ver1_doc)
      .Input(0, "X",
             "The input tensor to convert to a string. Can be any number type, bool, or a string",
             "T1")
      .Output(0, "AsString", "input converted to a string, std::string", "tensor(string)")
      .TypeConstraint(
          "T1",
          {"tensor(int8)",
           "tensor(int16)",
           "tensor(int32)",
           "tensor(int64)",
           "tensor(uint8)",
           "tensor(uint16)",
           "tensor(uint32)",
           "tensor(uint64)",
           "tensor(float)",
           "tensor(double)",
           "tensor(bool)",
           "tensor(string)"},
          "Constrain input type to number, bool, or string tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);

        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() =
            ctx.getInputType(0)->tensor_type().shape();
      });


  MS_AUTOML_OPERATOR_SCHEMA(SampleAdd)
      .SinceVersion(1)
      .SetDomain(kMSAutoMLDomain)
      .SetDoc(DateTimeTransformer_ver1_doc)
      .Input(0, "X",
             "The input represents a number of seconds passed since the epoch, suitable to properly construct"
             "an instance of std::chrono::system_clock::time_point",
             "T1")
      .Output(0, "Y", "The output which is a Microsoft::DateTimeFeaturizer::TimePoint structure", "T2")
      .TypeConstraint(
          "T1",
          {"tensor(int64)"},
          "Constrain input type to int64 scalar tensor.")
      .TypeConstraint(
          "T2",
          {"opaque(com.microsoft.automl,DateTimeFeaturizer_TimePoint)"},
          "Constrain output type to an AutoML specific Microsoft::Featurizers::TimePoint type"
          "currently not part of ONNX standard. When it becomes a part of the standard we will adjust this"
          "kernel definition and move it to ONNX repo");
}
}  // namespace automl
}  // namespace onnxruntime
