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

void RegisterCatImputerFeaturizer(void) {
  static const char* doc = R"DOC(
        Imputes (populates) values with the mode (most common value) encountered during
        training. This featurizer supports float and double for most (if not all) frameworks
        due to the existance of NaN in those types. Other types require 'optional' support
        within the host frameworks and programming languages.

        C++-style pseudo signature:
          std::float_t execute(std::float_t const &value);
          std::double_t execute(std::double_t const &value);
          template <typename T> T execute(std::optional<T> const &value);

        Examples (where 55.5 is the mode value):
          execute(1.0) -> 1.0
          execute(NaN) -> 55.5
          execute(2.0) -> 2.0
    )DOC";

  MS_AUTOML_OPERATOR_SCHEMA(CatImputerTransformer)
      .SinceVersion(1)
      .SetDomain(kMSAutoMLDomain)
      .SetDoc(doc)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "tensor(uint8)")
      .Input(
          1,
          "Input",
          "No information is available",
          "T")
      .Output(
          0,
          "Output",
          "No information is available",
          "T")
      .TypeConstraint(
          "T",
          {"tensor(float)", "tensor(double)", "tensor(string)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromInputToOutput(ctx, 1, 0);
            if (!hasNInputShapes(ctx, 1)) {
              return;
            }
            propagateShapeFromInputToOutput(ctx, 1, 0);
          });
}

void RegisterStringFeaturizer(void) {
  static const char* doc = R"DOC(
        Converts the input into a string representation based on the input's type.

        C++-style pseudo signature:
          template <typename T> std::string execute(T const &value);

        Examples:
          execute(1) -> "1"
          execute(3.14) -> "3.14"
    )DOC";

  MS_AUTOML_OPERATOR_SCHEMA(StringTransformer)
      .SinceVersion(1)
      .SetDomain(kMSAutoMLDomain)
      .SetDoc(doc)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "tensor(uint8)")
      .Input(
          1,
          "Input",
          "No information is available",
          "InputT")
      .Output(
          0,
          "Output",
          "No information is available",
          "tensor(string)")
      .TypeConstraint(
          "InputT",
          {"tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)", "tensor(float)", "tensor(double)", "tensor(bool)", "tensor(string)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
            propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_STRING, 0);

            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(1)->tensor_type().shape();
          });
}

void RegisterDateTimeFeaturizer(void) {
  static const char* doc = R"DOC(
        Extracts various datetime-related values from a UTC time_point.

        C++-style pseudo signature:
          TimePoint execute(std::chron::system_clock::time_point const &value);

        Examples:
          Given a time_point 'value' representing "November 17, 1976 12:27:04PM":

          "November 17, 1976 12:27:04PM" => {
            "year": 1976,
            "month": 11,
            "day": 17,
            "hour": 12,
            "minute": 27,
            "second": 04,
            "amPm": 2,        // PM
            "hour12": 12,
            "dayOfWeek": 3,   // Wednesday
            "dayOfQuarter": 48,
            "dayOfYear": 321,
            "weekOfMonth": 2,
            "quarterOfYear": 4,
            "halfOfYear": 2,
            "weekIso": 47,
            "yearIso": 1976,
            "monthLabel": "November",
            "amPmLabel": "pm",
            "dayOfWeekLabel": "Wednesday",
            "holidayName": "",
            "isPaidTimeOff": 0
          }
    )DOC";

  MS_AUTOML_OPERATOR_SCHEMA(DateTimeTransformer)
      .SinceVersion(1)
      .SetDomain(kMSAutoMLDomain)
      .SetDoc(doc)
      .Input(
          0,
          "State",
          "State generated during training that is used for prediction",
          "tensor(uint8)")
      .Input(
          1,
          "Input",
          "No information is available",
          "tensor(int64)")
      .Output(0, "year", "No information available", "OutputT0")
      .Output(1, "month", "No information available", "OutputT1")
      .Output(2, "day", "No information available", "OutputT1")
      .Output(3, "hour", "No information available", "OutputT1")
      .Output(4, "minute", "No information available", "OutputT1")
      .Output(5, "second", "No information available", "OutputT1")
      .Output(6, "amPm", "No information available", "OutputT1")
      .Output(7, "hour12", "No information available", "OutputT1")
      .Output(8, "dayOfWeek", "No information available", "OutputT1")
      .Output(9, "dayOfQuarter", "No information available", "OutputT1")
      .Output(10, "dayOfYear", "No information available", "OutputT2")
      .Output(11, "weekOfMonth", "No information available", "OutputT2")
      .Output(12, "quarterOfYear", "No information available", "OutputT1")
      .Output(13, "halfOfYear", "No information available", "OutputT1")
      .Output(14, "weekIso", "No information available", "OutputT1")
      .Output(15, "yearIso", "No information available", "OutputT0")
      .Output(16, "monthLabel", "No information available", "OutputT3")
      .Output(17, "amPmLabel", "No information available", "OutputT3")
      .Output(18, "dayOfWeekLabel", "No information available", "OutputT3")
      .Output(19, "holidayName", "No information available", "OutputT3")
      .Output(20, "isPaidTimeOff", "No information available", "OutputT1")
      .TypeConstraint(
          "OutputT0",
          {"tensor(int32_t)"},
          "No information is available")
      .TypeConstraint(
          "OutputT1",
          {"tensor(uint8_t)"},
          "No information is available")
      .TypeConstraint(
          "OutputT2",
          {"tensor(uint16_t)"},
          "No information is available")
      .TypeConstraint(
          "OutputT3",
          {"tensor(string)"},
          "No information is available")
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) {
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
            ctx.getOutputType(11)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT16);
            ctx.getOutputType(12)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(13)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(14)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            ctx.getOutputType(15)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
            ctx.getOutputType(16)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
            ctx.getOutputType(17)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
            ctx.getOutputType(18)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
            ctx.getOutputType(19)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
            ctx.getOutputType(20)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);

            for (int i = 0; i < ctx.getNumOutputs(); ++i) {
              *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(1)->tensor_type().shape();
            }
          });
}

void RegisterAutoMLSchemas() {
  RegisterCatImputerFeaturizer();
  RegisterStringFeaturizer();
  //RegisterDateTimeFeaturizer();

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
      .Input(0, "State",
             "State generated for the Catagory Imputer during training.",
             "tensor(uint8)")
      .Input(1, "X",
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

  static const char* MaxAbsScaler_ver1_doc = R"DOC(
    Scales numbers <TODO: fill in more info later>
    Microsoft::MaxAbsScalerFeaturizer which is a part of a shared library.
  )DOC";

  MS_AUTOML_OPERATOR_SCHEMA(MaxAbsScalarTransformer)
      .SinceVersion(1)
      .SetDomain(kMSAutoMLDomain)
      .SetDoc(MaxAbsScaler_ver1_doc)
      .Input(0, "State",
             "State generated for the MaxAbsScaler during training.",
             "tensor(uint8)")
      .Input(1, "X",
             "The input tensor that needs missing values filled. Can be float or double.",
             "InputT")
      .Output(0, "ScaledValues", "Input tensor with missing values replaced", "T1")
      .TypeConstraint(
          "InputT",
          {"tensor(int8)",
           "tensor(int16)",
           "tensor(int32)",
           "tensor(int64)",
           "tensor(uint8)",
           "tensor(uint16)",
           "tensor(uint32)",
           "tensor(uint64)",
           "tensor(float)",
           "tensor(double)"},
          "Constrain input type to a float or double or string tensor.")
      .TypeConstraint(
          "T1",
          {"tensor(float)",
           "tensor(double)"},
          "Constrain input type to a float or double or string tensor.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        auto input_elem_type = ctx.getInputType(1)->tensor_type().elem_type();
        if (input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_INT8 ||
            input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_INT16 ||
            input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
            input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_UINT16 ||
            input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        } else if (input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_INT32 ||
            input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_INT64 ||
            input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_UINT32 ||
            input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_UINT64 ||
            input_elem_type == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE);
        }

        if (!hasNInputShapes(ctx, 1)) {
          return;
        }
        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() =
            ctx.getInputType(1)->tensor_type().shape();
      });

}
}  // namespace automl
}  // namespace onnxruntime
