// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "core/graph/automl_ops/automl_defs.h"
#include "core/graph/op.h"

#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

#define MS_AUTOML_OPERATOR_SCHEMA(name)                         MS_AUTOML_OPERATOR_SCHEMA_UNIQ_HELPER(__COUNTER__, name)
#define MS_AUTOML_OPERATOR_SCHEMA_UNIQ_HELPER(Counter, name)    MS_AUTOML_OPERATOR_SCHEMA_UNIQ(Counter, name)

#define MS_AUTOML_OPERATOR_SCHEMA_UNIQ(Counter, name)               \
  static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce(    \
      op_schema_register_once##name##Counter) ONNX_UNUSED =         \
      ONNX_NAMESPACE::OpSchema(#name, __FILE__, __LINE__)

#define MS_AUTOML_OPERATOR_SCHEMA_ELSEWHERE(name, schema_func)                          MS_AUTOML_OPERATOR_SCHEMA_UNIQ_HELPER_ELSEWHERE(__COUNTER__, name, schema_func)
#define MS_AUTOML_OPERATOR_SCHEMA_UNIQ_HELPER_ELSEWHERE(Counter, name, schema_func)     MS_AUTOML_OPERATOR_SCHEMA_UNIQ_ELSEWHERE(Counter, name, schema_func)

#define MS_AUTOML_OPERATOR_SCHEMA_UNIQ_ELSEWHERE(Counter, name, schema_func)    \
  static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce(                \
      op_schema_register_once##name##Counter) ONNX_UNUSED =                     \
      schema_func(ONNX_NAMESPACE::OpSchema(#name, __FILE__, __LINE__))

namespace onnxruntime {
namespace automl {

using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL;

// Forward declarations
static RegisterCatImputerFeaturizer(void);
static RegisterDateTimeFeaturizer(void);
static RegisterHashOneHotVectorizerFeaturizer(void);
static RegisterImputationMarkerFeaturizer(void);
static RegisterLabelEncoderFeaturizer(void);
static RegisterMaxAbsScalarFeaturizer(void);
static RegisterMinMaxScalarFeaturizer(void);
static RegisterMissingDummiesFeaturizer(void);
static RegisterOneHotEncoderFeaturizer(void);
static RegisterRobustScalarFeaturizer(void);
static RegisterStringFeaturizer(void);

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
void RegisterAutoMLSchemas() {
    RegisterCatImputerFeaturizer();
    RegisterDateTimeFeaturizer();
    RegisterHashOneHotVectorizerFeaturizer();
    RegisterImputationMarkerFeaturizer();
    RegisterLabelEncoderFeaturizer();
    RegisterMaxAbsScalarFeaturizer();
    RegisterMinMaxScalarFeaturizer();
    RegisterMissingDummiesFeaturizer();
    RegisterOneHotEncoderFeaturizer();
    RegisterRobustScalarFeaturizer();
    RegisterStringFeaturizer();
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
void RegisterCatImputerFeaturizer(void) {
    static const char * doc = R"DOC(
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
            "tensor(uint8)"
        )
        .Input(
            1,
            "Input",
            "No information is available",
            "T"
        )
        .Output(
            0,
            "Output",
            "No information is available",
            "T"
        )
        .TypeConstraint(
            "T",
            {"tensor(float_t)", "tensor(double_t)", "tensor(string)"},
            "No information is available"
        )
        .TypeAndShapeInferenceFunction(
            [](ONNX_NAMESPACE::InferenceContext& ctx) {
                propagateElemTypeFromInputToOutput(ctx, 1, 0);
                if (!hasNInputShapes(ctx, 1)) {
                    return;
                }
                propagateShapeFromInputToOutput(ctx, 1, 0);
            }
        )
    ;
}

void RegisterDateTimeFeaturizer(void) {
    static const char * doc = R"DOC(
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
            "tensor(uint8)"
        )
        .Input(
            1,
            "Input",
            "No information is available",
            "tensor(std::int64_t)"
        )
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
            {"tensor(tensor(int32_t))"},
            "No information is available"
        )
        .TypeConstraint(
            "OutputT1",
            {"tensor(tensor(uint8_t))"},
            "No information is available"
        )
        .TypeConstraint(
            "OutputT2",
            {"tensor(tensor(uint16_t))"},
            "No information is available"
        )
        .TypeConstraint(
            "OutputT3",
            {"tensor(tensor(string))"},
            "No information is available"
        )
        .TypeAndShapeInferenceFunction(
            [](ONNX_NAMESPACE::InferenceContext &ctx) {
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

                for(int i = 0; i < ctx.getNumOutputs(); ++i) {
                    *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(1)->tensor_type().shape();
                }
            }
        )
    ;
}

void RegisterHashOneHotVectorizerFeaturizer(void) {
    static const char * doc = R"DOC(
        Hashes the input to a categorical value, then produces a one hot encoded vector
        based on that value.

        C++-style pseudo signature:
            template <typename T> HashOneHotVectorizerStruct execute(T const &value);

        Examples:
          Assuming the hashing algorithm...
            "A" -> 1
            "B" -> 2
            "C" -> 5

          and 'numCols' set to 8:

            execute("A") -> [1, 0, 0, 0, 0, 0, 0, 0]
            execute("B") -> [0, 1, 0, 0, 0, 0, 0, 0]
            execute("C") -> [0, 0, 0, 0, 1, 0, 0, 0]
    )DOC";

    MS_AUTOML_OPERATOR_SCHEMA(HashOneHotVectorizerTransformer)
        .SinceVersion(1)
        .SetDomain(kMSAutoMLDomain)
        .SetDoc(doc)
        .Input(
            0,
            "State",
            "State generated during training that is used for prediction",
            "tensor(uint8)"
        )
        .Input(
            1,
            "Input",
            "No information is available",
            "InputT"
        )
        .Output(0, "ColIndex", "No information available", "OutputT0")
        .Output(1, "NumCols", "No information available", "OutputT0")
        .Output(2, "Val", "No information available", "OutputT1")
        .TypeConstraint(
            "InputT",
            {"tensor(tensor(int8_t))", "tensor(tensor(int16_t))", "tensor(tensor(int32_t))", "tensor(tensor(int64_t))", "tensor(tensor(uint8_t))", "tensor(tensor(uint16_t))", "tensor(tensor(uint32_t))", "tensor(tensor(uint64_t))", "tensor(tensor(float_t))", "tensor(tensor(double_t))", "tensor(tensor(bool))", "tensor(tensor(string))"},
            "No information is available"
        )
        .TypeConstraint(
            "OutputT0",
            {"tensor(tensor(uint32_t))"},
            "No information is available"
        )
        .TypeConstraint(
            "OutputT1",
            {"tensor(tensor(bool))"},
            "No information is available"
        )
        .TypeAndShapeInferenceFunction(
            [](ONNX_NAMESPACE::InferenceContext &ctx) {
                ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT32);
                ctx.getOutputType(1)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT32);
                ctx.getOutputType(2)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);

                for(int i = 0; i < ctx.getNumOutputs(); ++i) {
                    *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(1)->tensor_type().shape();
                }
            }
        )
    ;
}

void RegisterImputationMarkerFeaturizer(void) {
    static const char * doc = R"DOC(
        Returns true if the input is null, false if it is not.

        C++-style pseudo signature:
          bool execute(std::float_t const &value);
          bool execute(std::double_t const &value);
          template <typename T> bool execute(std::optional<T> const &value);

        Examples:
          3.0 -> false
          NaN -> true
          "foo" -> false
          std::optional<std::string>() -> true
          std::optional<std::string>("bar") -> false
    )DOC";

    MS_AUTOML_OPERATOR_SCHEMA(ImputationMarkerTransformer)
        .SinceVersion(1)
        .SetDomain(kMSAutoMLDomain)
        .SetDoc(doc)
        .Input(
            0,
            "State",
            "State generated during training that is used for prediction",
            "tensor(uint8)"
        )
        .Input(
            1,
            "Input",
            "No information is available",
            "InputT"
        )
        .Output(
            0,
            "Output",
            "No information is available",
            "tensor(bool)"
        )
        .TypeConstraint(
            "InputT",
            {"tensor(tensor(float_t))", "tensor(tensor(double_t))", "tensor(tensor(string))"},
            "No information is available"
        )
        .TypeAndShapeInferenceFunction(
            [](ONNX_NAMESPACE::InferenceContext& ctx) {
                propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_BOOL, 0);

                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(1)->tensor_type().shape();
            }
        )
    ;
}

void RegisterLabelEncoderFeaturizer(void) {
    static const char * doc = R"DOC(
        Returns a unique id for the input based on all values encountered during training.

        C++-style pseudo signature:
          template <typename T> std::uint32_t execute(T const &value);

        Examples:
          Assuming the training data of ["A", "B", "C"]...

          execute("A") -> 1
          execute("B") -> 2
          execute("C") -> 3
          execute("This value was not seen during training") -> 0
    )DOC";

    MS_AUTOML_OPERATOR_SCHEMA(LabelEncoderTransformer)
        .SinceVersion(1)
        .SetDomain(kMSAutoMLDomain)
        .SetDoc(doc)
        .Input(
            0,
            "State",
            "State generated during training that is used for prediction",
            "tensor(uint8)"
        )
        .Input(
            1,
            "Input",
            "No information is available",
            "InputT"
        )
        .Output(
            0,
            "Output",
            "No information is available",
            "tensor(uint32_t)"
        )
        .TypeConstraint(
            "InputT",
            {"tensor(tensor(int8_t))", "tensor(tensor(int16_t))", "tensor(tensor(int32_t))", "tensor(tensor(int64_t))", "tensor(tensor(uint8_t))", "tensor(tensor(uint16_t))", "tensor(tensor(uint32_t))", "tensor(tensor(uint64_t))", "tensor(tensor(float_t))", "tensor(tensor(double_t))", "tensor(tensor(bool))", "tensor(tensor(string))"},
            "No information is available"
        )
        .TypeAndShapeInferenceFunction(
            [](ONNX_NAMESPACE::InferenceContext& ctx) {
                propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_UINT32, 0);

                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(1)->tensor_type().shape();
            }
        )
    ;
}

void RegisterMaxAbsScalarFeaturizer(void) {
    static const char * doc = R"DOC(
        Scales input based on the maximum absolute value of all data encountered during training.

        C++-style pseudo signature:
          std::float_t execute(std::uint16_t value);
          std::double_t execute(std::uint32_t value);

        Examples:
          Given a training set of [1.0, -2.0, 3.0, -4.0], where 4.0 is the absolute value of the
          maximum value encountered...

          execute(1.0) -> 1.0 / 4.0
          execute(-4.0) -> -4.0 / 4.0
          execute(100.0) -> 100 / 4.0
    )DOC";

    MS_AUTOML_OPERATOR_SCHEMA(MaxAbsScalarTransformer)
        .SinceVersion(1)
        .SetDomain(kMSAutoMLDomain)
        .SetDoc(doc)
        .Input(
            0,
            "State",
            "State generated during training that is used for prediction",
            "tensor(uint8)"
        )
        .Input(
            1,
            "Input",
            "No information is available",
            "InputT"
        )
        .Output(
            0,
            "Output",
            "No information is available",
            "tensor(float_t)"
        )
        .TypeConstraint(
            "InputT",
            {"tensor(tensor(int8_t))", "tensor(tensor(int16_t))", "tensor(tensor(uint8_t))", "tensor(tensor(uint16_t))", "tensor(tensor(float_t))"},
            "No information is available"
        )
        .TypeAndShapeInferenceFunction(
            [](ONNX_NAMESPACE::InferenceContext& ctx) {
                propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 0);

                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(1)->tensor_type().shape();
            }
        )
    ;

    MS_AUTOML_OPERATOR_SCHEMA(MaxAbsScalarTransformer)
        .SinceVersion(1)
        .SetDomain(kMSAutoMLDomain)
        .SetDoc(doc)
        .Input(
            0,
            "State",
            "State generated during training that is used for prediction",
            "tensor(uint8)"
        )
        .Input(
            1,
            "Input",
            "No information is available",
            "InputT"
        )
        .Output(
            0,
            "Output",
            "No information is available",
            "tensor(double_t)"
        )
        .TypeConstraint(
            "InputT",
            {"tensor(tensor(int32_t))", "tensor(tensor(int64_t))", "tensor(tensor(uint32_t))", "tensor(tensor(uint64_t))", "tensor(tensor(double_t))"},
            "No information is available"
        )
        .TypeAndShapeInferenceFunction(
            [](ONNX_NAMESPACE::InferenceContext& ctx) {
                propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, 0);

                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(1)->tensor_type().shape();
            }
        )
    ;
}

void RegisterMinMaxScalarFeaturizer(void) {
    static const char * doc = R"DOC(
        Scales input based on the scale that results from the minimum and maximum values encountered
        during training.

        C++-style pseudo signature:
            template <typeanem T> std::double_t(T const &value);

        Examples:
          Given the training data [1, 2, 3, 4, 5];
            min: 1
            max: 5
            scale (<max> - <min>): 4

          execute(2) = 2 / 4
          execute(20) = 20 / 4
    )DOC";

    MS_AUTOML_OPERATOR_SCHEMA(MinMaxScalarTransformer)
        .SinceVersion(1)
        .SetDomain(kMSAutoMLDomain)
        .SetDoc(doc)
        .Input(
            0,
            "State",
            "State generated during training that is used for prediction",
            "tensor(uint8)"
        )
        .Input(
            1,
            "Input",
            "No information is available",
            "InputT"
        )
        .Output(
            0,
            "Output",
            "No information is available",
            "tensor(double_t)"
        )
        .TypeConstraint(
            "InputT",
            {"tensor(tensor(int8_t))", "tensor(tensor(int16_t))", "tensor(tensor(int32_t))", "tensor(tensor(int64_t))", "tensor(tensor(uint8_t))", "tensor(tensor(uint16_t))", "tensor(tensor(uint32_t))", "tensor(tensor(uint64_t))", "tensor(tensor(float_t))", "tensor(tensor(double_t))"},
            "No information is available"
        )
        .TypeAndShapeInferenceFunction(
            [](ONNX_NAMESPACE::InferenceContext& ctx) {
                propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, 0);

                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(1)->tensor_type().shape();
            }
        )
    ;
}

void RegisterMissingDummiesFeaturizer(void) {
    static const char * doc = R"DOC(
        Returns 1 if the input is null, 0 if it is not.

        C++-style pseudo signature:
            std::int8_t execute(std::float_t const &value);
            std::int8_t execute(std::double_t const &value);
            template <typename T> std::int8_t execute(T const &value);

        Examples:
          1.0 -> 0
          NaN -> 1
          "foo" -> 0
          std::optional<std::string>() -> 1
          std::optional<std::string>("bar") -> 0
    )DOC";

    MS_AUTOML_OPERATOR_SCHEMA(MissingDummiesTransformer)
        .SinceVersion(1)
        .SetDomain(kMSAutoMLDomain)
        .SetDoc(doc)
        .Input(
            0,
            "State",
            "State generated during training that is used for prediction",
            "tensor(uint8)"
        )
        .Input(
            1,
            "Input",
            "No information is available",
            "InputT"
        )
        .Output(
            0,
            "Output",
            "No information is available",
            "tensor(int8_t)"
        )
        .TypeConstraint(
            "InputT",
            {"tensor(tensor(float_t))", "tensor(tensor(double_t))", "tensor(tensor(string))"},
            "No information is available"
        )
        .TypeAndShapeInferenceFunction(
            [](ONNX_NAMESPACE::InferenceContext& ctx) {
                propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_INT8, 0);

                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(1)->tensor_type().shape();
            }
        )
    ;
}

void RegisterOneHotEncoderFeaturizer(void) {
    static const char * doc = R"DOC(
        Produces a one hot vector based on categories calculated during training.

        C++-style pseudo signature:
          template <typename T> OneHotVector execute(T const &value);

        Examples:
          Assuming the training data [10, 20, 30, 40]...

          execute(10) -> [0, 1, 0, 0, 0]
          execute(20) -> [0, 0, 1, 0, 0]
          execute(30) -> [0, 0, 0, 1, 0]
          execute(40) -> [0, 0, 0, 0, 1]
          execute(200) -> [1, 0, 0, 0, 0]
          execute(-1) -> [1, 0, 0, 0, 0]
    )DOC";

    MS_AUTOML_OPERATOR_SCHEMA(OneHotEncoderTransformer)
        .SinceVersion(1)
        .SetDomain(kMSAutoMLDomain)
        .SetDoc(doc)
        .Input(
            0,
            "State",
            "State generated during training that is used for prediction",
            "tensor(uint8)"
        )
        .Input(
            1,
            "Input",
            "No information is available",
            "InputT"
        )
        .Output(0, "index", "No information available", "OutputT0")
        .Output(1, "size", "No information available", "OutputT0")
        .Output(2, "appearances", "No information available", "OutputT0")
        .TypeConstraint(
            "InputT",
            {"tensor(tensor(int8_t))", "tensor(tensor(int16_t))", "tensor(tensor(int32_t))", "tensor(tensor(int64_t))", "tensor(tensor(uint8_t))", "tensor(tensor(uint16_t))", "tensor(tensor(uint32_t))", "tensor(tensor(uint64_t))", "tensor(tensor(float_t))", "tensor(tensor(double_t))", "tensor(tensor(bool))", "tensor(tensor(string))"},
            "No information is available"
        )
        .TypeConstraint(
            "OutputT0",
            {"tensor(tensor(uint32_t))"},
            "No information is available"
        )
        .TypeAndShapeInferenceFunction(
            [](ONNX_NAMESPACE::InferenceContext &ctx) {
                ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT32);
                ctx.getOutputType(1)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT32);
                ctx.getOutputType(2)->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT32);

                for(int i = 0; i < ctx.getNumOutputs(); ++i) {
                    *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(1)->tensor_type().shape();
                }
            }
        )
    ;
}

void RegisterRobustScalarFeaturizer(void) {
    static const char * doc = R"DOC(
        MinMaxScalarEstimator + centering?

        C++-style pseudo signature:
            TODO

        Examples:
          TODO
    )DOC";

    MS_AUTOML_OPERATOR_SCHEMA(RobustScalarTransformer)
        .SinceVersion(1)
        .SetDomain(kMSAutoMLDomain)
        .SetDoc(doc)
        .Input(
            0,
            "State",
            "State generated during training that is used for prediction",
            "tensor(uint8)"
        )
        .Input(
            1,
            "Input",
            "No information is available",
            "InputT"
        )
        .Output(
            0,
            "Output",
            "No information is available",
            "tensor(float_t)"
        )
        .TypeConstraint(
            "InputT",
            {"tensor(tensor(int8_t))", "tensor(tensor(int16_t))", "tensor(tensor(uint8_t))", "tensor(tensor(uint16_t))", "tensor(tensor(float_t))"},
            "No information is available"
        )
        .TypeAndShapeInferenceFunction(
            [](ONNX_NAMESPACE::InferenceContext& ctx) {
                propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 0);

                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(1)->tensor_type().shape();
            }
        )
    ;

    MS_AUTOML_OPERATOR_SCHEMA(RobustScalarTransformer)
        .SinceVersion(1)
        .SetDomain(kMSAutoMLDomain)
        .SetDoc(doc)
        .Input(
            0,
            "State",
            "State generated during training that is used for prediction",
            "tensor(uint8)"
        )
        .Input(
            1,
            "Input",
            "No information is available",
            "InputT"
        )
        .Output(
            0,
            "Output",
            "No information is available",
            "tensor(double_t)"
        )
        .TypeConstraint(
            "InputT",
            {"tensor(tensor(int32_t))", "tensor(tensor(int64_t))", "tensor(tensor(uint32_t))", "tensor(tensor(uint64_t))", "tensor(tensor(double_t))"},
            "No information is available"
        )
        .TypeAndShapeInferenceFunction(
            [](ONNX_NAMESPACE::InferenceContext& ctx) {
                propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, 0);

                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(1)->tensor_type().shape();
            }
        )
    ;
}

void RegisterStringFeaturizer(void) {
    static const char * doc = R"DOC(
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
            "tensor(uint8)"
        )
        .Input(
            1,
            "Input",
            "No information is available",
            "InputT"
        )
        .Output(
            0,
            "Output",
            "No information is available",
            "tensor(string)"
        )
        .TypeConstraint(
            "InputT",
            {"tensor(tensor(int8_t))", "tensor(tensor(int16_t))", "tensor(tensor(int32_t))", "tensor(tensor(int64_t))", "tensor(tensor(uint8_t))", "tensor(tensor(uint16_t))", "tensor(tensor(uint32_t))", "tensor(tensor(uint64_t))", "tensor(tensor(float_t))", "tensor(tensor(double_t))", "tensor(tensor(bool))", "tensor(tensor(string))"},
            "No information is available"
        )
        .TypeAndShapeInferenceFunction(
            [](ONNX_NAMESPACE::InferenceContext& ctx) {
                propagateElemTypeFromDtypeToOutput(ctx, ONNX_NAMESPACE::TensorProto_DataType_STRING, 0);

                *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = ctx.getInputType(1)->tensor_type().shape();
            }
        )
    ;
}

} // namespace automl
} // namespace onnxruntime
