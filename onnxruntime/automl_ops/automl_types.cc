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




// This has to be under onnxruntime to properly specialize a function template

namespace automl {

// No types to register, but keeping this here so if we need to register them the we can more easily do so.
#define REGISTER_CUSTOM_PROTO(TYPE, reg_fn)            \
  {                                                    \
    MLDataType mltype = DataTypeImpl::GetType<TYPE>(); \
    reg_fn(mltype);                                    \
  }

// No types to register, but keeping this here so if we need to register them the we can more easily do so.
void RegisterAutoMLTypes(const std::function<void(MLDataType)>& /*reg_fn*/) {}
#undef REGISTER_CUSTOM_PROTO
} // namespace automl
} // namespace onnxruntime
