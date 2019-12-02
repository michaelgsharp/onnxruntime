# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# This source code should not depend on the onnxruntime and may be built independently

file(GLOB automl_featurizers_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/automl/featurizers/src/*.h"
    "${ONNXRUNTIME_ROOT}/core/automl/featurizers/src/Featurizers/*.h"
    "${ONNXRUNTIME_ROOT}/core/automl/featurizers/src/Featurizers/*.cpp"
)

source_group(TREE ${ONNXRUNTIME_ROOT}/core/automl/ FILES ${onnxruntime_automl_featurizers_srcs})

add_library(automl_featurizers ${automl_featurizers_srcs})

target_include_directories(automl_featurizers PRIVATE ${ONNXRUNTIME_ROOT} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

set_target_properties(automl_featurizers PROPERTIES FOLDER "AutoMLFeaturizers")

source_group(TREE ${ONNXRUNTIME_ROOT}/core/automl/ FILES ${automl_featurizers_tests_srcs})


if (WIN32)
    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include.
    set_target_properties(automl_featurizers PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/ConfigureVisualStudioCodeAnalysis.props)
endif()
