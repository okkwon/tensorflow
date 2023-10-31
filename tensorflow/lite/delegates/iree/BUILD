load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/tools/evaluation/tasks:build_def.bzl", "task_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "iree_runtime_call",
    srcs = ["iree_runtime_call.c"],
    hdrs = ["iree_runtime_call.h"],
    deps = [
        "@iree_core//runtime/src/iree/runtime",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core/c:common",
    ],
)

cc_library(
    name = "iree_delegate",
    srcs = [
        "iree_delegate.cc",
    ],
    hdrs = [
        "iree_delegate.h",
    ],
    deps = [
        ":iree_runtime_call",
        "@com_google_absl//absl/log",
        "@iree_core//runtime/src/iree/runtime:impl",
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",        
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates/utils:simple_delegate",
    ],
)

################################ Tester classes ################################

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        "//tensorflow/lite/kernels/internal:cppmath",
        "//tensorflow/lite/kernels/internal:types",
    ],
)

cc_library(
    name = "binary_elementwise_tester",
    testonly = 1,
    srcs = ["binary_elementwise_tester.cc"],
    hdrs = ["binary_elementwise_tester.h"],
    deps = [
        ":test_util",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_conversion_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "@FP16",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

############################## Integration tests ###############################

cc_library(
    name = "test_main",
    testonly = 1,
    deps = [
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "add_test",
    srcs = ["add_test.cc"],
    deps = [
        ":binary_elementwise_tester",
        ":iree_delegate",
        ":test_main",
#        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)


#### The following are for using the iree test delegate in TFLite tooling ####
# cc_library(
#     name = "iree_delegate_provider",
#     srcs = ["iree_delegate_provider.cc"],
#     copts = tflite_copts(),
#     deps = [
#         ":iree_delegate",
#         "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
#     ],
#     alwayslink = 1,
# )
# 
# cc_binary(
#     name = "benchmark_model_plus_iree_delegate",
#     copts = tflite_copts(),
#     linkopts = task_linkopts(),
#     deps = [
#         ":iree_delegate_provider",
#         "//tensorflow/lite/tools/benchmark:benchmark_model_main",
#     ],
# )
# 
# cc_binary(
#     name = "inference_diff_plus_iree_delegate",
#     copts = tflite_copts(),
#     linkopts = task_linkopts(),
#     deps = [
#         ":iree_delegate_provider",
#         "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
#         "//tensorflow/lite/tools/evaluation/tasks/inference_diff:run_eval_lib",
#     ],
# )
# 
# cc_binary(
#     name = "imagenet_classification_eval_plus_iree_delegate",
#     copts = tflite_copts(),
#     linkopts = task_linkopts(),
#     deps = [
#         ":iree_delegate_provider",
#         "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
#         "//tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification:run_eval_lib",
#     ],
# )
# 
# cc_binary(
#     name = "coco_object_detection_eval_plus_iree_delegate",
#     copts = tflite_copts(),
#     linkopts = task_linkopts(),
#     deps = [
#         ":iree_delegate_provider",
#         "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
#         "//tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval_lib",
#     ],
# )
# 
# sh_test(
#     name = "external_delegate_test",
#     srcs = ["external_delegate_test.sh"],
#     data = [
#         "//tensorflow/lite/delegates/iree:iree_external_delegate.so",
#         "//tensorflow/lite/tools/benchmark:benchmark_model",
#         "@tflite_mobilenet_float//:mobilenet_v1_1.0_224.tflite",
#     ],
#     visibility = ["//visibility:private"],
# )
