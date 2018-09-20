load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
  name = "lsh_pmdp.so",
  srcs = ["lsh_pmdp.cc"],
)

tf_custom_op_library(
  name = "lsh_gpu.so",
  srcs = ["lsh_gpu.cc"],
  gpu_srcs = ["lsh_gpu.cu.cc"],
)

