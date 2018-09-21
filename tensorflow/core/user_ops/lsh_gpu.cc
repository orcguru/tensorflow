/*
 * computeProductModDefaultPrime function implementation based on E2LSH-0.1
 */
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("LshGpu")
  .Input("a: int32")
  .Input("b: int32")
  .Output("h: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

void LshGpuKernelLauncher(const int lsh_m, const int M, const int L, const int* a, const int* b, int* out);

class LshGpuOp: public OpKernel {
public:
  explicit LshGpuOp(OpKernelConstruction* context): OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor_a = context->input(0);
    const Tensor& input_tensor_b = context->input(1);
    auto input_a = input_tensor_a.flat<int32>();
    auto input_b = input_tensor_b.matrix<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    const int lsh_m = input_a.size() - 1;
    const int M = input_b.dimension(0);
    const int lsh_m_mul_L = input_b.dimension(1);
    const int L = lsh_m_mul_L/lsh_m;

    TensorShape ts;
    ts.AddDim(M*L);
    OP_REQUIRES_OK(context, context->allocate_output(0, ts, &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    LshGpuKernelLauncher(lsh_m, M, L, input_a.data(), input_b.data(), output_flat.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("LshGpu").Device(DEVICE_GPU), LshGpuOp);
