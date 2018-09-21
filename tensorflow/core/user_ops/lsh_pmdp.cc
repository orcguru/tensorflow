/*
 * computeProductModDefaultPrime function implementation based on E2LSH-0.1
 */
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("LshPmdp")
  .Input("a: int32")
  .Input("b: int32")
  .Input("mod_flag: int32")
  .Output("h: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

class LshPmdpOp: public OpKernel {
public:
  explicit LshPmdpOp(OpKernelConstruction* context): OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor_a = context->input(0);
    const Tensor& input_tensor_b = context->input(1);
    const Tensor& input_tensor_mf = context->input(2);
    auto input_a = input_tensor_a.flat<int32>();
    auto input_b = input_tensor_b.matrix<int32>();
    auto input_mf = input_tensor_mf.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    const int lsh_m = input_a.size();
    const int M = input_b.dimension(0);
    const int lsh_m_mul_L = input_b.dimension(1);
    const int L = lsh_m_mul_L/lsh_m;
    const int mf = input_mf(0);

    TensorShape ts;
    ts.AddDim(M*L);
    OP_REQUIRES_OK(context, context->allocate_output(0, ts, &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    for (int p = 0; p < M; p++) {
      for (int li = 0; li < L; li++) {
        unsigned long h = 0;
        for (int i = 0; i < lsh_m; i++) {
          h = h + ((unsigned long)((unsigned int)input_a(i)) * (unsigned long)((unsigned int)input_b(p, li*lsh_m+i)));
          h = (h & 4294967295UL) + 5 * (h >> 32);
          if (h >= 4294967291UL) {
            h = h - 4294967291UL;
          }
        }
        if (mf != 0) {
          output_flat(p*L+li) = (unsigned int)(h%(unsigned long)mf);
        } else {
          output_flat(p*L+li) = (unsigned int)h;
        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("LshPmdp").Device(DEVICE_CPU), LshPmdpOp);
