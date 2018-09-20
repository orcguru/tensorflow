/*
 * computeProductModDefaultPrime function implementation based on E2LSH-0.1
 */
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;

REGISTER_OP("LshMcpu")
  .Input("a: int32")
  .Input("b: int32")
  .Output("h: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

class LshMcpuOp: public OpKernel {
public:
  explicit LshMcpuOp(OpKernelConstruction* context): OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor_a = context->input(0);
    const Tensor& input_tensor_b = context->input(1);
    auto input_a = input_tensor_a.flat<int32>();
    auto input_b = input_tensor_b.matrix<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    TensorShape ts;
    ts.AddDim(input_b.dimension(0));
    OP_REQUIRES_OK(context, context->allocate_output(0, ts, &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    const int P = input_b.dimension(0);
    const int N = input_b.dimension(1);
    const DeviceBase::CpuWorkerThreads& worker_threads =
      *context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads.num_threads, worker_threads.workers, P,
      2,
      [&input_a, &input_b, &output_flat, N](int64 start_channel, int64 end_channel) {
        for (int p = start_channel; p < end_channel; p++) {
          unsigned long h = 0;
          for (int i = 0; i < N; i++) {
            h = h + ((unsigned long)((unsigned int)input_a(i)) * (unsigned long)((unsigned int)input_b(p, i)));
            h = (h & 4294967295UL) + 5 * (h >> 32);
            if (h >= 4294967291UL) {
              h = h - 4294967291UL;
            }
          }
          output_flat(p) = (unsigned int)h;
        }
      });
  }
};

REGISTER_KERNEL_BUILDER(Name("LshMcpu").Device(DEVICE_CPU), LshMcpuOp);
