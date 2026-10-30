# Baseline GPU profile

Active profiled steps: 16
Total GPU time: 14649.057 ms

Groups use kernel-name heuristics; this is GPU time, not wall time or own-kernel share.

| Group | ms | % of GPU time |
|---|---:|---:|
| matmul | 5711.701 | 38.99 |
| elementwise | 3393.829 | 23.17 |
| cast/copy | 2616.697 | 17.86 |
| attention | 903.232 | 6.17 |
| softmax/loss | 892.413 | 6.09 |
| nf4 dequant | 690.928 | 4.72 |
| reduce | 254.158 | 1.73 |
| other | 186.099 | 1.27 |

Top 30 kernels

| Kernel | ms | % | Calls |
|---|---:|---:|---:|
| `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_nn` | 2276.739 | 15.54 | 2180 |
| `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_tn` | 1923.366 | 13.13 | 884 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>)` | 1244.828 | 8.50 | 9456 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>)` | 1206.194 | 8.23 | 4311 |
| `turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn` | 997.177 | 6.81 | 896 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>)` | 941.378 | 6.43 | 10752 |
| `void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous namespace)::LogSoftMaxForwardEpilogue>(float*, float const*, int)` | 671.328 | 4.58 | 263 |
| `void kDequantizeBlockwise<__half, 512, 64, 8, 2>(float*, unsigned char*, float*, __half*, int, int)` | 650.812 | 4.44 | 6224 |
| `fmha_cutlassB_f16_aligned_128x64_k128_sm75(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 128, 64, 128, false>::Params)` | 623.193 | 4.25 | 448 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)` | 527.086 | 3.60 | 5744 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 449.432 | 3.07 | 7168 |
| `fmha_cutlassF_f16_aligned_32x128_rf_sm75(PyTorchMemEffAttention::AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, true, true>::Params)` | 280.039 | 1.91 | 448 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::silu_backward_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float, float)#1}, std::array<char*, 3ul> >(int, at::native::(anonymous namespace)::silu_backward_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float, float)#1}, std::array<char*, 3ul>)` | 279.484 | 1.91 | 448 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1})` | 267.520 | 1.83 | 4000 |
| `void at::native::(anonymous namespace)::cunn_SoftMaxBackward<4, float, float, float, at::native::(anonymous namespace)::LogSoftMaxBackwardEpilogue>(float*, float const*, float const*, long)` | 220.273 | 1.50 | 16 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>)` | 191.925 | 1.31 | 448 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1})` | 139.611 | 0.95 | 7568 |
| `turing_fp16_s1688gemm_fp16_128x128_ldg8_relu_f2f_tn` | 138.448 | 0.95 | 952 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::exp_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::exp_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>)` | 115.813 | 0.79 | 183 |
| `turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn` | 106.866 | 0.73 | 330 |
| `void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, 4, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<2>, at::native::memory::StoreWithCast<1> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<2>, at::native::memory::StoreWithCast<1>)` | 105.670 | 0.72 | 1344 |
| `Memcpy DtoD (Device -> Device)` | 101.522 | 0.69 | 1824 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4>)` | 100.796 | 0.69 | 1223 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)` | 99.212 | 0.68 | 1888 |
| `void at::native::(anonymous namespace)::fused_dropout_kernel_vec<float, float, unsigned int, 1, 4, bool>(at::cuda::detail::TensorInfo<float const, unsigned int>, at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<bool, unsigned int>, unsigned int, float, at::PhiloxCudaState)` | 81.280 | 0.55 | 896 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::masked_scale_kernel<bool, float, float>(at::Tensor&, at::Tensor const&, at::Tensor const&, float)::{lambda(float, bool)#1}, std::array<char*, 3ul> >(int, at::native::(anonymous namespace)::masked_scale_kernel<bool, float, float>(at::Tensor&, at::Tensor const&, at::Tensor const&, float)::{lambda(float, bool)#1}, std::array<char*, 3ul>)` | 71.069 | 0.49 | 864 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)` | 69.029 | 0.47 | 1872 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase&, float)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase&, float)::{lambda(float)#1}, std::array<char*, 2ul>)` | 67.057 | 0.46 | 912 |
| `turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn` | 63.670 | 0.43 | 1458 |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<4u>, unsigned int, 4, 64, 64>(at::native::(anonymous namespace)::OpaqueType<4u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<4u>, unsigned int, 64, 64>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int)` | 51.972 | 0.35 | 896 |

Source: https://www.kaggle.com/code/ansargabit/mindx-person-a-memory-profile (version 2). Baseline collected with a no-op apply() in a disposable checkout copy. Team solution files were not changed. Profile timings are not benchmark wall times.

Classification gives attention priority over CUTLASS, and ignores the substring `nocast` when identifying casts. Regenerate with `python tools/profile_summary.py <profile.json> --out docs`.

To collect a fresh unpatched baseline profile on Kaggle after preparing data:

```bash
python task/prepare_data.py
python tools/profile_summary.py --run-base-profile --out /kaggle/working/out
```

The script copies the repository to a temporary directory, replaces apply() only in that copy, and writes the profile and report to the requested output directory. The official scored run uses the real team solution.
