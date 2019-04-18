// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/image/imageOps.hpp"

#include "cuda/util.hpp"

#include "../deviceBuffer.hpp"
#include "../deviceStream.hpp"
#include "../gpuKernelDef.h"

#include <backend/common/image/rotate.gpu>

namespace VideoStitch {
namespace Image {

Status flip(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, size_t length, GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(length, 16), (unsigned)Cuda::ceilDiv(length, 16), 1);

  flipKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), src.get(), (unsigned)length);
  return CUDA_STATUS;
}

Status flop(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, size_t length, GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(length, 16), (unsigned)Cuda::ceilDiv(length, 16), 1);

  flopKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), src.get(), (unsigned)length);
  return CUDA_STATUS;
}

Status rotate(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, size_t length, GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(length, 16), (unsigned)Cuda::ceilDiv(length, 16), 1);

  rotateKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), src.get(), (unsigned)length);
  return CUDA_STATUS;
}

Status rotateLeft(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, size_t length, GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(length, 16), (unsigned)Cuda::ceilDiv(length, 16), 1);

  rotateLeftKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), src.get(), (unsigned)length);
  return CUDA_STATUS;
}

Status rotateRight(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, size_t length, GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(length, 16), (unsigned)Cuda::ceilDiv(length, 16), 1);

  rotateRightKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), src.get(), (unsigned)length);
  return CUDA_STATUS;
}

}  // namespace Image
}  // namespace VideoStitch
