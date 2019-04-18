// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/core1/mergerKernel.hpp"

#include "../deviceBuffer.hpp"
#include "../deviceStream.hpp"

#include "backend/common/imageOps.hpp"
#include "cuda/error.hpp"
#include "cuda/util.hpp"
#include "libvideostitch/panoDef.hpp"
#include "core1/imageMapping.hpp"

namespace VideoStitch {

namespace {

// CUDA/OpenCL shared implementation
#include "../gpuKernelDef.h"

#include <backend/common/core1/mergerKernel.gpu>
}  // namespace

namespace Core {

Status countInputs(TextureTarget t, const PanoDefinition& pano, GPU::Buffer<uint32_t> pbo, const ImageMapping& fromIm,
                   GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();

  if (fromIm.getOutputRect(t).empty()) {
    return Status::OK();
  }

  const dim3 dimBlock(16, 16, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(fromIm.getOutputRect(t).getWidth(), dimBlock.x),
                     (unsigned)Cuda::ceilDiv(fromIm.getOutputRect(t).getHeight(), dimBlock.y), 1);
  if (fromIm.getOutputRect(t).right() >= (int64_t)pano.getWidth()) {
    countInputsKernelWrap<<<dimGrid, dimBlock, 0, stream>>>(
        pbo.get(), (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), fromIm.getDeviceOutputBuffer(t).get(),
        (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight(),
        (unsigned)fromIm.getOutputRect(t).left(), (unsigned)fromIm.getOutputRect(t).top());
  } else {
    countInputsKernelNoWrap<<<dimGrid, dimBlock, 0, stream>>>(
        pbo.get(), (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), fromIm.getDeviceOutputBuffer(t).get(),
        (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight(),
        (unsigned)fromIm.getOutputRect(t).left(), (unsigned)fromIm.getOutputRect(t).top());
  }
  return CUDA_STATUS;
}

Status colorMap(const PanoDefinition& pano, GPU::Buffer<uint32_t> pbo, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();

  const int64_t size = pano.getWidth() * pano.getHeight();
  dim3 dimBlock(512);
  dim3 dimGrid(Cuda::compute2DGridForFlatBuffer(size, 512));
  colormapKernel<<<dimGrid, dimBlock, 0, stream>>>(pbo.get(), (unsigned)pano.getWidth(), (unsigned)size);
  return CUDA_ERROR(cudaStreamSynchronize(stream));
}

Status stitchingError(TextureTarget t, const PanoDefinition& pano, GPU::Buffer<uint32_t> pbo,
                      const ImageMapping& fromIm, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();

  if (fromIm.getOutputRect(t).empty()) {
    return Status::OK();
  }

  const dim3 dimBlock(16, 16, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(fromIm.getOutputRect(t).getWidth(), dimBlock.x),
                     (unsigned)Cuda::ceilDiv(fromIm.getOutputRect(t).getHeight(), dimBlock.y), 1);
  if (fromIm.getOutputRect(t).right() >= (int64_t)pano.getWidth()) {
    stitchingErrorKernelWrap<<<dimGrid, dimBlock, 0, stream>>>(
        pbo.get(), (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), fromIm.getDeviceOutputBuffer(t).get(),
        (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight(),
        (unsigned)fromIm.getOutputRect(t).left(), (unsigned)fromIm.getOutputRect(t).top());
  } else {
    stitchingErrorKernelNoWrap<<<dimGrid, dimBlock, 0, stream>>>(
        pbo.get(), (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), fromIm.getDeviceOutputBuffer(t).get(),
        (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight(),
        (unsigned)fromIm.getOutputRect(t).left(), (unsigned)fromIm.getOutputRect(t).top());
  }
  return CUDA_STATUS;
}

Status exposureDiffRGB(TextureTarget t, const PanoDefinition& pano, GPU::Buffer<uint32_t> pbo,
                       const ImageMapping& fromIm, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();

  if (fromIm.getOutputRect(t).empty()) {
    return Status::OK();
  }

  const dim3 dimBlock(16, 16, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(fromIm.getOutputRect(t).getWidth(), dimBlock.x),
                     (unsigned)Cuda::ceilDiv(fromIm.getOutputRect(t).getHeight(), dimBlock.y), 1);
  if (fromIm.getOutputRect(t).right() >= (int64_t)pano.getWidth()) {
    exposureErrorRGBKernelWrap<<<dimGrid, dimBlock, 0, stream>>>(
        pbo.get(), (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), fromIm.getDeviceOutputBuffer(t).get(),
        (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight(),
        (unsigned)fromIm.getOutputRect(t).left(), (unsigned)fromIm.getOutputRect(t).top());
  } else {
    exposureErrorRGBKernelNoWrap<<<dimGrid, dimBlock, 0, stream>>>(
        pbo.get(), (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), fromIm.getDeviceOutputBuffer(t).get(),
        (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight(),
        (unsigned)fromIm.getOutputRect(t).left(), (unsigned)fromIm.getOutputRect(t).top());
  }
  return CUDA_STATUS;
}

Status amplitude(const PanoDefinition& pano, GPU::Buffer<uint32_t> pbo, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();

  const int64_t size = pano.getWidth() * pano.getHeight();
  dim3 dimBlock(512);
  dim3 dimGrid(Cuda::compute2DGridForFlatBuffer(size, 512));
  amplitudeKernel<<<dimGrid, dimBlock, 0, stream>>>(pbo.get(), 0, (3 * 256 * 256), (unsigned)pano.getWidth(),
                                                    (unsigned)size);
  return CUDA_ERROR(cudaStreamSynchronize(stream));
}

Status disregardNoDiffArea(const PanoDefinition& pano, GPU::Buffer<uint32_t> pbo, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();

  const int64_t size = pano.getWidth() * pano.getHeight();
  dim3 dimBlock(512);
  dim3 dimGrid(Cuda::compute2DGridForFlatBuffer(size, 512));
  maskOutSingleInput<<<dimGrid, dimBlock, 0, stream>>>(pbo.get(), (unsigned)pano.getWidth(), (unsigned)size);
  return CUDA_ERROR(cudaStreamSynchronize(stream));
}

Status checkerMerge(TextureTarget t, const PanoDefinition& pano, GPU::Buffer<uint32_t> pbo, const ImageMapping& fromIm,
                    unsigned checkerSize, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();

  if (fromIm.getOutputRect(t).empty()) {
    return Status::OK();
  }
  const dim3 dimBlock(16, 16, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(fromIm.getOutputRect(t).getWidth(), dimBlock.x),
                     (unsigned)Cuda::ceilDiv(fromIm.getOutputRect(t).getHeight(), dimBlock.y), 1);
  if (fromIm.getOutputRect(t).right() >= (int64_t)pano.getWidth()) {
    checkerInsertKernelWrap<<<dimGrid, dimBlock, 0, stream>>>(
        pbo.get(), (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), fromIm.getDeviceOutputBuffer(t).get(),
        (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight(),
        (unsigned)fromIm.getOutputRect(t).left(), (unsigned)fromIm.getOutputRect(t).top(), checkerSize);
  } else {
    checkerInsertKernelNoWrap<<<dimGrid, dimBlock, 0, stream>>>(
        pbo.get(), (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), fromIm.getDeviceOutputBuffer(t).get(),
        (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight(),
        (unsigned)fromIm.getOutputRect(t).left(), (unsigned)fromIm.getOutputRect(t).top(), checkerSize);
  }
  return CUDA_STATUS;
}

Status noblend(TextureTarget t, const PanoDefinition& pano, GPU::Buffer<uint32_t> pbo, const ImageMapping& fromIm,
               GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();

  if (fromIm.getOutputRect(t).empty()) {
    return Status::OK();
  }

  const dim3 dimBlock(16, 16, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(fromIm.getOutputRect(t).getWidth(), dimBlock.x),
                     (unsigned)Cuda::ceilDiv(fromIm.getOutputRect(t).getHeight(), dimBlock.y), 1);
  if (fromIm.getOutputRect(t).right() >= (int64_t)pano.getWidth()) {
    noblendKernelWrap<<<dimGrid, dimBlock, 0, stream>>>(
        pbo.get(), (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), fromIm.getDeviceOutputBuffer(t).get(),
        (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight(),
        (unsigned)fromIm.getOutputRect(t).left(), (unsigned)fromIm.getOutputRect(t).top());
  } else {
    noblendKernelNoWrap<<<dimGrid, dimBlock, 0, stream>>>(
        pbo.get(), (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), fromIm.getDeviceOutputBuffer(t).get(),
        (unsigned)fromIm.getOutputRect(t).getWidth(), (unsigned)fromIm.getOutputRect(t).getHeight(),
        (unsigned)fromIm.getOutputRect(t).left(), (unsigned)fromIm.getOutputRect(t).top());
  }
  return CUDA_STATUS;
}
}  // namespace Core
}  // namespace VideoStitch
