// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/image/downsampler.hpp"
#include "gpu/2dBuffer.hpp"

#include "../deviceBuffer2D.hpp"
#include "../deviceStream.hpp"
#include "../surface.hpp"
#include "../gpuKernelDef.h"

#include "cuda/util.hpp"

#include "backend/cuda/core1/kernels/defKernel.cu"

#include "backend/common/image/downsampler.gpu"

#include <cuda_runtime.h>
#include <cassert>

namespace VideoStitch {
namespace Image {

Status downsample(GPU::Buffer2D& in, GPU::Buffer2D& out, GPU::Stream stream) {
  assert(in.getWidth() / out.getWidth() == in.getHeight() / out.getHeight());
  int factor = (int)(in.getWidth() / out.getWidth());
  const dim3 dimBlock(16, 16, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(out.getWidth(), dimBlock.x),
                     (unsigned)Cuda::ceilDiv(out.getHeight(), dimBlock.y), 1);
  downsamplePlanarKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      out.get().raw(), (unsigned)out.getPitch(), in.get().raw(), (unsigned)in.getPitch(), (unsigned)in.getWidth(),
      (unsigned)in.getHeight(), factor);
  return CUDA_STATUS;
}

Status downsampleRGBASurf2x(GPU::Surface& dst, const GPU::Surface& src, unsigned dstWidth, unsigned dstHeight,
                            GPU::Stream stream) {
  if (src.width() != 2 * dst.width() || src.height() != 2 * dst.height()) {
    return {Origin::GPU, ErrType::UnsupportedAction, "Downsampling RGBA surfaces only implemented for even dimensions"};
  }
  const dim3 dimBlock(16, 16, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(dstWidth, dimBlock.x), (unsigned)Cuda::ceilDiv(dstHeight, dimBlock.y), 1);
  downsampleRGBASurfKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get().surface(), src.get().texture(), dstWidth,
                                                                   dstHeight);
  return CUDA_STATUS;
}

Status downsampleRGBA(int factor, GPU::Buffer2D& in, GPU::Buffer2D& out, GPU::Stream stream) {
  const dim3 dimBlock(16, 16, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(out.getWidth(), dimBlock.x),
                     (unsigned)Cuda::ceilDiv(out.getHeight(), dimBlock.y), 1);
  downsampleRGBAKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      out.get().raw(), (unsigned)out.getPitch(), in.get().raw(), (unsigned)in.getPitch(), (unsigned)in.getWidth(),
      (unsigned)in.getHeight(), factor);
  return CUDA_STATUS;
}

Status downsampleRGB(int factor, GPU::Buffer2D& in, GPU::Buffer2D& out, GPU::Stream stream) {
  const dim3 dimBlock(16, 16, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(out.getWidth(), dimBlock.x),
                     (unsigned)Cuda::ceilDiv(out.getHeight(), dimBlock.y), 1);
  downsampleRGBKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(out.get(), (unsigned)out.getPitch(), in.get().raw(),
                                                              (unsigned)in.getPitch(), (unsigned)in.getWidth(),
                                                              (unsigned)in.getHeight(), factor);
  return CUDA_STATUS;
}

Status downsampleYUV422(int factor, GPU::Buffer2D& in, GPU::Buffer2D& out, GPU::Stream stream) {
  const dim3 dimBlock(16, 16, 1);
  // each thread accumulates for 2 (horizontally) consecutive destination pixels
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(out.getWidth() / 2, dimBlock.x),
                     (unsigned)Cuda::ceilDiv(out.getHeight(), dimBlock.y), 1);
  downsampleYUV422Kernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      out.get().raw(), (unsigned)out.getPitch(), in.get().raw(), (unsigned)in.getPitch(), (unsigned)in.getWidth(),
      (unsigned)in.getHeight(), factor);
  return CUDA_STATUS;
}

Status downsampleYUV422P10(int factor, GPU::Buffer2D& yIn, GPU::Buffer2D& uIn, GPU::Buffer2D& vIn, GPU::Buffer2D& yOut,
                           GPU::Buffer2D& uOut, GPU::Buffer2D& vOut, GPU::Stream stream) {
  const dim3 dimBlock(16, 16, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(yOut.getWidth(), dimBlock.x),
                     (unsigned)Cuda::ceilDiv(yOut.getHeight(), dimBlock.y), 1);
  downsampleYUV422P10Kernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      yOut.get().raw(), (unsigned)yOut.getPitch(), uOut.get().raw(), (unsigned)uOut.getPitch(), vOut.get().raw(),
      (unsigned)vOut.getPitch(), yIn.get().raw(), (unsigned)yIn.getPitch(), uIn.get().raw(), (unsigned)uIn.getPitch(),
      vIn.get().raw(), (unsigned)vIn.getPitch(), (unsigned)yIn.getWidth(), (unsigned)yIn.getHeight(), factor);
  return CUDA_STATUS;
}

Status downsampleYV12(int factor, GPU::Buffer2D& yIn, GPU::Buffer2D& uIn, GPU::Buffer2D& vIn, GPU::Buffer2D& yOut,
                      GPU::Buffer2D& uOut, GPU::Buffer2D& vOut, GPU::Stream stream) {
  const dim3 dimBlock(16, 16, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(yOut.getWidth(), dimBlock.x),
                     (unsigned)Cuda::ceilDiv(yOut.getHeight(), dimBlock.y), 1);
  downsampleYV12Kernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      yOut.get().raw(), (unsigned)yOut.getPitch(), uOut.get().raw(), (unsigned)uOut.getPitch(), vOut.get().raw(),
      (unsigned)vOut.getPitch(), yIn.get().raw(), (unsigned)yIn.getPitch(), uIn.get().raw(), (unsigned)uIn.getPitch(),
      vIn.get().raw(), (unsigned)vIn.getPitch(), (unsigned)yIn.getWidth(), (unsigned)yIn.getHeight(), factor);
  return CUDA_STATUS;
}

Status downsampleNV12(int factor, GPU::Buffer2D& yIn, GPU::Buffer2D& uvIn, GPU::Buffer2D& yOut, GPU::Buffer2D& uvOut,
                      GPU::Stream stream) {
  const dim3 dimBlock(16, 16, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(yOut.getWidth(), dimBlock.x),
                     (unsigned)Cuda::ceilDiv(yOut.getHeight(), dimBlock.y), 1);
  downsampleNV12Kernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      yOut.get().raw(), (unsigned)yOut.getPitch(), uvOut.get().raw(), (unsigned)uvOut.getPitch(), yIn.get().raw(),
      (unsigned)yIn.getPitch(), uvIn.get().raw(), (unsigned)uvIn.getPitch(), (unsigned)yIn.getWidth(),
      (unsigned)yIn.getHeight(), factor);
  return CUDA_STATUS;
}
}  // namespace Image
}  // namespace VideoStitch
