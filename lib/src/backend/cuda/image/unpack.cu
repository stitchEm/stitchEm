// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "image/unpack.hpp"
#include "colorArrayDevice.hpp"

#include "backend/cuda/deviceBuffer.hpp"
#include "backend/cuda/deviceBuffer2D.hpp"
#include "backend/cuda/surface.hpp"
#include "backend/cuda/deviceStream.hpp"

#include "cuda/util.hpp"

#include "unpackKernel.cu"

#include <cuda_runtime.h>
#include <cassert>

const unsigned int CudaBlockSize = 16;

namespace VideoStitch {
namespace Image {

// ---------------- Convert RGBA -> other colorspace --------------------------

Status unpackRGB(GPU::Buffer2D& dst, const GPU::Buffer<const uint32_t>& array, std::size_t width, std::size_t height,
                 GPU::Stream s) {
  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  unpackKernelRGB<<<dimGrid, dimBlock, 0, s.get()>>>(dst.get().raw(), (unsigned)dst.getPitch(), array.get(),
                                                     (unsigned)width, (unsigned)height);
  return CUDA_STATUS;
}

Status unpackRGB(GPU::Buffer2D& dst, const GPU::Surface& surf, std::size_t width, std::size_t height, GPU::Stream s) {
  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  unpackSourceKernelRGB<<<dimGrid, dimBlock, 0, s.get()>>>(dst.get().raw(), (unsigned)dst.getPitch(),
                                                           surf.get().surface(), (unsigned)width, (unsigned)height);
  return CUDA_STATUS;
}

Status unpackRGBA(GPU::Buffer2D& dst, const GPU::Buffer<const uint32_t>& array, std::size_t /*width*/,
                  std::size_t /*height*/, GPU::Stream s) {
  return CUDA_ERROR(cudaMemcpy2DAsync(dst.get().raw(), (unsigned)dst.getPitch(), array.get(), dst.getWidth(),
                                      dst.getWidth(), dst.getHeight(), cudaMemcpyDeviceToDevice, s.get()));
}

Status unpackRGBA(GPU::Buffer2D& dst, const GPU::Surface& surf, std::size_t width, std::size_t height, GPU::Stream s) {
  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  unpackSourceKernelRGBA<<<dimGrid, dimBlock, 0, s.get()>>>(
      (uint32_t*)dst.get().raw(), (unsigned)dst.getPitch() / sizeof(uint32_t),  // pitch is in bytes
      surf.get().surface(), (unsigned)width, (unsigned)height);
  return CUDA_STATUS;
}

Status unpackF32C1(GPU::Buffer2D& dst, const GPU::Buffer<const uint32_t>& array, std::size_t /*width*/,
                   std::size_t /*height*/, GPU::Stream s) {
  return CUDA_ERROR(cudaMemcpy2DAsync(dst.get().raw(), (unsigned)dst.getPitch(), array.get(), dst.getWidth(),
                                      dst.getWidth(), dst.getHeight(), cudaMemcpyDeviceToDevice, s.get()));
}

Status unpackF32C1(GPU::Buffer2D& dst, const GPU::Surface& surf, std::size_t width, std::size_t height, GPU::Stream s) {
  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  unpackSourceKernelF32C1<<<dimGrid, dimBlock, 0, s.get()>>>(
      (float*)dst.get().raw(), (unsigned)dst.getPitch() / sizeof(float),  // pitch is in bytes
      surf.get().surface(), (unsigned)width, (unsigned)height);
  return CUDA_STATUS;
}

Status unpackGrayscale16(GPU::Buffer2D& /* dst */, const GPU::Buffer<const uint32_t>& /* input */, size_t /* width*/,
                         size_t /* height */, GPU::Stream /* s */) {
  // TODO
  return {Origin::GPU, ErrType::UnsupportedAction,
          "Color space conversion for Grayscale16 not implemented from buffer"};
}

Status unpackGrayscale16(GPU::Buffer2D& dst, const GPU::Surface& surf, size_t width, size_t height, GPU::Stream s) {
  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  unpackSourceKernelGrayscale16<<<dimGrid, dimBlock, 0, s.get()>>>(
      (uint16_t*)dst.get().raw(), (unsigned)dst.getPitch() / sizeof(uint16_t),  // pitch is in bytes
      surf.get().surface(), (unsigned)width, (unsigned)height);
  return CUDA_STATUS;
}

Status unpackDepth(GPU::Buffer2D& yDst, GPU::Buffer2D& uDst, GPU::Buffer2D& vDst,
                   const GPU::Buffer<const uint32_t>& array, std::size_t width, std::size_t height, GPU::Stream s) {
  assert(!(width & 1));
  assert(!(height & 1));
  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv((width + 1) / 2, dimBlock.x),
                     (unsigned)Cuda::ceilDiv((height + 1) / 2, dimBlock.y), 1);
  unpackKernelDepth<<<dimGrid, dimBlock, 0, s.get()>>>(
      yDst.get().raw(), (unsigned)yDst.getPitch(), uDst.get().raw(), (unsigned)uDst.getPitch(), vDst.get().raw(),
      (unsigned)vDst.getPitch(), (float*)array.get().raw(), (unsigned)width, (unsigned)height);
  return CUDA_STATUS;
}

Status unpackDepth(GPU::Buffer2D& yDst, GPU::Buffer2D& uDst, GPU::Buffer2D& vDst, const GPU::Surface& surf,
                   std::size_t width, std::size_t height, GPU::Stream s) {
  assert(!(width & 1));
  assert(!(height & 1));
  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv((width + 1) / 2, dimBlock.x),
                     (unsigned)Cuda::ceilDiv((height + 1) / 2, dimBlock.y), 1);
  unpackSourceKernelDepth<<<dimGrid, dimBlock, 0, s.get()>>>(
      yDst.get().raw(), (unsigned)yDst.getPitch(), uDst.get().raw(), (unsigned)uDst.getPitch(), vDst.get().raw(),
      (unsigned)vDst.getPitch(), surf.get().surface(), (unsigned)width, (unsigned)height);
  return CUDA_STATUS;
}

Status unpackYV12(GPU::Buffer2D& yDst, GPU::Buffer2D& uDst, GPU::Buffer2D& vDst,
                  const GPU::Buffer<const uint32_t>& array, std::size_t width, std::size_t height, GPU::Stream s) {
  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv((width + 1) / 2, dimBlock.x),
                     (unsigned)Cuda::ceilDiv((height + 1) / 2, dimBlock.y), 1);
  unpackKernelYV12<<<dimGrid, dimBlock, 0, s.get()>>>(
      yDst.get().raw(), (unsigned)yDst.getPitch(), uDst.get().raw(), (unsigned)uDst.getPitch(), vDst.get().raw(),
      (unsigned)vDst.getPitch(), array.get(), (unsigned)width, (unsigned)height);
  return CUDA_STATUS;
}

Status unpackYV12(GPU::Buffer2D& yDst, GPU::Buffer2D& uDst, GPU::Buffer2D& vDst, const GPU::Surface& surf,
                  std::size_t width, std::size_t height, GPU::Stream s) {
  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv((width + 1) / 2, dimBlock.x),
                     (unsigned)Cuda::ceilDiv((height + 1) / 2, dimBlock.y), 1);
  unpackSourceKernelYV12<<<dimGrid, dimBlock, 0, s.get()>>>(
      yDst.get().raw(), (unsigned)yDst.getPitch(), uDst.get().raw(), (unsigned)uDst.getPitch(), vDst.get().raw(),
      (unsigned)vDst.getPitch(), surf.get().surface(), (unsigned)width, (unsigned)height);
  return CUDA_STATUS;
}

Status unpackNV12(GPU::Buffer2D& yDst, GPU::Buffer2D& uvDst, const GPU::Surface& surf, std::size_t width,
                  std::size_t height, GPU::Stream s) {
  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv((width + 1) / 2, dimBlock.x),
                     (unsigned)Cuda::ceilDiv((height + 1) / 2, dimBlock.y), 1);
  unpackSourceKernelNV12<<<dimGrid, dimBlock, 0, s.get()>>>(yDst.get().raw(), (unsigned)yDst.getPitch(),
                                                            uvDst.get().raw(), (unsigned)uvDst.getPitch(),
                                                            surf.get().surface(), (unsigned)width, (unsigned)height);
  return CUDA_STATUS;
}

Status unpackNV12(GPU::Buffer2D& yDst, GPU::Buffer2D& uvDst, const GPU::Buffer<const uint32_t>& array,
                  std::size_t width, std::size_t height, GPU::Stream s) {
  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv((width + 1) / 2, dimBlock.x),
                     (unsigned)Cuda::ceilDiv((height + 1) / 2, dimBlock.y), 1);
  unpackKernelNV12<<<dimGrid, dimBlock, 0, s.get()>>>(yDst.get().raw(), (unsigned)yDst.getPitch(), uvDst.get().raw(),
                                                      (unsigned)uvDst.getPitch(), array.get(), (unsigned)width,
                                                      (unsigned)height);
  return CUDA_STATUS;
}

Status unpackYUY2(GPU::Buffer2D& dst, const GPU::Buffer<const uint32_t>& src, std::size_t width, std::size_t height,
                  GPU::Stream stream) {
  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv((width + 1) / 2, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y),
                     1);
  unpackYUY2Kernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get().raw(), (unsigned)dst.getPitch(), src.get(),
                                                           (unsigned)width, (unsigned)height);
  return CUDA_STATUS;
}

Status unpackYUY2(GPU::Buffer2D&, const GPU::Surface&, std::size_t, std::size_t, GPU::Stream) {
  return Status{Origin::GPU, ErrType::ImplementationError, "Unpacking not implemented from Surface"};
}

Status unpackUYVY(GPU::Buffer2D& dst, const GPU::Buffer<const uint32_t>& src, std::size_t width, std::size_t height,
                  GPU::Stream stream) {
  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv((width + 1) / 2, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y),
                     1);
  unpackUYVYKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get().raw(), (unsigned)dst.getPitch(), src.get(),
                                                           (unsigned)width, (unsigned)height);
  return CUDA_STATUS;
}

Status unpackUYVY(GPU::Buffer2D&, const GPU::Surface&, std::size_t, std::size_t, GPU::Stream) {
  return Status{Origin::GPU, ErrType::ImplementationError, "Unpacking not implemented from Surface"};
}

Status convertGrayscale(GPU::Buffer<uint32_t> dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                        std::size_t height, GPU::Stream stream) {
  const dim3 dimBlock2D(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid2D((unsigned)Cuda::ceilDiv(width, dimBlock2D.x), (unsigned)Cuda::ceilDiv(height, dimBlock2D.y), 1);
  convertKernelGrayscale<<<dimGrid2D, dimBlock2D, 0, stream.get()>>>(dst.get(), src.get(), (unsigned)width,
                                                                     (unsigned)height);
  return CUDA_STATUS;
}

Status unpackYUV422P10(GPU::Buffer2D& yDst, GPU::Buffer2D& uDst, GPU::Buffer2D& vDst,
                       const GPU::Buffer<const uint32_t>& src, std::size_t width, std::size_t height,
                       GPU::Stream stream) {
  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv((width + 1) / 2, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y),
                     1);
  unpackYUV422P10Kernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      reinterpret_cast<uint16_t*>(yDst.get().raw()), (unsigned)yDst.getPitch() / 2,
      reinterpret_cast<uint16_t*>(uDst.get().raw()), (unsigned)uDst.getPitch() / 2,
      reinterpret_cast<uint16_t*>(vDst.get().raw()), (unsigned)vDst.getPitch() / 2, src.get(), (unsigned)width,
      (unsigned)height);
  return CUDA_STATUS;
}

Status unpackYUV422P10(GPU::Buffer2D&, GPU::Buffer2D&, GPU::Buffer2D&, const GPU::Surface&, std::size_t, std::size_t,
                       GPU::Stream) {
  return Status{Origin::GPU, ErrType::ImplementationError, "Unpacking not implemented from Surface"};
}

Status unpackGrayscale(GPU::Buffer2D& dst, const GPU::Surface& src, std::size_t width, std::size_t height,
                       GPU::Stream stream) {
  const dim3 dimBlock2D(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid2D((unsigned)Cuda::ceilDiv(width, dimBlock2D.x), (unsigned)Cuda::ceilDiv(height, dimBlock2D.y), 1);
  unpackKernelGrayscale<<<dimGrid2D, dimBlock2D, 0, stream.get()>>>(
      dst.get().raw(), (unsigned)dst.getPitch(), src.get().surface(), (unsigned)width, (unsigned)height);
  return CUDA_STATUS;
}

// ---------------- Convert other colorspace -> RGBA --------------------------

Status convertRGBToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width, std::size_t height,
                        GPU::Stream stream) {
  const dim3 dimBlock2D(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid2D((unsigned)Cuda::ceilDiv(width, dimBlock2D.x), (unsigned)Cuda::ceilDiv(height, dimBlock2D.y), 1);
  convertRGBToRGBAKernel<<<dimGrid2D, dimBlock2D, 0, stream.get()>>>(dst.get().surface(), src.get(), (unsigned)width,
                                                                     (unsigned)height);
  return CUDA_STATUS;
}

Status convertRGB210ToRGBA(GPU::Surface& dst, GPU::Buffer<const uint32_t> src, std::size_t width, std::size_t height,
                           GPU::Stream stream) {
  const dim3 dimBlock2D(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid2D((unsigned)Cuda::ceilDiv(width, dimBlock2D.x), (unsigned)Cuda::ceilDiv(height, dimBlock2D.y), 1);
  convertRGB210ToRGBAKernel<<<dimGrid2D, dimBlock2D, 0, stream.get()>>>(dst.get().surface(), src.get(), (unsigned)width,
                                                                        (unsigned)height);
  return CUDA_STATUS;
}

Status convertBGRToRGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                        std::size_t height, GPU::Stream stream) {
  const dim3 dimBlock2D(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid2D((unsigned)Cuda::ceilDiv(width, dimBlock2D.x), (unsigned)Cuda::ceilDiv(height, dimBlock2D.y), 1);
  convertBGRToRGBAKernel<<<dimGrid2D, dimBlock2D, 0, stream.get()>>>(dst.get(), src.get(), (unsigned)width,
                                                                     (unsigned)height);
  return CUDA_STATUS;
}

Status convertBGRUToRGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                         std::size_t height, GPU::Stream stream) {
  const dim3 dimBlock2D(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid2D((unsigned)Cuda::ceilDiv(width / 2, dimBlock2D.x),
                       (unsigned)Cuda::ceilDiv(height / 2, dimBlock2D.y), 1);
  convertBGRUToRGBAKernel<<<dimGrid2D, dimBlock2D, 0, stream.get()>>>(dst.get(), src.get(), (unsigned)width,
                                                                      (unsigned)height);
  return CUDA_STATUS;
}

Status convertBayerRGGBToRGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                              std::size_t height, GPU::Stream stream) {
  assert(!(width & 1));
  assert(!(height & 1));
  const dim3 dimBlock2D(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid2D((unsigned)Cuda::ceilDiv(width / 2, dimBlock2D.x),
                       (unsigned)Cuda::ceilDiv(height / 2, dimBlock2D.y), 1);
  convertBayerRGGBToRGBAKernel<<<dimGrid2D, dimBlock2D, sizeof(uint32_t) * (dimBlock2D.x + 1) * (dimBlock2D.y + 1),
                                 stream.get()>>>(dst.get(), src.get(), (unsigned)width, (unsigned)height);
  return CUDA_STATUS;
}

Status convertBayerBGGRToRGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                              std::size_t height, GPU::Stream stream) {
  assert(!(width & 1));
  assert(!(height & 1));
  const dim3 dimBlock2D(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid2D((unsigned)Cuda::ceilDiv(width / 2, dimBlock2D.x),
                       (unsigned)Cuda::ceilDiv(height / 2, dimBlock2D.y), 1);
  convertBayerBGGRToRGBAKernel<<<dimGrid2D, dimBlock2D, sizeof(uint32_t) * (dimBlock2D.x + 1) * (dimBlock2D.y + 1),
                                 stream.get()>>>(dst.get(), src.get(), (unsigned)width, (unsigned)height);
  return CUDA_STATUS;
}

Status convertBayerGRBGToRGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                              std::size_t height, GPU::Stream stream) {
  assert(!(width & 1));
  assert(!(height & 1));
  const dim3 dimBlock2D(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid2D((unsigned)Cuda::ceilDiv(width / 2, dimBlock2D.x),
                       (unsigned)Cuda::ceilDiv(height / 2, dimBlock2D.y), 1);
  convertBayerGRBGToRGBAKernel<<<dimGrid2D, dimBlock2D, sizeof(uint32_t) * (dimBlock2D.x + 1) * (dimBlock2D.y + 1),
                                 stream.get()>>>(dst.get(), src.get(), (unsigned)width, (unsigned)height);
  return CUDA_STATUS;
}

Status convertBayerGBRGToRGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                              std::size_t height, GPU::Stream stream) {
  assert(!(width & 1));
  assert(!(height & 1));
  const dim3 dimBlock2D(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid2D((unsigned)Cuda::ceilDiv(width / 2, dimBlock2D.x),
                       (unsigned)Cuda::ceilDiv(height / 2, dimBlock2D.y), 1);
  convertBayerGBRGToRGBAKernel<<<dimGrid2D, dimBlock2D, sizeof(uint32_t) * (dimBlock2D.x + 1) * (dimBlock2D.y + 1),
                                 stream.get()>>>(dst.get(), src.get(), (unsigned)width, (unsigned)height);
  return CUDA_STATUS;
}

Status convertUYVYToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width, std::size_t height,
                         GPU::Stream stream) {
  const dim3 dimBlock(16, 16, 1);
  assert(!(width & 1));
  assert(!(height & 1));
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(width / 2, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  convertUYVYToRGBAKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get().surface(), src.get(), (unsigned)width,
                                                                  (unsigned)height);
  return CUDA_STATUS;
}

Status convertYUV422P10ToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                              std::size_t height, GPU::Stream stream) {
  assert(!(width & 1));
  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(width / 2, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  convertYUV422P10ToRGBAKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      dst.get().surface(), src.as<const uint16_t>().get(), (unsigned)width, (unsigned)height);
  return CUDA_STATUS;
}

Status convertYUY2ToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width, std::size_t height,
                         GPU::Stream stream) {
  const dim3 dimBlock(16, 16, 1);
  assert(!(width & 1));
  assert(!(height & 1));
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(width / 2, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  convertYUY2ToRGBAKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get().surface(), src.get(), (unsigned)width,
                                                                  (unsigned)height);
  return CUDA_STATUS;
}

Status convertYV12ToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width, std::size_t height,
                         GPU::Stream stream) {
  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  assert(!(width & 1));
  assert(!(height & 1));
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(width / 2, dimBlock.x), (unsigned)Cuda::ceilDiv(height / 2, dimBlock.y),
                     1);
  convertYV12ToRGBAKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get().surface(), src.get(), (unsigned)width,
                                                                  (unsigned)height);
  return CUDA_STATUS;
}

Status convertNV12ToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width, std::size_t height,
                         GPU::Stream stream) {
  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  assert(!(width & 1));
  assert(!(height & 1));
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(width / 2, dimBlock.x), (unsigned)Cuda::ceilDiv(height / 2, dimBlock.y),
                     1);
  convertNV12ToRGBAKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get().surface(), src.get(), (unsigned)width,
                                                                  (unsigned)height);
  return CUDA_STATUS;
}

Status convertYUV420ToMono(GPU::Buffer<unsigned char> dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                           std::size_t height, GPU::Stream stream) {
  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  assert(!(width & 1));
  assert(!(height & 1));
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(width / 2, dimBlock.x), (unsigned)Cuda::ceilDiv(height / 2, dimBlock.y),
                     1);
  unpackMonoKernelYUV420P<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), src.get(), (unsigned)width,
                                                                  (unsigned)height);
  return CUDA_STATUS;
}

Status convertGrayscaleToRGBA(GPU::Surface& dst, GPU::Buffer<const unsigned char> src, std::size_t width,
                              std::size_t height, GPU::Stream stream) {
  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  convertGrayscaleKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get().surface(), src.get(), (unsigned)width,
                                                                 (unsigned)height);
  return CUDA_STATUS;
}
}  // namespace Image
}  // namespace VideoStitch
