// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "backend/common/imageOps.hpp"
#include "backend/common/vectorOps.hpp"

#include "cuda/util.hpp"

#include "../deviceBuffer.hpp"
#include "../deviceStream.hpp"

#include "backend/common/imageOps.hpp"

#define CUDABLOCKSIZE 512

namespace VideoStitch {
namespace Image {

namespace {
template <typename PixelTypeIn, typename PixelTypeOut>
__global__ void subtractRGBKernel(uint32_t *dst, const uint32_t *toSubtract, std::size_t size) {
  std::size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    uint32_t vSrc = toSubtract[i];
    uint32_t vDst = dst[i];
    int32_t toSubtractIsSolid = !!PixelTypeIn::a(vSrc);
    int32_t r = PixelTypeIn::r(vDst) - toSubtractIsSolid * PixelTypeIn::r(vSrc);
    int32_t g = PixelTypeIn::g(vDst) - toSubtractIsSolid * PixelTypeIn::g(vSrc);
    int32_t b = PixelTypeIn::b(vDst) - toSubtractIsSolid * PixelTypeIn::b(vSrc);
    dst[i] = PixelTypeOut::pack(r, g, b, PixelTypeIn::a(vDst));
  }
}

template <typename T>
__global__ void subtractKernel(T *dst, const T *toSubtract, std::size_t size) {
  std::size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    dst[i] -= toSubtract[i];
  }
}
}  // namespace

Status subtractRGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toSubtract, std::size_t size,
                    GPU::Stream stream) {
  dim3 dimBlock(CUDABLOCKSIZE);
  dim3 dimGrid(Cuda::compute2DGridForFlatBuffer(size, CUDABLOCKSIZE));
  subtractRGBKernel<RGBA, RGBA><<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), toSubtract.get(), size);
  return CUDA_STATUS;
}

Status subtractRGB210(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toSubtract, std::size_t size,
                      GPU::Stream stream) {
  dim3 dimBlock(CUDABLOCKSIZE);
  dim3 dimGrid(Cuda::compute2DGridForFlatBuffer(size, CUDABLOCKSIZE));
  subtractRGBKernel<RGB210, RGB210><<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), toSubtract.get(), size);
  return CUDA_STATUS;
}

Status subtract(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toSubtract, std::size_t size,
                GPU::Stream stream) {
  dim3 dimBlock(CUDABLOCKSIZE);
  dim3 dimGrid(Cuda::compute2DGridForFlatBuffer(size, CUDABLOCKSIZE));
  subtractRGBKernel<RGBA, RGB210><<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), toSubtract.get(), size);
  return CUDA_STATUS;
}

template <typename T>
Status subtractRaw(GPU::Buffer<T> dst, GPU::Buffer<const T> toSubtract, std::size_t size, GPU::Stream stream) {
  dim3 dimBlock(CUDABLOCKSIZE);
  dim3 dimGrid(Cuda::compute2DGridForFlatBuffer(size, CUDABLOCKSIZE));
  subtractKernel<T><<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), toSubtract.get(), size);
  return CUDA_STATUS;
}

namespace {
template <typename A, typename B, typename Result, bool clamp = false>
__global__ void addRGBKernel(uint32_t *dst, const uint32_t *toAdd, std::size_t size) {
  std::size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    uint32_t vSrc = toAdd[i];
    uint32_t vDst = dst[i];
    int32_t srcIsSolid = !!B::a(vSrc);
    int32_t dstIsSolid = !!A::a(vDst);
    int32_t r = dstIsSolid * (A::r(vDst) + srcIsSolid * B::r(vSrc));
    int32_t g = dstIsSolid * (A::g(vDst) + srcIsSolid * B::g(vSrc));
    int32_t b = dstIsSolid * (A::b(vDst) + srcIsSolid * B::b(vSrc));
    if (clamp) {
      dst[i] = Result::pack(clamp8(r), clamp8(g), clamp8(b), dstIsSolid * 0xff);
    } else {
      dst[i] = Result::pack(r, g, b, dstIsSolid * 0xff);
    }
  }
}

__global__ void addRGB210Kernel(uint32_t *dst, const uint32_t *toAdd0, const uint32_t *toAdd1, std::size_t size) {
  std::size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    uint32_t vSrc = toAdd0[i];
    uint32_t vDst = toAdd1[i];
    int32_t srcIsSolid = !!RGB210::a(vSrc);
    int32_t dstIsSolid = !!RGB210::a(vDst);
    int32_t r = dstIsSolid * (RGB210::r(vDst) + srcIsSolid * RGB210::r(vSrc));
    int32_t g = dstIsSolid * (RGB210::g(vDst) + srcIsSolid * RGB210::g(vSrc));
    int32_t b = dstIsSolid * (RGB210::b(vDst) + srcIsSolid * RGB210::b(vSrc));
    dst[i] = RGB210::pack(r, g, b, dstIsSolid * 0xff);
  }
}

template <typename T>
__global__ void addKernel(T *dst, const T *toAdd, std::size_t size) {
  std::size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    dst[i] += toAdd[i];
  }
}

template <typename T>
__global__ void addKernel(T *dst, const T *toAdd0, const T *toAdd1, std::size_t size) {
  std::size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    dst[i] = toAdd0[i] + toAdd1[i];
  }
}
}  // namespace

template <typename A, typename B, typename Result>
Status add(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toAdd, std::size_t size, GPU::Stream stream) {
  dim3 dimBlock(CUDABLOCKSIZE);
  dim3 dimGrid(Cuda::compute2DGridForFlatBuffer(size, CUDABLOCKSIZE));
  addRGBKernel<A, B, Result><<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), toAdd.get(), size);
  return CUDA_STATUS;
}

template <typename T>
Status addRaw(GPU::Buffer<T> dst, GPU::Buffer<const T> toAdd, std::size_t size, GPU::Stream stream) {
  dim3 dimBlock(CUDABLOCKSIZE);
  dim3 dimGrid(Cuda::compute2DGridForFlatBuffer(size, CUDABLOCKSIZE));
  addKernel<T><<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), toAdd.get(), size);
  return CUDA_STATUS;
}

template <typename T>
Status addRaw(GPU::Buffer<T> dst, GPU::Buffer<const T> toAdd0, GPU::Buffer<const T> toAdd1, std::size_t size,
              GPU::Stream stream) {
  dim3 dimBlock(CUDABLOCKSIZE);
  dim3 dimGrid(Cuda::compute2DGridForFlatBuffer(size, CUDABLOCKSIZE));
  addKernel<T><<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), toAdd0.get(), toAdd1.get(), size);
  return CUDA_STATUS;
}

Status addRGB210(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toAdd0, GPU::Buffer<const uint32_t> toAdd1,
                 std::size_t size, GPU::Stream stream) {
  dim3 dimBlock(CUDABLOCKSIZE);
  dim3 dimGrid(Cuda::compute2DGridForFlatBuffer(size, CUDABLOCKSIZE));
  addRGB210Kernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), toAdd0.get(), toAdd1.get(), size);
  return CUDA_STATUS;
}

Status add10(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toSubtract, std::size_t size, GPU::Stream stream) {
  dim3 dimBlock(CUDABLOCKSIZE);
  dim3 dimGrid(Cuda::compute2DGridForFlatBuffer(size, CUDABLOCKSIZE));
  addRGBKernel<RGB210, RGB210, RGB210><<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), toSubtract.get(), size);
  return CUDA_STATUS;
}

Status add10n8(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toSubtract, std::size_t size,
               GPU::Stream stream) {
  dim3 dimBlock(CUDABLOCKSIZE);
  dim3 dimGrid(Cuda::compute2DGridForFlatBuffer(size, CUDABLOCKSIZE));
  addRGBKernel<RGB210, RGBA, RGB210><<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), toSubtract.get(), size);
  return CUDA_STATUS;
}

Status add10n8Clamp(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toSubtract, std::size_t size,
                    GPU::Stream stream) {
  dim3 dimBlock(CUDABLOCKSIZE);
  dim3 dimGrid(Cuda::compute2DGridForFlatBuffer(size, CUDABLOCKSIZE));
  addRGBKernel<RGB210, RGBA, RGBA, true><<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), toSubtract.get(), size);
  return CUDA_STATUS;
}

Status addClamp(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> toSubtract, std::size_t size,
                GPU::Stream stream) {
  dim3 dimBlock(CUDABLOCKSIZE);
  dim3 dimGrid(Cuda::compute2DGridForFlatBuffer(size, CUDABLOCKSIZE));
  addRGBKernel<RGB210, RGB210, RGBA, true><<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), toSubtract.get(), size);
  return CUDA_STATUS;
}

namespace {
template <typename T>
__global__ void andOperatorKernel(T *dst, const T *toAnd, std::size_t size) {
  std::size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    dst[i] = dst[i] & toAnd[i];
  }
}

template <typename T>
__global__ void mulOperatorKernel(T *dst, const T toMul, std::size_t size) {
  std::size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    dst[i] = dst[i] * toMul;
  }
}
}  // namespace

template <typename T>
Status andOperatorRaw(GPU::Buffer<T> dst, GPU::Buffer<const T> toAnd, std::size_t size, GPU::Stream stream) {
  dim3 dimBlock(CUDABLOCKSIZE);
  dim3 dimGrid(Cuda::compute2DGridForFlatBuffer(size, CUDABLOCKSIZE));
  andOperatorKernel<T><<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), toAnd.get(), size);
  return CUDA_STATUS;
}

template Status andOperatorRaw(GPU::Buffer<unsigned char> dst, GPU::Buffer<const unsigned char> toAnd, std::size_t size,
                               GPU::Stream stream);

template <typename T>
Status mulOperatorRaw(GPU::Buffer<T> dst, const T toMul, std::size_t size, GPU::Stream stream) {
  dim3 dimBlock(CUDABLOCKSIZE);
  dim3 dimGrid(Cuda::compute2DGridForFlatBuffer(size, CUDABLOCKSIZE));
  mulOperatorKernel<T><<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), toMul, size);
  return CUDA_STATUS;
}

template Status mulOperatorRaw(GPU::Buffer<float2> dst, const float2 toMul, std::size_t size, GPU::Stream stream);

#include "backend/common/image/imageOps.inst"
}  // namespace Image
}  // namespace VideoStitch
