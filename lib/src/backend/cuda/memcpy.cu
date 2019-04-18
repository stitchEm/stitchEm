// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/memcpy.hpp"

#include "deviceBuffer.hpp"
#include "deviceStream.hpp"
#include "surface.hpp"

#include "backend/common/imageOps.hpp"

#include "cuda/util.hpp"

namespace VideoStitch {
namespace GPU {

__global__ void copyCubemapFace(uint32_t* __restrict__ src, int faceDim, int face, cudaSurfaceObject_t surf) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < faceDim && y < faceDim) {
    uint32_t val = src[y * faceDim + x];
    uchar4 pix = make_uchar4(Image::RGBA::r(val), Image::RGBA::g(val), Image::RGBA::b(val), Image::RGBA::a(val));

    surfCubemapwrite(pix, surf, (int)x * sizeof(uchar4), (int)y, face);
  }
}

Status memcpyCubemapAsync(CubemapSurface& cubemapSurface, Buffer<uint32_t> xPosPbo, Buffer<uint32_t> xNegPbo,
                          Buffer<uint32_t> yPosPbo, Buffer<uint32_t> yNegPbo, Buffer<uint32_t> zPosPbo,
                          Buffer<uint32_t> zNegPbo, size_t faceDim, const Stream& stream) {
  dim3 block(16, 16, 1);
  dim3 grid((unsigned)Cuda::ceilDiv(faceDim, block.x), (unsigned)Cuda::ceilDiv(faceDim, block.y), 1);

  copyCubemapFace<<<grid, block, 0, stream.get()>>>(xPosPbo.get(), (int)faceDim, 0, cubemapSurface.get().surface());
  copyCubemapFace<<<grid, block, 0, stream.get()>>>(xNegPbo.get(), (int)faceDim, 1, cubemapSurface.get().surface());
  copyCubemapFace<<<grid, block, 0, stream.get()>>>(yPosPbo.get(), (int)faceDim, 2, cubemapSurface.get().surface());
  copyCubemapFace<<<grid, block, 0, stream.get()>>>(yNegPbo.get(), (int)faceDim, 3, cubemapSurface.get().surface());
  copyCubemapFace<<<grid, block, 0, stream.get()>>>(zPosPbo.get(), (int)faceDim, 4, cubemapSurface.get().surface());
  copyCubemapFace<<<grid, block, 0, stream.get()>>>(zNegPbo.get(), (int)faceDim, 5, cubemapSurface.get().surface());

  return CUDA_STATUS;
}

__global__ void resetArrayKernel(cudaSurfaceObject_t dst, size_t width, size_t height) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    surf2Dwrite(0, dst, x * sizeof(uint32_t), y);
  }
}

Status memsetToZeroAsync(Surface& dst, const Stream& stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(dst.width(), 16), (unsigned)Cuda::ceilDiv(dst.height(), 16), 1);

  resetArrayKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get().surface(), dst.width(), dst.height());
  return CUDA_STATUS;
}

}  // namespace GPU
}  // namespace VideoStitch
