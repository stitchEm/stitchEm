// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/processors/photoCorr.hpp"

#include "../surface.hpp"
#include "../deviceStream.hpp"

#include "backend/common/imageOps.hpp"
#include "cuda/util.hpp"
#include "core/kernels/photoStack.cu"

namespace VideoStitch {
namespace Core {

template <class PhotoCorrection>
__global__ void preStitchPhotoCorrectionKernel(cudaSurfaceObject_t buffer, const int width, const int height,
                                               const float rMult, const float gMult, const float bMult,
                                               const float vigCenterX, const float vigCenterY,
                                               const float inverseDemiDiagonalSquared, const float vigCoeff0,
                                               const float vigCoeff1, const float vigCoeff2, const float vigCoeff3,
                                               const TransformPhotoParam photoParam) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const float fx = (float)x - (float)width / 2.0f + 0.5f;  // FIXME: do we need to apply center shift here ?
    const float fy = (float)y - (float)height / 2.0f + 0.5f;
    /**
     * Compute vignetting:
     * Done before uv is shifted for texture fetch.
     * vigMult = a0 + a1 * r + a2 * r^2 + a3 * r^3
     *         = a0 + r * (a1 + r * (a2 + r * a3))
     */
    const float vigRadiusSquared =
        ((fx - vigCenterX) * (fx - vigCenterX) + (fy - vigCenterY) * (fy - vigCenterY)) * inverseDemiDiagonalSquared;
    float vigMult = vigRadiusSquared * vigCoeff3;
    vigMult += vigCoeff2;
    vigMult *= vigRadiusSquared;
    vigMult += vigCoeff1;
    vigMult *= vigRadiusSquared;
    vigMult += vigCoeff0;
    vigMult = 1.0f / vigMult;

    uint32_t v;
    surf2Dread(&v, buffer, x * sizeof(uint32_t), y);
    float3 color = make_float3(Image::RGBA::r(v), Image::RGBA::g(v), Image::RGBA::b(v));

    // exposure correction
    color = PhotoCorrection::corr(color, photoParam.floatParam, (float*)photoParam.transformData);
    color.x *= rMult * vigMult;
    color.y *= gMult * vigMult;
    color.z *= bMult * vigMult;
    color = PhotoCorrection::invCorr(color, photoParam.floatParam, (float*)photoParam.transformData);

    uint32_t pixel = Image::RGBA::pack(Image::clamp8(__float2int_rn(color.x)), Image::clamp8(__float2int_rn(color.y)),
                                       Image::clamp8(__float2int_rn(color.z)), Image::RGBA::a(v));
    surf2Dwrite(pixel, buffer, x * sizeof(uint32_t), y);
  }
}

#define RUN_KERNEL(PhotoCorrection)                                                                                   \
  preStitchPhotoCorrectionKernel<PhotoCorrection><<<dimGrid, dimBlock, 0, stream.get()>>>(                            \
      buffer.get().surface(), width, height, rMult, gMult, bMult, vigCenterX, vigCenterY, inverseDemiDiagonalSquared, \
      vigCoeff0, vigCoeff1, vigCoeff2, vigCoeff3, photoParam);

Status linearPhotoCorrection(GPU::Surface& buffer, const int width, const int height, const float rMult,
                             const float gMult, const float bMult, const float vigCenterX, const float vigCenterY,
                             const float inverseDemiDiagonalSquared, const float vigCoeff0, const float vigCoeff1,
                             const float vigCoeff2, const float vigCoeff3, const TransformPhotoParam& photoParam,
                             GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  RUN_KERNEL(LinearPhotoCorrection);
  return CUDA_STATUS;
}

Status gammaPhotoCorrection(GPU::Surface& buffer, const int width, const int height, const float rMult,
                            const float gMult, const float bMult, const float vigCenterX, const float vigCenterY,
                            const float inverseDemiDiagonalSquared, const float vigCoeff0, const float vigCoeff1,
                            const float vigCoeff2, const float vigCoeff3, const TransformPhotoParam& photoParam,
                            GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  RUN_KERNEL(GammaPhotoCorrection);
  return CUDA_STATUS;
}

Status emorPhotoCorrection(GPU::Surface& buffer, const int width, const int height, const float rMult,
                           const float gMult, const float bMult, const float vigCenterX, const float vigCenterY,
                           const float inverseDemiDiagonalSquared, const float vigCoeff0, const float vigCoeff1,
                           const float vigCoeff2, const float vigCoeff3, const TransformPhotoParam& photoParam,
                           GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);
  RUN_KERNEL(EmorPhotoCorrection);
  return CUDA_STATUS;
}
#undef RUN_KERNEL
}  // namespace Core
}  // namespace VideoStitch
