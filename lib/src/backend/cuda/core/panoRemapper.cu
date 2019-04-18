// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "core1/panoRemapper.hpp"

#include "core/rect.hpp"
#include "core/transformGeoParams.hpp"

#include "backend/common/imageOps.hpp"

#include "backend/cuda/deviceBuffer.hpp"
#include "backend/cuda/deviceStream.hpp"
#include "backend/cuda/surface.hpp"

#include "cuda/error.hpp"
#include "cuda/util.hpp"

#include "gpu/core1/transform.hpp"
#include "gpu/buffer.hpp"
#include "gpu/memcpy.hpp"
#include "gpu/allocator.hpp"

#include "libvideostitch/panoDef.hpp"

#include "backend/cuda/core/transformStack.cu"
#include "core/kernels/withinStack.cu"

namespace VideoStitch {
namespace Core {

template <Convert2D3DFnT toSphere, class OutputCropper>
__global__ void remapKernel(uint32_t* g_odata, cudaTextureObject_t remapTex, int panoWidth, int panoHeight,
                            const float2 inPanoScale, const float2 outPanoScale, const vsfloat3x3 R) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < panoWidth && y < panoHeight) {
    if (OutputCropper::isPanoPointVisible(x, y, panoWidth, panoHeight)) {
      float2 uv = make_float2((float)x, (float)y);

      /**
       * The transformations are applied relative to the center of the panorama image
       */
      uv.x -= (panoWidth - 1) / 2.0f;
      uv.y -= (panoHeight - 1) / 2.0f;

      /**
       * Apply transform stack
       */
      uv.x /= outPanoScale.x;
      uv.y /= outPanoScale.y;

      float3 pt = toSphere(uv);

      pt = rotateSphere(pt, R);

      uv = SphereToErect(pt);

      uv.x *= inPanoScale.x;
      uv.y *= inPanoScale.y;

      /**
       * See notes in warp kernel
       * compensate fetching offset with cudaFilterModeLinear by adding 0.5f
       * https://stackoverflow.com/questions/10643790/texture-memory-tex2d-basics
       */
      uv.x += panoWidth / 2.0f;
      uv.y += panoHeight / 2.0f;

      float4 px = tex2D<float4>(remapTex, uv.x, uv.y);
      g_odata[y * panoWidth + x] = Image::RGBA::pack(__float2uint_rn(px.x * 255.), __float2uint_rn(px.y * 255.),
                                                     __float2uint_rn(px.z * 255.), __float2uint_rn(px.w * 255.));
    } else {
      g_odata[y * panoWidth + x] = 0;
    }
  }
}

__global__ void remapCubemapKernel(uint32_t* __restrict__ xPositive, uint32_t* __restrict__ xNegative,
                                   uint32_t* __restrict__ yPositive, uint32_t* __restrict__ yNegative,
                                   uint32_t* __restrict__ zPositive, uint32_t* __restrict__ zNegative, int panoWidth,
                                   int panoHeight, cudaTextureObject_t remapTex, int faceDim, const float2 panoScale,
                                   const vsfloat3x3 R) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < faceDim && y < faceDim) {
    /* compensate fetching offset with cudaFilterModeLinear by adding 0.5f */
    float2 uv = make_float2(x + 0.5f, y + 0.5f);
    uv = (uv / faceDim) * 2.f - make_float2(1.f, 1.f);

    float3 pt;
    for (unsigned int face = 0; face < 6; face++) {
      // Layer 0 is positive X face
      if (face == 0) {
        pt.x = 1;
        pt.y = -uv.y;
        pt.z = -uv.x;
      }
      // Layer 1 is negative X face
      else if (face == 1) {
        pt.x = -1;
        pt.y = -uv.y;
        pt.z = uv.x;
      }
      // Layer 2 is positive Y face
      else if (face == 2) {
        pt.x = uv.x;
        pt.y = 1;
        pt.z = uv.y;
      }
      // Layer 3 is negative Y face
      else if (face == 3) {
        pt.x = uv.x;
        pt.y = -1;
        pt.z = -uv.y;
      }
      // Layer 4 is positive Z face
      else if (face == 4) {
        pt.x = uv.x;
        pt.y = -uv.y;
        pt.z = 1;
      }
      // Layer 5 is negative Z face
      else if (face == 5) {
        pt.x = -uv.x;
        pt.y = -uv.y;
        pt.z = -1;
      }

      pt = rotateSphere(pt, R);

      float2 xy = SphereToErect(pt);

      xy *= panoScale;

      /**
       * See notes in warp kernel
       */
      xy.x += panoWidth / 2.0f;
      xy.y += panoHeight / 2.0f;

      float4 px = tex2D<float4>(remapTex, xy.x, xy.y);
      if (face == 0) {
        xPositive[y * faceDim + x] = Image::RGBA::pack(__float2uint_rn(px.x * 255.), __float2uint_rn(px.y * 255.),
                                                       __float2uint_rn(px.z * 255.), __float2uint_rn(px.w * 255.));
      } else if (face == 1) {
        xNegative[y * faceDim + x] = Image::RGBA::pack(__float2uint_rn(px.x * 255.), __float2uint_rn(px.y * 255.),
                                                       __float2uint_rn(px.z * 255.), __float2uint_rn(px.w * 255.));
      } else if (face == 2) {
        yPositive[y * faceDim + x] = Image::RGBA::pack(__float2uint_rn(px.x * 255.), __float2uint_rn(px.y * 255.),
                                                       __float2uint_rn(px.z * 255.), __float2uint_rn(px.w * 255.));
      } else if (face == 3) {
        yNegative[y * faceDim + x] = Image::RGBA::pack(__float2uint_rn(px.x * 255.), __float2uint_rn(px.y * 255.),
                                                       __float2uint_rn(px.z * 255.), __float2uint_rn(px.w * 255.));
      } else if (face == 4) {
        zPositive[y * faceDim + x] = Image::RGBA::pack(__float2uint_rn(px.x * 255.), __float2uint_rn(px.y * 255.),
                                                       __float2uint_rn(px.z * 255.), __float2uint_rn(px.w * 255.));
      } else if (face == 5) {
        zNegative[y * faceDim + x] = Image::RGBA::pack(__float2uint_rn(px.x * 255.), __float2uint_rn(px.y * 255.),
                                                       __float2uint_rn(px.z * 255.), __float2uint_rn(px.w * 255.));
      }
    }
  }
}

template <bool equiangular>
__global__ void rotateCubemapKernel(uint32_t* __restrict__ xPositive, uint32_t* __restrict__ xNegative,
                                    uint32_t* __restrict__ yPositive, uint32_t* __restrict__ yNegative,
                                    uint32_t* __restrict__ zPositive, uint32_t* __restrict__ zNegative, int faceDim,
                                    cudaTextureObject_t remapTex, const vsfloat3x3 R) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < faceDim && y < faceDim) {
    float2 uv = make_float2((float)x, (float)y);
    uv = (uv / faceDim) * 2.f - make_float2(1.f, 1.f);

    if (equiangular) {
      uv.x = tanf_vs(uv.x * PI_F_VS / 4.);
      uv.y = tanf_vs(uv.y * PI_F_VS / 4.);
    }

    float3 pt;
    for (unsigned int face = 0; face < 6; face++) {
      // Layer 0 is positive X face
      if (face == 0) {
        pt.x = 1;
        pt.y = -uv.y;
        pt.z = -uv.x;
      }
      // Layer 1 is negative X face
      else if (face == 1) {
        pt.x = -1;
        pt.y = -uv.y;
        pt.z = uv.x;
      }
      // Layer 2 is positive Y face
      else if (face == 2) {
        pt.x = uv.x;
        pt.y = 1;
        pt.z = uv.y;
      }
      // Layer 3 is negative Y face
      else if (face == 3) {
        pt.x = uv.x;
        pt.y = -1;
        pt.z = -uv.y;
      }
      // Layer 4 is positive Z face
      else if (face == 4) {
        pt.x = uv.x;
        pt.y = -uv.y;
        pt.z = 1;
      }
      // Layer 5 is negative Z face
      else if (face == 5) {
        pt.x = -uv.x;
        pt.y = -uv.y;
        pt.z = -1;
      }

      pt = rotateSphere(pt, R);

      if (equiangular) {
        // first normalize with Chebyshev distance to project back on the cube
        float cheb = fmaxf(abs(pt.x), abs(pt.y));
        cheb = fmaxf(cheb, abs(pt.z));
        pt /= cheb;
        // then reinflate the cube
        pt.x = 4. / PI_F_VS * atanf_vs(pt.x);
        pt.y = 4. / PI_F_VS * atanf_vs(pt.y);
        pt.z = 4. / PI_F_VS * atanf_vs(pt.z);
      }

      float4 px = texCubemap<float4>(remapTex, pt.x, pt.y, pt.z);
      if (face == 0) {
        xPositive[y * faceDim + x] = Image::RGBA::pack(__float2uint_rn(px.x * 255.), __float2uint_rn(px.y * 255.),
                                                       __float2uint_rn(px.z * 255.), __float2uint_rn(px.w * 255.));
      } else if (face == 1) {
        xNegative[y * faceDim + x] = Image::RGBA::pack(__float2uint_rn(px.x * 255.), __float2uint_rn(px.y * 255.),
                                                       __float2uint_rn(px.z * 255.), __float2uint_rn(px.w * 255.));
      } else if (face == 2) {
        yPositive[y * faceDim + x] = Image::RGBA::pack(__float2uint_rn(px.x * 255.), __float2uint_rn(px.y * 255.),
                                                       __float2uint_rn(px.z * 255.), __float2uint_rn(px.w * 255.));
      } else if (face == 3) {
        yNegative[y * faceDim + x] = Image::RGBA::pack(__float2uint_rn(px.x * 255.), __float2uint_rn(px.y * 255.),
                                                       __float2uint_rn(px.z * 255.), __float2uint_rn(px.w * 255.));
      } else if (face == 4) {
        zPositive[y * faceDim + x] = Image::RGBA::pack(__float2uint_rn(px.x * 255.), __float2uint_rn(px.y * 255.),
                                                       __float2uint_rn(px.z * 255.), __float2uint_rn(px.w * 255.));
      } else if (face == 5) {
        zNegative[y * faceDim + x] = Image::RGBA::pack(__float2uint_rn(px.x * 255.), __float2uint_rn(px.y * 255.),
                                                       __float2uint_rn(px.z * 255.), __float2uint_rn(px.w * 255.));
      }
    }
  }
}

Status rotateCubemap(const PanoDefinition& pano, GPU::CubemapSurface& cubemapSurface, GPU::Buffer<uint32_t> xPosPbo,
                     GPU::Buffer<uint32_t> xNegPbo, GPU::Buffer<uint32_t> yPosPbo, GPU::Buffer<uint32_t> yNegPbo,
                     GPU::Buffer<uint32_t> zPosPbo, GPU::Buffer<uint32_t> zNegPbo, const Matrix33<double>& perspective,
                     bool equiangular, GPU::Stream stream) {
  vsfloat3x3 rotation;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      rotation.values[i][j] = (float)perspective(i, j);
    }
  }

  dim3 block(16, 16, 1);
  dim3 grid((unsigned)Cuda::ceilDiv(pano.getLength(), block.x), (unsigned)Cuda::ceilDiv(pano.getLength(), block.y), 1);

  if (equiangular) {
    rotateCubemapKernel<true><<<grid, block, 0, stream.get()>>>(
        xPosPbo.get(), xNegPbo.get(), yPosPbo.get(), yNegPbo.get(), zPosPbo.get(), zNegPbo.get(), (int)pano.getLength(),
        cubemapSurface.get().texture(), rotation);
  } else {
    rotateCubemapKernel<false><<<grid, block, 0, stream.get()>>>(
        xPosPbo.get(), xNegPbo.get(), yPosPbo.get(), yNegPbo.get(), zPosPbo.get(), zNegPbo.get(), (int)pano.getLength(),
        cubemapSurface.get().texture(), rotation);
  }

  return CUDA_STATUS;
}

__device__ float3 positiveX(float2& uv) {
  float3 pt;
  pt.x = 1;
  pt.y = -uv.y;
  pt.z = uv.x;
  return pt;
}
__device__ float3 negativeX(float2& uv) {
  float3 pt;
  pt.x = -1;
  pt.y = -uv.y;
  pt.z = -uv.x;
  return pt;
}
__device__ float3 positiveY(float2& uv) {
  float3 pt;
  pt.x = uv.x;
  pt.y = 1;
  pt.z = -uv.y;
  return pt;
}
__device__ float3 negativeY(float2& uv) {
  float3 pt;
  pt.x = uv.x;
  pt.y = -1;
  pt.z = uv.y;
  return pt;
}
__device__ float3 positiveZ(float2& uv) {
  float3 pt;
  pt.x = uv.x;
  pt.y = -uv.y;
  pt.z = -1;
  return pt;
}
__device__ float3 negativeZ(float2& uv) {
  float3 pt;
  pt.x = -uv.x;
  pt.y = -uv.y;
  pt.z = 1;
  return pt;
}

template <float3 (*project)(float2&), bool equiangular>
__global__ void remapMaskFace(unsigned char* __restrict__ face, int dstOffsetX, int dstOffsetY, int bbWidth,
                              int bbHeight, int panoWidth, int panoHeight, cudaTextureObject_t remapTex, int srcOffsetX,
                              int srcOffsetY, int faceDim, const float2 panoScale) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < bbWidth && y < bbHeight) {
    /* compensate fetching offset with cudaFilterModeLinear by adding 0.5f */
    float2 uv = make_float2(x + dstOffsetX + 0.5f, y + dstOffsetY + 0.5f);
    uv = (uv / faceDim) * 2.f - make_float2(1.f, 1.f);

    if (equiangular) {
      uv.x = tanf_vs(uv.x * PI_F_VS / 4.);
      uv.y = tanf_vs(uv.y * PI_F_VS / 4.);
    }

    float3 pt = project(uv);

    float2 xy = SphereToErect(pt);

    xy *= panoScale;

    /**
     * See notes in warp kernel
     */
    xy.x += panoWidth / 2.0f;
    xy.y += panoHeight / 2.0f;

    xy.x -= srcOffsetX;
    xy.y -= srcOffsetY;
    if (xy.x < 0.) {
      xy.x += panoWidth;
    }

    float px = tex2D<float>(remapTex, xy.x, xy.y);
    face[y * bbWidth + x] = __float2uint_rn(px * 255.);
  }
}

Status reprojectAlphaToCubemap(int panoWidth, int panoHeight, int faceLength, GPU::Surface& alphaSurface,
                               Rect equirectBB, GPU::Buffer<unsigned char> xPosAlpha, Rect xPosBB,
                               GPU::Buffer<unsigned char> xNegAlpha, Rect xNegBB, GPU::Buffer<unsigned char> yPosAlpha,
                               Rect yPosBB, GPU::Buffer<unsigned char> yNegAlpha, Rect yNegBB,
                               GPU::Buffer<unsigned char> zPosAlpha, Rect zPosBB, GPU::Buffer<unsigned char> zNegAlpha,
                               Rect zNegBB, bool equiangular, GPU::Stream stream) {
  dim3 block(16, 16, 1);
  float2 panoScale = {TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, panoWidth, 360.f),
                      2 * TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, panoHeight, 360.f)};

  if (!xPosBB.empty()) {
    dim3 gridXPos((unsigned)Cuda::ceilDiv(xPosBB.getWidth(), block.x),
                  (unsigned)Cuda::ceilDiv(xPosBB.getHeight(), block.y), 1);
    if (equiangular) {
      remapMaskFace<positiveX, true><<<gridXPos, block, 0, stream.get()>>>(
          xPosAlpha.get().raw(), (unsigned)xPosBB.left(), (unsigned)xPosBB.top(), (unsigned)xPosBB.getWidth(),
          (unsigned)xPosBB.getHeight(), (unsigned)panoWidth, (unsigned)panoHeight, alphaSurface.get().texture(),
          (unsigned)equirectBB.left(), (unsigned)equirectBB.top(), (unsigned)faceLength, panoScale);
    } else {
      remapMaskFace<positiveX, false><<<gridXPos, block, 0, stream.get()>>>(
          xPosAlpha.get().raw(), (unsigned)xPosBB.left(), (unsigned)xPosBB.top(), (unsigned)xPosBB.getWidth(),
          (unsigned)xPosBB.getHeight(), (unsigned)panoWidth, (unsigned)panoHeight, alphaSurface.get().texture(),
          (unsigned)equirectBB.left(), (unsigned)equirectBB.top(), (unsigned)faceLength, panoScale);
    }
  }
  if (!xNegBB.empty()) {
    dim3 gridXNeg((unsigned)Cuda::ceilDiv(xNegBB.getWidth(), block.x),
                  (unsigned)Cuda::ceilDiv(xNegBB.getHeight(), block.y), 1);
    if (equiangular) {
      remapMaskFace<negativeX, true><<<gridXNeg, block, 0, stream.get()>>>(
          xNegAlpha.get().raw(), (unsigned)xNegBB.left(), (unsigned)xNegBB.top(), (unsigned)xNegBB.getWidth(),
          (unsigned)xNegBB.getHeight(), (unsigned)panoWidth, (unsigned)panoHeight, alphaSurface.get().texture(),
          (unsigned)equirectBB.left(), (unsigned)equirectBB.top(), (unsigned)faceLength, panoScale);
    } else {
      remapMaskFace<negativeX, false><<<gridXNeg, block, 0, stream.get()>>>(
          xNegAlpha.get().raw(), (unsigned)xNegBB.left(), (unsigned)xNegBB.top(), (unsigned)xNegBB.getWidth(),
          (unsigned)xNegBB.getHeight(), (unsigned)panoWidth, (unsigned)panoHeight, alphaSurface.get().texture(),
          (unsigned)equirectBB.left(), (unsigned)equirectBB.top(), (unsigned)faceLength, panoScale);
    }
  }
  if (!yPosBB.empty()) {
    dim3 gridYPos((unsigned)Cuda::ceilDiv(yPosBB.getWidth(), block.x),
                  (unsigned)Cuda::ceilDiv(yPosBB.getHeight(), block.y), 1);
    if (equiangular) {
      remapMaskFace<positiveY, true><<<gridYPos, block, 0, stream.get()>>>(
          yPosAlpha.get().raw(), (unsigned)yPosBB.left(), (unsigned)yPosBB.top(), (unsigned)yPosBB.getWidth(),
          (unsigned)yPosBB.getHeight(), (unsigned)panoWidth, (unsigned)panoHeight, alphaSurface.get().texture(),
          (unsigned)equirectBB.left(), (unsigned)equirectBB.top(), (unsigned)faceLength, panoScale);
    } else {
      remapMaskFace<positiveY, false><<<gridYPos, block, 0, stream.get()>>>(
          yPosAlpha.get().raw(), (unsigned)yPosBB.left(), (unsigned)yPosBB.top(), (unsigned)yPosBB.getWidth(),
          (unsigned)yPosBB.getHeight(), (unsigned)panoWidth, (unsigned)panoHeight, alphaSurface.get().texture(),
          (unsigned)equirectBB.left(), (unsigned)equirectBB.top(), (unsigned)faceLength, panoScale);
    }
  }
  if (!yNegBB.empty()) {
    dim3 gridYNeg((unsigned)Cuda::ceilDiv(yNegBB.getWidth(), block.x),
                  (unsigned)Cuda::ceilDiv(yNegBB.getHeight(), block.y), 1);
    if (equiangular) {
      remapMaskFace<negativeY, true><<<gridYNeg, block, 0, stream.get()>>>(
          yNegAlpha.get().raw(), (unsigned)yNegBB.left(), (unsigned)yNegBB.top(), (unsigned)yNegBB.getWidth(),
          (unsigned)yNegBB.getHeight(), (unsigned)panoWidth, (unsigned)panoHeight, alphaSurface.get().texture(),
          (unsigned)equirectBB.left(), (unsigned)equirectBB.top(), (unsigned)faceLength, panoScale);
    } else {
      remapMaskFace<negativeY, false><<<gridYNeg, block, 0, stream.get()>>>(
          yNegAlpha.get().raw(), (unsigned)yNegBB.left(), (unsigned)yNegBB.top(), (unsigned)yNegBB.getWidth(),
          (unsigned)yNegBB.getHeight(), (unsigned)panoWidth, (unsigned)panoHeight, alphaSurface.get().texture(),
          (unsigned)equirectBB.left(), (unsigned)equirectBB.top(), (unsigned)faceLength, panoScale);
    }
  }
  if (!zPosBB.empty()) {
    dim3 gridZPos((unsigned)Cuda::ceilDiv(zPosBB.getWidth(), block.x),
                  (unsigned)Cuda::ceilDiv(zPosBB.getHeight(), block.y), 1);
    if (equiangular) {
      remapMaskFace<positiveZ, true><<<gridZPos, block, 0, stream.get()>>>(
          zPosAlpha.get().raw(), (unsigned)zPosBB.left(), (unsigned)zPosBB.top(), (unsigned)zPosBB.getWidth(),
          (unsigned)zPosBB.getHeight(), (unsigned)panoWidth, (unsigned)panoHeight, alphaSurface.get().texture(),
          (unsigned)equirectBB.left(), (unsigned)equirectBB.top(), (unsigned)faceLength, panoScale);
    } else {
      remapMaskFace<positiveZ, false><<<gridZPos, block, 0, stream.get()>>>(
          zPosAlpha.get().raw(), (unsigned)zPosBB.left(), (unsigned)zPosBB.top(), (unsigned)zPosBB.getWidth(),
          (unsigned)zPosBB.getHeight(), (unsigned)panoWidth, (unsigned)panoHeight, alphaSurface.get().texture(),
          (unsigned)equirectBB.left(), (unsigned)equirectBB.top(), (unsigned)faceLength, panoScale);
    }
  }
  if (!zNegBB.empty()) {
    dim3 gridZNeg((unsigned)Cuda::ceilDiv(zNegBB.getWidth(), block.x),
                  (unsigned)Cuda::ceilDiv(zNegBB.getHeight(), block.y), 1);
    if (equiangular) {
      remapMaskFace<negativeZ, true><<<gridZNeg, block, 0, stream.get()>>>(
          zNegAlpha.get().raw(), (unsigned)zNegBB.left(), (unsigned)zNegBB.top(), (unsigned)zNegBB.getWidth(),
          (unsigned)zNegBB.getHeight(), (unsigned)panoWidth, (unsigned)panoHeight, alphaSurface.get().texture(),
          (unsigned)equirectBB.left(), (unsigned)equirectBB.top(), (unsigned)faceLength, panoScale);
    } else {
      remapMaskFace<negativeZ, false><<<gridZNeg, block, 0, stream.get()>>>(
          zNegAlpha.get().raw(), (unsigned)zNegBB.left(), (unsigned)zNegBB.top(), (unsigned)zNegBB.getWidth(),
          (unsigned)zNegBB.getHeight(), (unsigned)panoWidth, (unsigned)panoHeight, alphaSurface.get().texture(),
          (unsigned)equirectBB.left(), (unsigned)equirectBB.top(), (unsigned)faceLength, panoScale);
    }
  }
  return CUDA_STATUS;
}

template <Convert2D3DFnT toSphere, class OutputCropper>
Status reprojectPanorama(GPU::Buffer<uint32_t> pbo, float2 dstScale, GPU::Surface& tex, float2 srcScale, unsigned width,
                         unsigned height, const Matrix33<double>& perspective, GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);

  vsfloat3x3 rotation;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      rotation.values[i][j] = (float)perspective(i, j);
    }
  }

  remapKernel<toSphere, OutputCropper><<<dimGrid, dimBlock, 0, stream.get()>>>(pbo.get(), tex.get().texture(), width,
                                                                               height, srcScale, dstScale, rotation);

  return CUDA_STATUS;
}

Status reprojectRectilinear(GPU::Buffer<uint32_t> pbo, float2 outScale, GPU::Surface& tex, float2 inScale,
                            unsigned width, unsigned height, const Matrix33<double>& perspective, GPU::Stream stream) {
  return reprojectPanorama<RectToSphere, OutputRectCropper>(pbo, outScale, tex, inScale, width, height, perspective,
                                                            stream);
}
Status reprojectEquirectangular(GPU::Buffer<uint32_t> pbo, float2 outScale, GPU::Surface& tex, float2 inScale,
                                unsigned width, unsigned height, const Matrix33<double>& perspective,
                                GPU::Stream stream) {
  return reprojectPanorama<ErectToSphere, OutputRectCropper>(pbo, outScale, tex, inScale, width, height, perspective,
                                                             stream);
}
Status reprojectFullFrameFisheye(GPU::Buffer<uint32_t> pbo, float2 outScale, GPU::Surface& tex, float2 inScale,
                                 unsigned width, unsigned height, const Matrix33<double>& perspective,
                                 GPU::Stream stream) {
  return reprojectPanorama<FisheyeToSphere, OutputRectCropper>(pbo, outScale, tex, inScale, width, height, perspective,
                                                               stream);
}
Status reprojectCircularFisheye(GPU::Buffer<uint32_t> pbo, float2 outScale, GPU::Surface& tex, float2 inScale,
                                unsigned width, unsigned height, const Matrix33<double>& perspective,
                                GPU::Stream stream) {
  return reprojectPanorama<FisheyeToSphere, OutputCircleCropper>(pbo, outScale, tex, inScale, width, height,
                                                                 perspective, stream);
}
Status reprojectStereographic(GPU::Buffer<uint32_t> pbo, float2 outScale, GPU::Surface& tex, float2 inScale,
                              unsigned width, unsigned height, const Matrix33<double>& perspective,
                              GPU::Stream stream) {
  return reprojectPanorama<StereoToSphere, OutputRectCropper>(pbo, outScale, tex, inScale, width, height, perspective,
                                                              stream);
}

}  // namespace Core
}  // namespace VideoStitch
