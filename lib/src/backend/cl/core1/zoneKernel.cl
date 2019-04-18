// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "../gpuKernelDef.h"

#include "../image/imageFormat.h"

static inline bool OutputRectCropper_isPanoPointVisible(int x, int y, int panoWidth, int panoHeight) { return true; }

static inline bool OutputCircleCropper_isPanoPointVisible(int x, int y, int panoWidth, int panoHeight) { return true; }

static inline bool isWithinCropRect(const float2 uv, float width, float height, float cLeft, float cRight, float cTop,
                                    float cBottom) {
  return 0.0f <= uv.x && uv.x < width && 0.0f <= uv.y && uv.y < height && cLeft <= uv.x && uv.x <= cRight &&
         cTop <= uv.y && uv.y <= cBottom;
}

static inline bool isWithinCropCircle(const float2 uv, float width, float height, float cLeft, float cRight, float cTop,
                                      float cBottom) {
  const float centerX = (cRight + cLeft) / 2.0f;
  const float centerY = (cBottom + cTop) / 2.0f;
  const float radius = fmin(cRight - cLeft, cBottom - cTop) / 2.0f;
  return 0.0f <= uv.x && uv.x < width && 0.0f <= uv.y && uv.y < height &&
         (uv.x - centerX) * (uv.x - centerX) + (uv.y - centerY) * (uv.y - centerY) < radius * radius;
}

#include "backend/common/core/types.hpp"

#include "mapFunction.h"

#include "backend/common/core1/zoneKernel.gpu"

static const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

#define OUTPUTPROJECTION ErectToSphere
#define OUTPUTCROPPER OutputRectCropper_isPanoPointVisible
#define REMAPKERNEL remap_equirectangular
#include "remapKernel.cl.incl"
#undef REMAPKERNEL
#undef OUTPUTCROPPER
#undef OUTPUTPROJECTION

#define OUTPUTPROJECTION RectToSphere
#define OUTPUTCROPPER OutputRectCropper_isPanoPointVisible
#define REMAPKERNEL remap_rectilinear
#include "remapKernel.cl.incl"
#undef REMAPKERNEL
#undef OUTPUTCROPPER
#undef OUTPUTPROJECTION

#define OUTPUTPROJECTION FisheyeToSphere
#define OUTPUTCROPPER OutputRectCropper_isPanoPointVisible
#define REMAPKERNEL remap_fullframe_fisheye
#include "remapKernel.cl.incl"
#undef REMAPKERNEL
#undef OUTPUTCROPPER
#undef OUTPUTPROJECTION

#define OUTPUTPROJECTION FisheyeToSphere
#define OUTPUTCROPPER OutputCircleCropper_isPanoPointVisible
#define REMAPKERNEL remap_circular_fisheye
#include "remapKernel.cl.incl"
#undef REMAPKERNEL
#undef OUTPUTCROPPER
#undef OUTPUTPROJECTION

#define OUTPUTPROJECTION StereoToSphere
#define OUTPUTCROPPER OutputRectCropper_isPanoPointVisible
#define REMAPKERNEL remap_stereographic
#include "remapKernel.cl.incl"
#undef REMAPKERNEL
#undef OUTPUTCROPPER
#undef OUTPUTPROJECTION

kernel void remap_cubemap(global uint32_t* g_odata, read_only image2d_t tex, int panoWidth, int panoHeight,
                          const float2 panoScale, const vsfloat3x3 R, int faceDim, global uint32_t* xPositive,
                          global uint32_t* xNegative, global uint32_t* yPositive, global uint32_t* yNegative,
                          global uint32_t* zPositive, global uint32_t* zNegative) {
  const int x = (int)get_global_id(0);
  const int y = (int)get_global_id(1);

  if (x < faceDim && y < faceDim) {
    /* compensate fetching offset with CLK_FILTER_LINEAR by adding 0.5f */
    float2 uv = make_float2(x + 0.5f, y + 0.5f);
    uv = (uv / faceDim) * 2.f - make_float2(1.f, 1.f);

    float3 pt = {0, 0, 0};
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

      const float4 im = read_imagef(tex, smp, xy) * 255.0f;
      if (face == 0) {
        xPositive[y * faceDim + x] = (uchar)im.s3 << 24 | (uchar)im.s2 << 16 | (uchar)im.s1 << 8 | (uchar)im.s0;
      } else if (face == 1) {
        xNegative[y * faceDim + x] = (uchar)im.s3 << 24 | (uchar)im.s2 << 16 | (uchar)im.s1 << 8 | (uchar)im.s0;
      } else if (face == 2) {
        yPositive[y * faceDim + x] = (uchar)im.s3 << 24 | (uchar)im.s2 << 16 | (uchar)im.s1 << 8 | (uchar)im.s0;
      } else if (face == 3) {
        yNegative[y * faceDim + x] = (uchar)im.s3 << 24 | (uchar)im.s2 << 16 | (uchar)im.s1 << 8 | (uchar)im.s0;
      } else if (face == 4) {
        zPositive[y * faceDim + x] = (uchar)im.s3 << 24 | (uchar)im.s2 << 16 | (uchar)im.s1 << 8 | (uchar)im.s0;
      } else if (face == 5) {
        zNegative[y * faceDim + x] = (uchar)im.s3 << 24 | (uchar)im.s2 << 16 | (uchar)im.s1 << 8 | (uchar)im.s0;
      }
    }
  }
}
