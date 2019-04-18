// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "backend/cl/gpuKernelDef.h"
#include "backend/cl/image/imageFormat.h"

#include "backend/common/image/blurdef.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-align"

////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////

static __constant uint32_t c_Kernel[] = {1, 4, 6, 4, 1};

kernel void convolutionRowsKernel(global_mem uint32_t* __restrict__ dst, const global_mem uint32_t* __restrict__ src,
                                  int pitch, int wrap) {
  int idX = (int)get_local_id(0);
  int idY = (int)get_local_id(1);

  __local uint32_t s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

  // Offset to the left halo edge
  const int baseX = (int)((get_group_id(0) * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + idX);
  const int baseY = (int)(get_group_id(1) * ROWS_BLOCKDIM_Y + idY);

  src += baseY * pitch + baseX;
  dst += baseY * pitch + baseX;

  // Load main data
  for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
    s_Data[idY][idX + i * ROWS_BLOCKDIM_X] = (pitch - baseX > i * ROWS_BLOCKDIM_X)
                                                 ? src[i * ROWS_BLOCKDIM_X]
                                                 : (wrap ? src[i * ROWS_BLOCKDIM_X - baseX] : src[pitch - 1 - baseX]);
  }

  // Load left halo
  for (int i = 0; i < ROWS_HALO_STEPS; i++) {
    s_Data[idY][idX + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X)
                                                 ? src[i * ROWS_BLOCKDIM_X]
                                                 : (wrap ? src[pitch - baseX - i * ROWS_BLOCKDIM_X] : src[-baseX]);
  }

  // Load right halo
  for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
    s_Data[idY][idX + i * ROWS_BLOCKDIM_X] = (pitch - baseX > i * ROWS_BLOCKDIM_X)
                                                 ? src[i * ROWS_BLOCKDIM_X]
                                                 : (wrap ? src[i * ROWS_BLOCKDIM_X - baseX] : src[pitch - 1 - baseX]);
  }

  // Compute and store results
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
    uint32_t accR = 0;
    uint32_t accG = 0;
    uint32_t accB = 0;
    uint32_t divider = 0;
    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
      uint32_t v = s_Data[idY][idX + i * ROWS_BLOCKDIM_X + j];
      const int32_t isSolid = !!Image_RGBA_a(v);
      accR += isSolid * c_Kernel[KERNEL_RADIUS - j] * Image_RGBA_r(v);
      accG += isSolid * c_Kernel[KERNEL_RADIUS - j] * Image_RGBA_g(v);
      accB += isSolid * c_Kernel[KERNEL_RADIUS - j] * Image_RGBA_b(v);
      divider += isSolid * c_Kernel[KERNEL_RADIUS - j];
    }

    if (pitch - baseX > i * COLUMNS_BLOCKDIM_X) {
      dst[i * ROWS_BLOCKDIM_X] = (divider == 0) ? 0
                                                : Image_RGBA_pack(accR / divider, accG / divider, accB / divider,
                                                                  Image_RGBA_a(s_Data[idY][idX + i * ROWS_BLOCKDIM_X]));
    }
  }
}

kernel void convolutionColumnsKernel(global_mem uint32_t* __restrict__ dst, const global_mem uint32_t* __restrict__ src,
                                     int height, int pitch) {
  int idX = (int)get_local_id(0);
  int idY = (int)get_local_id(1);

  __local uint32_t s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

  // Offset to the upper halo edge
  const int baseX = (int)(get_group_id(0) * COLUMNS_BLOCKDIM_X + idX);
  const int baseY = (int)((get_group_id(1) * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + idY);
  src += baseY * pitch + baseX;
  dst += baseY * pitch + baseX;

  // Main data
  for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
    s_Data[idX][idY + i * COLUMNS_BLOCKDIM_Y] = (height - baseY > i * COLUMNS_BLOCKDIM_Y)
                                                    ? src[i * COLUMNS_BLOCKDIM_Y * pitch]
                                                    : src[(height - 1 - baseY) * pitch];
  }

  // Upper halo
  for (int i = 0; i < COLUMNS_HALO_STEPS; i++) {
    s_Data[idX][idY + i * COLUMNS_BLOCKDIM_Y] =
        (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? src[i * COLUMNS_BLOCKDIM_Y * pitch] : src[-baseY * pitch];
  }

  // Lower halo
  for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS;
       i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++) {
    s_Data[idX][idY + i * COLUMNS_BLOCKDIM_Y] = (height - baseY > i * COLUMNS_BLOCKDIM_Y)
                                                    ? src[i * COLUMNS_BLOCKDIM_Y * pitch]
                                                    : src[(height - 1 - baseY) * pitch];
  }

  // Compute and store results
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
    uint32_t accR = 0;
    uint32_t accG = 0;
    uint32_t accB = 0;
    uint32_t divider = 0;
    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
      uint32_t v = s_Data[idX][idY + i * COLUMNS_BLOCKDIM_Y + j];
      const int32_t isSolid = !!Image_RGBA_a(v);
      accR += isSolid * c_Kernel[KERNEL_RADIUS - j] * Image_RGBA_r(v);
      accG += isSolid * c_Kernel[KERNEL_RADIUS - j] * Image_RGBA_g(v);
      accB += isSolid * c_Kernel[KERNEL_RADIUS - j] * Image_RGBA_b(v);
      divider += isSolid * c_Kernel[KERNEL_RADIUS - j];
    }

    if (height - baseY > i * COLUMNS_BLOCKDIM_Y) {
      dst[i * COLUMNS_BLOCKDIM_Y * pitch] =
          (divider == 0) ? 0
                         : Image_RGBA_pack(accR / divider, accG / divider, accB / divider,
                                           Image_RGBA_a(s_Data[idX][idY + i * COLUMNS_BLOCKDIM_Y]));
    }
  }
}

#pragma clang diagnostic pop
