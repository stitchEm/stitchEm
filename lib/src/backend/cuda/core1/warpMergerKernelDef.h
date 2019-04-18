// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

__device__ void gradientMerge(const bool v, global_mem uint32_t* g_odata, surface_t panoSurf, const uint32_t p,
                              global_mem const unsigned char* mask, const int srcIndex, const int panoX,
                              const int panoY, const int panoWidth) {
  if (v && Image_RGBA_a(p)) {
    const unsigned char m = mask[srcIndex];
    uint32_t q;
    surface_read(&q, panoSurf, panoX, panoY);

    if (Image_RGBA_a(q)) {
      uint32_t mR = (m * Image_RGBA_r(p) + (255 - m) * Image_RGBA_r(q) + 127) / 255;
      uint32_t mG = (m * Image_RGBA_g(p) + (255 - m) * Image_RGBA_g(q) + 127) / 255;
      uint32_t mB = (m * Image_RGBA_b(p) + (255 - m) * Image_RGBA_b(q) + 127) / 255;
      surface_write(Image_RGBA_pack(mR, mG, mB, 0xff), panoSurf, panoX, panoY);
    } else {
      if (m) {
        surface_write(p, panoSurf, panoX, panoY);
      }
    }
  }
}

__device__ void noopMerge(const bool v, global_mem uint32_t* g_odata, surface_t panoSurf, const uint32_t p,
                          global_mem const unsigned char* mask, const int srcIndex, const int panoX, const int panoY,
                          const int panoWidth) {
  g_odata[srcIndex] = p;
}
