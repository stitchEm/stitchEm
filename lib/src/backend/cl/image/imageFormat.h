// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

static inline int clamp8(int c) { return clamp(c, 0, 255); }

/**
 * Solid RGB.
 */

/**
 * Get the red component for a pixel.
 * @param v pixel value.
 */
static inline unsigned int Image_RGBSolid_r(uchar3 v) { return v.x; }

/**
 * Get the green component for a pixel.
 * @param v pixel value.
 */
static inline unsigned int Image_RGBSolid_g(uchar3 v) { return v.y; }

/**
 * Get the blue component for a pixel.
 * @param v pixel value.
 */
static inline unsigned int Image_RGBSolid_b(uchar3 v) { return v.z; }

/**
 * Get the alpha component for a pixel.
 * @param v RGBA64 packed pixel.
 */
static inline unsigned int Image_RGBSolid_a(char3 v) { return 255; }

/**
 * Pack RGBA values into a 32 bits pixel as .
 * @param r Red component. Between 0 and 255.
 * @param g Green component. Between 0 and 255.
 * @param b Blue component. Between 0 and 255.
 * @param a Alpha component. Ignored.
 */
static inline uchar3 Image_RGBSolid_pack(unsigned int r, unsigned int g, unsigned int b, unsigned int a) {
  uchar3 v;
  v.x = (unsigned char)r;
  v.y = (unsigned char)g;
  v.z = (unsigned char)b;
  return v;
}

/**
 * RGBA format: 8 bits per component, AAAAAAAABBBBBBBBGGGGGGGGRRRRRRRR
 */

/**
 * Get the red component for a pixel.
 * @param v RGBA packed pixel.
 */
static inline unsigned int Image_RGBA_r(unsigned int v) { return v & (unsigned int)0xff; }

/**
 * Get the green component for a pixel.
 * @param v RGBA packed pixel.
 */
static inline unsigned int Image_RGBA_g(unsigned int v) { return (v >> 8) & (unsigned int)0xff; }

/**
 * Get the blue component for a pixel.
 * @param v RGBA packed pixel.
 */
static inline unsigned int Image_RGBA_b(unsigned int v) { return (v >> 16) & (unsigned int)0xff; }

/**
 * Get the alpha component for a pixel.
 * @param v RGBA packed pixel.
 */
static inline unsigned int Image_RGBA_a(unsigned int v) { return (v >> 24) & (unsigned int)0xff; }

/**
 * Pack RGBA values into a 32 bits pixel as .
 * @param r Red component. Between 0 and 255.
 * @param g Green component. Between 0 and 255.
 * @param b Blue component. Between 0 and 255.
 * @param a Alpha component. Between 0 and 255.
 */
static inline unsigned int Image_RGBA_pack(unsigned int r, unsigned int g, unsigned int b, unsigned int a) {
  return (a << 24) | (b << 16) | (g << 8) | r;
}

/**
 * RGBASolid format: 8 bits per component, alpha always 0xff
 */

/**
 * Get the red component for a pixel.
 * @param v RGBA packed pixel.
 */
static inline unsigned int Image_RGBASolid_r(unsigned int v) { return v & (unsigned int)0xff; }

/**
 * Get the green component for a pixel.
 * @param v RGBA packed pixel.
 */
static inline unsigned int Image_RGBASolid_g(unsigned int v) { return (v >> 8) & (unsigned int)0xff; }

/**
 * Get the blue component for a pixel.
 * @param v RGBA packed pixel.
 */
static inline unsigned int Image_RGBASolid_b(unsigned int v) { return (v >> 16) & (unsigned int)0xff; }

/**
 * Get the alpha component for a pixel.
 * @param v RGBA packed pixel.
 */
static inline unsigned int Image_RGBASolid_a(unsigned int v) { return (unsigned int)0xff; }

/**
 * Pack RGBA values into a 32 bits pixel as .
 * @param r Red component. Between 0 and 255.
 * @param g Green component. Between 0 and 255.
 * @param b Blue component. Between 0 and 255.
 * @param a Alpha component. Between 0 and 255.
 */
static inline unsigned int Image_RGBASolid_pack(unsigned int r, unsigned int g, unsigned int b, unsigned int a) {
  return 0xff000000u | (b << 16) | (g << 8) | r;
}

/**
 * RGB210 format: 1 alpha bit, 3 * signed 9-bit colors: A_RRRRRRRRRRGGGGGGGGGGBBBBBBBBBB
 * The colors are packed in an OpenGL-compatible way, as 10-bits unsigned colors.
 */

/**
 * Get the red component for a packed pixel.
 * @param v RGB210 packed pixel.
 */
static inline int Image_RGB210_r(unsigned int v) {
  int const m = 0x200;  // 1 << (10 - 1)
  unsigned int x = v & (unsigned int)0x3ff;
  x = (x >> 2) ^ ((0x3 & x) << 8);
  return (x ^ m) - m;
}

/**
 * Get the green component for a packed pixel.
 * @param v RGB210 packed pixel.
 */
static inline int Image_RGB210_g(unsigned int v) {
  int const m = 0x200;  // 1 << (10 - 1)
  unsigned int x = (v >> 10) & (unsigned int)0x3ff;
  x = (x >> 2) ^ ((0x3 & x) << 8);
  return (x ^ m) - m;
}

/**
 * Get the blue component for a packed pixel.
 * @param v RGB210 packed pixel.
 */
static inline int Image_RGB210_b(unsigned int v) {
  int const m = 0x200;  // 1 << (10 - 1)
  unsigned int x = (v >> 20) & (unsigned int)0x3ff;
  x = (x >> 2) ^ ((0x3 & x) << 8);
  return (x ^ m) - m;
}

/**
 * Get the alpha component for a packed pixel.
 * @param v RGB210 packed pixel.
 * @note This is guaranteed to return only 0 or 1.
 */
static inline int Image_RGB210_a(unsigned int v) { return (int)(v >> 31) == 0 ? 0 : 255; }

/**
 * Pack RGBA values into a 32 bits pixel.
 * @param r Red component. Between -511 and 511.
 * @param g Green component. Between -511 and 511.
 * @param b Blue component. Between -511 and 511.
 * @param a Alpha component. If <= 0, transparent. Else, solid.
 */
static inline unsigned int Image_RGB210_pack(int r, int g, int b, int a) {
  unsigned int packed_r = ((unsigned int)r << 2) + (((unsigned int)r & 0x300) >> 8);
  unsigned int packed_g = ((unsigned int)g << 2) + (((unsigned int)g & 0x300) >> 8);
  unsigned int packed_b = ((unsigned int)b << 2) + (((unsigned int)b & 0x300) >> 8);
  return (((unsigned int)!!(a > 0)) * (unsigned int)0x80000000) | ((packed_b & 0x3ff) << 20) |
         ((packed_g & 0x3ff) << 10) | (packed_r & 0x3ff);
}

/**
 * Pack RGBA values into a 32 bits pixel as 8-bit monochrome Y component.
 * @param r Red component. Between 0 and (255).
 * @param g Green component. Between 0 and (255).
 * @param b Blue component. Between 0 and (255).
 * @param a Alpha component. Ignored.
 */
static inline unsigned char Image_MonoY_pack(unsigned int r, unsigned int g, unsigned int b, unsigned int a) {
  return (unsigned char)(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16);
}

static inline void surface_write_i(unsigned rgba, write_only image2d_t img, int x, int y) {
  const int2 coords = {x, y};
  const float4 color = {Image_RGBA_r(rgba) / 255.f, Image_RGBA_g(rgba) / 255.f, Image_RGBA_b(rgba) / 255.f,
                        Image_RGBA_a(rgba) / 255.f};
  write_imagef(img, coords, color);
}

static inline void surface_write_f(float4 color, write_only image2d_t img, int x, int y) {
  const int2 coords = {x, y};
  write_imagef(img, coords, color);
}
