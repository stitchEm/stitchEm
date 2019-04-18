// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/buffer.hpp"
#include "gpu/surface.hpp"
#include "gpu/stream.hpp"

#include <stdint.h>

namespace VideoStitch {
namespace Render {

/**
 * @brief A class that draws something in a device buffer.
 */
class Renderer {
 public:
  virtual ~Renderer() {}

  /**
   * Draw in a device buffer.
   * @param dst Destination buffer.
   * @param dstWidth Buffer width.
   * @param dstHeight Buffer height.
   * @param left bounding box left
   * @param top bounding box top
   * @param right bounding box right
   * @param bottom bounding box bottom
   * @param color Strike color.
   * @param bgcolor Background color.
   * @param stream GPU Stream to run in
   * @note Asynchronous.
   */
  virtual void draw(uint32_t* dst, std::size_t dstWidth, std::size_t dstHeight, std::size_t left, std::size_t top,
                    std::size_t right, std::size_t bottom, uint32_t color, uint32_t bgcolor,
                    GPU::Stream stream) const = 0;
};

/**
 * Fill a buffer with a given color.
 * @param dst Destination buffer.
 * @param value Fill value.
 * @param size Buffer size (in uint32_t s)
 * @param stream GPU stream to run in.
 * @Note Asynchronous.
 */
Status fillBuffer(GPU::Buffer<uint32_t> dst, uint32_t value, size_t width, size_t height, GPU::Stream stream);

}  // namespace Render
}  // namespace VideoStitch
