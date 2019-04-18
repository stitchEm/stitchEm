// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef GLYPHS_H_
#define GLYPHS_H_

#include "gpu/render/render.hpp"

namespace VideoStitch {
namespace Render {

/**
 * @brief A renderer that fills the boundingbox.
 */
class FillRenderer {
 public:
  FillRenderer() {}

  Status draw(uint32_t* dst, int64_t dstWidth, int64_t dstHeight, int64_t left, int64_t top, int64_t right,
              int64_t bottom, uint32_t color, uint32_t bgcolor, cudaStream_t stream) const;
};
}  // namespace Render
}  // namespace VideoStitch

#endif
