// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/buffer.hpp"

#include "libvideostitch/preprocessor.hpp"

namespace VideoStitch {
namespace Core {

/**
 * @brief A processor that tints the input.
 * The result is just a luminosity-mapped version of the input with the given color.
 */
class TintPreProcessor : public PreProcessor {
 public:
  static TintPreProcessor* create(const Ptv::Value& config);
  /**
   * @param color tint color.
   */
  explicit TintPreProcessor(uint32_t color);
  ~TintPreProcessor();
  Status process(frameid_t frame, GPU::Surface& devBuffer, int64_t width, int64_t height, readerid_t inputId,
                 GPU::Stream& stream) const;
  void getDisplayName(std::ostream& os) const;

 private:
  const uint32_t color;
};
}  // namespace Core
}  // namespace VideoStitch
