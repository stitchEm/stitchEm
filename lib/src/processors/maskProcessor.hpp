// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/buffer.hpp"

#include "libvideostitch/preprocessor.hpp"

namespace VideoStitch {

class ThreadSafeOstream;

namespace Core {

/**
 * @brief A processor that shows the alpha-masked values by overlaying them
 */
class MaskPreProcessor : public PreProcessor {
 public:
  static MaskPreProcessor* create(const Ptv::Value& config);
  /**
   * @param color tint color.
   */
  explicit MaskPreProcessor(uint32_t color);
  ~MaskPreProcessor();

  Status process(frameid_t frame, GPU::Surface& devBuffer, int64_t width, int64_t height, readerid_t inputId,
                 GPU::Stream& stream) const;
  void getDisplayName(std::ostream& os) const;

 private:
  const uint32_t color;
};
}  // namespace Core
}  // namespace VideoStitch
