// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef POSTPROCESSOR_HPP_
#define POSTPROCESSOR_HPP_

#include "config.hpp"
#include "status.hpp"
#include "panoDef.hpp"

namespace VideoStitch {

namespace GPU {
template <typename T>
class Buffer;

class Stream;
}  // namespace GPU

namespace Ptv {
class Value;
}

namespace Core {
/**
 * @brief The common interface for postprocessors.
 *
 * A postprocessor is applied just after rendering a panorama. It can be used e.g. to overlay or transform the output
 * data. They act on a device buffer. Postprocessor are stateless and take their context through a Context object.
 */
class VS_EXPORT PostProcessor {
 public:
  virtual ~PostProcessor() {}

  /**
   * Creates a PostProcessor from a config. Returns NULL on failure.
   * @param config prostprocessor config.
   */
  static Potential<PostProcessor> create(const Ptv::Value& config);

  /**
   * Processes the given buffer.
   * @param devBuffer Device buffer to process.
   * @param pano the pano PanoDefinition
   * @param frame the current frame.
   * @param stream the GPU stream to do the processing on.
   */
  virtual Status process(GPU::Buffer<uint32_t>& devBuffer, const PanoDefinition& pano, frameid_t frame,
                         GPU::Stream& stream) const = 0;

 protected:
  PostProcessor() {}
};
}  // namespace Core
}  // namespace VideoStitch

#endif
