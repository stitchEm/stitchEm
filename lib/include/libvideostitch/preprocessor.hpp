// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef PREPROCESSOR_HPP_
#define PREPROCESSOR_HPP_

#include "config.hpp"
#include "status.hpp"

#include <iosfwd>

namespace VideoStitch {

class ThreadSafeOstream;

namespace GPU {
class Surface;
class Stream;
}  // namespace GPU

namespace Input {
class Reader;
}
namespace Ptv {
class Value;
}
namespace Util {
class OpaquePtr;
}

namespace Core {
/**
 * @brief The common interface for preprocessors.
 *
 * Note that this class is performance-critical, so if you think about implementing
 * a custom reader you should be aware of general computing performance issues and read
 * the design docs of the input library.
 *
 * A preprocessor is applied just before mapping an image. It can be used e.g. to overlay or transform the input data.
 * They act on a device buffer.
 * Preprocessor are stateless and take their context through a context object of type OpaquePtr.
 */
class VS_EXPORT PreProcessor {
 public:
  virtual ~PreProcessor() {}

  /**
   * Creates a PreProcessor from a config. Returns NULL on failure.
   */
  static Potential<PreProcessor> create(const Ptv::Value& config, Util::OpaquePtr** ctx = NULL);

  /**
   * Processes the given buffer.
   * The kernel call should use the given stream, and should be asynchronous
   * for better performance.
   * @param frame current frame
   * @param surface GPU surface to process.
   * @param width input width
   * @param height input height
   * @param inputId input id
   * @param stream CUDA stream where to do the computations.
   */
  virtual Status process(frameid_t frame, GPU::Surface& surface, int64_t width, int64_t height, readerid_t inputId,
                         GPU::Stream& stream) const = 0;

  /**
   * Returns a display name for the reader. Never assume any format for that. EVER.
   * That means that the only thing that's allowed with that is to diplay it to the user.
   * No testing for equality, parsing...
   * @param os Output stream.
   */
  virtual void getDisplayName(std::ostream& os) const = 0;

 protected:
  PreProcessor() {}
};
}  // namespace Core
}  // namespace VideoStitch

#endif
