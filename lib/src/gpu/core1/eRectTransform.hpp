// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef ERECTTRANSFORM_HPP_
#define ERECTTRANSFORM_HPP_

#include "matrix.hpp"
#include <gpu/buffer.hpp>
#include <gpu/cachedBuffer.hpp>
#include <gpu/channelFormat.hpp>
#include <gpu/stream.hpp>

#include <stdint.h>

namespace VideoStitch {
namespace Core {

class InputDefinition;
class PanoDefinition;
class GeoTransform;
struct Rect;

class ERectTransform {
 public:
  /**
   * Creates a Transform that maps the given input into the given panorama.
   * @param pano The PanoDefinition
   * @param im The InputDefinition
   */
  static ERectTransform* create(const PanoDefinition& pano, const InputDefinition& im);

  virtual ~ERectTransform() {}

  /**
   * Maps the contents of the current buffer to the output texture
   * @param time Current time value.
   * @param outputWidth Width of the output buffer
   * @param outputHeight Height of the output buffer
   * @param outputBuffer Output buffer. Dimensions are in @a outputBounds.
   * @param eRectUp The up vector of the cylinder used for equi-rectangular projection, it is (0, 0, 1) by default
   * @param pano Panorama definition. Must be the same as the one that was used to create the Transform.
   * @param im Input definition. Must be the same as the one that was used to create the Transform.
   * @param inputBuffer Input buffer. Has the the size of the input that was used for the creation of this Transform.
   * @param stream The cuda stream where to do the computations.
   * @note Asynchronous.
   */
  virtual Status mapBuffer(const int time, const size_t outputWidth, const size_t outputHeight,
                           GPU::Buffer<uint32_t> outputBuffer, const Vector3<double> pole, const PanoDefinition& pano,
                           const InputDefinition& im, const GPU::Buffer<const uint32_t> inputBuffer,
                           GPU::Stream stream) const = 0;

 protected:
};
}  // namespace Core
}  // namespace VideoStitch

#endif
