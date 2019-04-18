// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "core1/textureTarget.hpp"
#include "core1/imageMerger.hpp"

#include "gpu/buffer.hpp"
#include "gpu/surface.hpp"
#include "gpu/stream.hpp"

#include "libvideostitch/matrix.hpp"

#include <stdint.h>

namespace VideoStitch {
namespace Core {

class InputDefinition;
class PanoDefinition;
class PhotoTransform;

struct Rect;

class Transform {
 public:
  /**
   * Creates a Transform that maps the given input.
   * @param im The InputDefinition
   */
  static Transform* create(const InputDefinition& im, const ImageMerger::Format type);
  static Transform* create(const InputDefinition& im);

  virtual ~Transform() {}

  /**
   * Set the imId bit in each pixel of devOut in which the input will be mapped.
   * @param devOut Output buffer. Size is the same as @a pano.
   * @param pano Panorama definition. Must be the same as the one that was used to create the Transform.
   * @param im Input definition. Must be the same as the one that was used to create the Transform.
   * @param maskDevBuffer If not NULL, pixels that are 1 in the mask are set to zero alpha.
   * @param stream The cuda stream where to do the computations.
   * @note Asynchronous.
   */
  virtual Status computeZone(GPU::Buffer<uint32_t> devOut, const PanoDefinition& pano, const InputDefinition& im,
                             videoreaderid_t imId, GPU::Buffer<const unsigned char> maskDevBuffer,
                             GPU::Stream stream) const = 0;
  virtual Status cubemapMap(GPU::Buffer<uint32_t> xPos, GPU::Buffer<uint32_t> xNeg, GPU::Buffer<uint32_t> yPos,
                            GPU::Buffer<uint32_t> yNeg, GPU::Buffer<uint32_t> zPos, GPU::Buffer<uint32_t> zNeg,
                            const PanoDefinition& pano, const InputDefinition& im, videoreaderid_t imId,
                            GPU::Buffer<const unsigned char> maskDevBuffer, bool equiangular,
                            GPU::Stream stream) const = 0;

  /**
   * Maps the contents of the current texture into the output buffer.
   * @param frame Current frame id.
   * @param devOut Output buffer. Dimensions are in @a outputBounds.
   * @param mask merger mask buffer (if blending is combined with mapping).
   * @param outputBounds Size of @a devOut.
   * @param pano Panorama definition. Must be the same as the one that was used to create the Transform.
   * @param im Input definition. Must be the same as the one that was used to create the Transform.
   * @param surface Input texture. Has the the size of the input that was used for the creation of this Transform.
   * @param stream The GPU stream where to do the computations.
   * @note Asynchronous.
   */
  virtual Status mapBuffer(frameid_t frame, GPU::Buffer<uint32_t> pbo, GPU::Surface&, const unsigned char* mask,
                           const Rect& boundingBox, const PanoDefinition&, const InputDefinition&, GPU::Surface&,
                           GPU::Stream) const = 0;
  virtual Status warpCubemap(frameid_t, GPU::Buffer<uint32_t> xPosPbo, const Rect& xPosBB,
                             GPU::Buffer<uint32_t> xNegPbo, const Rect& xNegBB, GPU::Buffer<uint32_t> yPosPbo,
                             const Rect& yPosBB, GPU::Buffer<uint32_t> yNegPbo, const Rect& yNegBB,
                             GPU::Buffer<uint32_t> zPosPbo, const Rect& zPosBB, GPU::Buffer<uint32_t> zNegPbo,
                             const Rect& zNegBB, const PanoDefinition&, const InputDefinition&, GPU::Surface&,
                             bool equiangular, GPU::Stream) const = 0;

  /**
   * Maps the contents of the current texture into the output buffer using precomputed coordinate buffer.
   * @param frame Current frame id.
   * @param devOut Output buffer. Dimensions are in @a outputBounds.
   * @param mask merger mask buffer (if blending is combined with mapping).
   * @param coordIn Input lookup coordinate buffer. Dimensions are in @a outputBounds.
   * @param coordShrinkFactor Input lookup coordinate shrink factor applied to the Dimensions in @a outputBounds.
   * @param outputBounds Size of @a devOut.
   * @param pano Panorama definition. Must be the same as the one that was used to create the Transform.
   * @param im Input definition. Must be the same as the one that was used to create the Transform.
   * @param texDevice Input array. Has the the size of the input that was used for the creation of this Transform.
   * @param channelDesc The channel description for devArray.
   * @param stream The GPU stream where to do the computations.
   * @note Asynchronous.
   */
  virtual Status mapBufferLookup(frameid_t frame, GPU::Buffer<uint32_t> devOut, GPU::Surface& surf,
                                 const unsigned char* mask, const GPU::Surface& coordIn, const float coordShrinkFactor,
                                 const Rect& outputBounds, const PanoDefinition& pano, const InputDefinition& im,
                                 GPU::Surface& surface, GPU::Stream stream) const = 0;

  /**
   * Precomputed the wrapped coordinate buffer.
   * @param frame Current frame id.
   * @param devCoord Output coordinate buffer. Dimensions are in @a outputBounds.
   * @param outputBounds Size of @a devOut.
   * @param pano Panorama definition. Must be the same as the one that was used to create the Transform.
   * @param im Input definition. Must be the same as the one that was used to create the Transform.
   * @param stream The GPU stream where to do the computations.
   * @note Asynchronous.
   */
  virtual Status mapBufferCoord(frameid_t frame, GPU::Surface& devCoord, const Rect& outputBounds,
                                const PanoDefinition& pano, const InputDefinition& im, GPU::Stream stream) const = 0;

  /**
   * Precomputed the lookup buffer from the input space to the output.
   * @param frame Current frame id.
   * @param scaleFactor The original input size is multiplied by a scale factor to support sub-pixel accuracy
   * @param inputCoord The ouput coordinate buffer. Store the lookup pixel coordinate in the output panorama. Size of
   * the buffer is imWidth * scaleFactor, imHeight * scaleFactor
   * @param pano Panorama definition. Must be the same as the one that was used to create the Transform.
   * @param im Input definition. Must be the same as the one that was used to create the Transform.
   * @param stream The GPU stream where to do the computations.
   * @note Asynchronous.
   */
  virtual Status mapCoordInput(int time, const int scaleFactor, GPU::Buffer<float2> inputCoord,
                               const PanoDefinition& pano, const InputDefinition& im, GPU::Stream gpuStream) const = 0;

  /**
   * Maps the distortion value of the current input into the output buffer.
   * @param time Current time.
   * @param devOut Output buffer. Dimensions are in @a outputBounds.
   * @param outputBounds Size of @a devOut.
   * @param pano Panorama definition. Must be the same as the one that was used to create the Transform.
   * @param im Input definition. Must be the same as the one that was used to create the Transform.
   * @param stream The GPU stream where to do the computations.
   * @note Asynchronous.
   */
  virtual Status mapDistortion(int time, GPU::Buffer<unsigned char> devOut, const Rect& outputBounds,
                               const PanoDefinition& pano, const InputDefinition& im, GPU::Stream stream) const = 0;

  /**
   * TODO
   * Maps the distortion value of the current input into the output buffer.
   * @param time Current time.
   * @param dst Destination input surface
   * @param src Source input surface
   * @param pano Panorama definition. Must be the same as the one that was used to create the Transform.
   * @param im Input definition. Must be the same as the one that was used to create the Transform.
   * @param undistortedFocal Focal used for the undistorted frame.
   * @param stream The GPU stream where to do the computations.
   * @note Asynchronous.
   */
  virtual Status undistortInput(int time, GPU::Surface& dst, const GPU::Surface& src, const PanoDefinition& pano,
                                const InputDefinition& recordedInput, const InputDefinition& undistortedOutput,
                                GPU::Stream& stream) const = 0;

 protected:
  friend class TransformParameters;
};
}  // namespace Core
}  // namespace VideoStitch
