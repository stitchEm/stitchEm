// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "textureTarget.hpp"

#include "gpu/hostBuffer.hpp"
#include "gpu/uniqueBuffer.hpp"

#include "libvideostitch/input.hpp"
#include "libvideostitch/output.hpp"
#include "libvideostitch/panoDef.hpp"

#include <vector>

namespace VideoStitch {
namespace Core {

/**
 * @brief The InputsMap class manage computation and storage of the inputs map
 * This inputs map contains for each pixel of the destination cubemap the list of inputs which are used to produce it.
 */
class InputsMapCubemap {
 public:
  static Potential<InputsMapCubemap> create(const PanoDefinition &pano);
  virtual ~InputsMapCubemap();

 public:
  Status compute(const std::map<readerid_t, Input::VideoReader *> &, const PanoDefinition &);

  GPU::Buffer<uint32_t> getMask(TextureTarget t) {
    switch (t) {
      case CUBE_MAP_POSITIVE_X:
        return xPos.borrow();
      case CUBE_MAP_NEGATIVE_X:
        return xNeg.borrow();
      case CUBE_MAP_POSITIVE_Y:
        return yPos.borrow();
      case CUBE_MAP_NEGATIVE_Y:
        return yNeg.borrow();
      case CUBE_MAP_POSITIVE_Z:
        return zPos.borrow();
      case CUBE_MAP_NEGATIVE_Z:
        return zNeg.borrow();
      default:
        assert(false);
        return xPos.borrow();
    }
  }

 private:
  explicit InputsMapCubemap(const PanoDefinition &pano);
  Status allocateBuffers();

  GPU::UniqueBuffer<uint32_t> xPos, xNeg, yPos, yNeg, zPos, zNeg;

  int64_t length;
};

}  // namespace Core
}  // namespace VideoStitch
