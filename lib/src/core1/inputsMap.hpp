// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/hostBuffer.hpp"
#include "gpu/uniqueBuffer.hpp"
#ifndef VS_OPENCL
#include "maskinterpolation/inputMaskInterpolation.hpp"
#endif

#include "libvideostitch/input.hpp"
#include "libvideostitch/output.hpp"
#include "libvideostitch/panoDef.hpp"

#include <stdint.h>
#include <vector>

namespace VideoStitch {
namespace Core {

class StereoRigDefinition;

/**
 * @brief The InputsMap class manage computation and storage of the inputs map
 * This inputs map contains for each pixel of the destination panorama the list of inputs which are used to produce it.
 */
class InputsMap {
 public:
  static Potential<InputsMap> create(const PanoDefinition& pano);
  virtual ~InputsMap();

 public:
  Status compute(const std::map<readerid_t, Input::VideoReader*>&, const PanoDefinition&, const bool);
  Status compute(const std::map<readerid_t, Input::VideoReader*>&, const PanoDefinition&, const StereoRigDefinition*,
                 Eye, const bool);

#ifndef VS_OPENCL
  Status loadPrecomputedMap(const frameid_t frameId, const PanoDefinition& pano,
                            const std::map<readerid_t, Input::VideoReader*>& readers,
                            std::unique_ptr<MaskInterpolation::InputMaskInterpolation>& inputMaskInterpolation,
                            bool& loaded);
#endif

  // These two functions are related to the frame id
  std::pair<frameid_t, frameid_t> getBoundedFrameIds() const;

  GPU::Buffer<uint32_t> getMask() { return setupBuffer.borrow(); }

 private:
  explicit InputsMap(const PanoDefinition& pano);

  Status allocateBuffers();

 private:
  GPU::UniqueBuffer<uint32_t> setupBuffer;

  std::pair<frameid_t, frameid_t> _boundedFrames;
  int64_t _width;
  int64_t _height;
};

}  // namespace Core
}  // namespace VideoStitch
