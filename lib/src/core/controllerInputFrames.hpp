// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "core/readerController.hpp"

#include "panoPipeline.hpp"

#include "libvideostitch/allocator.hpp"
#include "libvideostitch/imageMergerFactory.hpp"
#include "libvideostitch/imageWarperFactory.hpp"
#include "libvideostitch/imageFlowFactory.hpp"
#include "libvideostitch/inputFactory.hpp"

#include "gpu/2dBuffer.hpp"

#include <algorithm>
#include <utility>

namespace VideoStitch {
namespace Core {

/**
 A wrapper around ReaderController that simplifies the API just to get frames with the default reader.
 Additionally, frames are unpacked on the GPU, and then copied back to the host.

 TODO performance is probably terrible

 */
template <PixelFormat destinationColor, typename readbackType>
class ControllerInputFrames {
 public:
  static Potential<ControllerInputFrames> create(const Core::PanoDefinition* pano);
  ~ControllerInputFrames();

  Status seek(frameid_t frame);

  // load a set of frames from the readers
  // memory is valid until the next load() call
  Status load(std::map<readerid_t, PotentialValue<GPU::HostBuffer<readbackType>>>& frames, mtime_t* date = nullptr);

 private:
  Status init(const Core::PanoDefinition*);

  Status processFrame(Buffer readerFrame, GPU::HostBuffer<readbackType> readbackDestination, readerid_t readerID);

  Core::ReaderController* readerController;
  std::vector<GPU::HostBuffer<readbackType>> readbackFrames;
  GPU::Buffer<unsigned char> devBuffer;
  GPU::Buffer2D grayscale;
  SourceSurface* surf = nullptr;
  GPU::Stream stream;
};

}  // namespace Core
}  // namespace VideoStitch
