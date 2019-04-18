// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "core/videoPipeline.hpp"

namespace VideoStitch {
namespace Core {

class SourceSurface;
class Transform;
class OverrideOutputDefinition;

/**
 * Pipeline to process the inputs without doing any stitching
 */
class InputPipeline : public VideoPipeline {
 public:
  virtual ~InputPipeline();

  Status process(mtime_t date, FrameRate frameRate, std::map<readerid_t, Input::PotentialFrame>& inputBuffers,
                 std::vector<ExtractOutput*> extracts);

 protected:
  InputPipeline(const std::vector<Input::VideoReader*>&, const PanoDefinition& pano);

  Status init() override;

  // Init processing for this group of surfaces, called once per frame
  virtual Status preprocessGroup(const std::map<videoreaderid_t, SourceSurface*>& src, GPU::Stream& stream) = 0;

  // Called after preprocessGroup for each input individually
  virtual Status processInput(videoreaderid_t sourceID, int frame, GPU::Surface& dst,
                              const std::map<videoreaderid_t, SourceSurface*>& src, const InputDefinition& inputDef,
                              GPU::Stream& stream) const = 0;

  std::unique_ptr<PanoDefinition> panoDef;

 private:
  std::map<videoreaderid_t, SourceSurface*> processedSurfaces;
};

}  // namespace Core
}  // namespace VideoStitch
