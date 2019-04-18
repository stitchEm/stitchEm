// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "undistort/inputPipeline.hpp"

#include "core/surfacePyramid.hpp"

namespace VideoStitch {
namespace Core {

class DepthDefinition;

/**
 * Pipeline to compute depth for inputs
 */
class DepthPipeline : public InputPipeline {
 public:
  virtual ~DepthPipeline();

  static Potential<DepthPipeline> createDepthPipeline(const std::vector<Input::VideoReader*>& inputs,
                                                      const PanoDefinition& pano, const DepthDefinition& depthDef);

 protected:
  DepthPipeline(const std::vector<Input::VideoReader*>&, const PanoDefinition& pano, const DepthDefinition& depthDef);
  Status initDepth(const DepthDefinition& depthDef);

 private:
  virtual Status preprocessGroup(const std::map<videoreaderid_t, SourceSurface*>& src, GPU::Stream& stream) override;

  virtual Status processInput(videoreaderid_t sourceID, int frame, GPU::Surface& dst,
                              const std::map<videoreaderid_t, SourceSurface*>& src, const InputDefinition& inputDef,
                              GPU::Stream& stream) const override;

 protected:
  std::vector<InputPyramid> inputPyramids;
  DepthPyramid* depthPyramid;

  std::unique_ptr<DepthDefinition> depthDef;
};

}  // namespace Core
}  // namespace VideoStitch
