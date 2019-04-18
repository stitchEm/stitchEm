// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "depthPipeline.hpp"

#include "sphereSweep.hpp"

// TODO tmp: remove
#include "gpu/uniqueBuffer.hpp"
#include "gpu/memcpy.hpp"

#include "libvideostitch/depthDef.hpp"

#define USE_SGM 1
#define USE_BILATERAL 1

namespace VideoStitch {
namespace Core {

DepthPipeline::DepthPipeline(const std::vector<Input::VideoReader*>& readers, const PanoDefinition& pano,
                             const DepthDefinition& depthDef)
    : InputPipeline(readers, pano), depthPyramid(nullptr), depthDef(new DepthDefinition(depthDef)) {}

DepthPipeline::~DepthPipeline() { delete depthPyramid; }

Status DepthPipeline::preprocessGroup(const std::map<videoreaderid_t, SourceSurface*>& src, GPU::Stream& stream) {
  for (videoreaderid_t i = 0; i < panoDef->numVideoInputs(); i++) {
    FAIL_RETURN(inputPyramids[i].compute(src.at(i), stream));
  }

  return Status::OK();
}

Status DepthPipeline::processInput(videoreaderid_t sourceID, int frame, GPU::Surface& dst,
                                   const std::map<videoreaderid_t, SourceSurface*>& /*src*/,
                                   const InputDefinition& /*inputDef*/, GPU::Stream& stream) const {
#if USE_SGM
  const auto sgmPostProcessing = SGMPostProcessing::On;
#else
  const auto sgmPostProcessing = SGMPostProcessing::Off;
#endif

#if USE_BILATERAL
  const auto bilateralFilterPostProcessing = BilateralFilterPostProcessing::On;
#else
  const auto bilateralFilterPostProcessing = BilateralFilterPostProcessing::Off;
#endif

  return Core::sphereSweepInputMultiScale(sourceID, frame, dst, inputPyramids, *depthPyramid, *panoDef, *depthDef,
                                          sgmPostProcessing, bilateralFilterPostProcessing, stream);
}

Status DepthPipeline::initDepth(const DepthDefinition& depthDef) {
  FAIL_RETURN(init());

  for (const InputDefinition& videoInputDef : panoDef->getVideoInputs()) {
    InputPyramid inputPyramid{depthDef.getNumPyramidLevels(), (int)videoInputDef.getWidth(),
                              (int)videoInputDef.getHeight()};
    inputPyramids.push_back(std::move(inputPyramid));
  }

  // TODO check all inputs have the same resolution
  if (depthPyramid) {
    delete depthPyramid;
  }
  depthPyramid = nullptr;
  size_t inputWidth = panoDef->getVideoInput(0).getWidth();
  size_t inputHeight = panoDef->getVideoInput(0).getHeight();
  depthPyramid = new DepthPyramid{depthDef.getNumPyramidLevels(), (int)inputWidth, (int)inputHeight};

  return Status::OK();
}

Potential<DepthPipeline> DepthPipeline::createDepthPipeline(const std::vector<Input::VideoReader*>& readers,
                                                            const PanoDefinition& pano,
                                                            const DepthDefinition& depthDef) {
  std::unique_ptr<DepthPipeline> ret(new DepthPipeline(readers, pano, depthDef));
  FAIL_RETURN(ret->initDepth(depthDef));
  return Potential<DepthPipeline>{ret.release()};
}

}  // namespace Core
}  // namespace VideoStitch
