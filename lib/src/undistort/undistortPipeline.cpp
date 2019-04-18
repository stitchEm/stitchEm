// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "undistortPipeline.hpp"

#include "core/stitchOutput/stitchOutput.hpp"
#include "common/container.hpp"
#include "gpu/core1/transform.hpp"

#include "libvideostitch/allocator.hpp"
#include "libvideostitch/gpu_device.hpp"

namespace VideoStitch {
namespace Core {

UndistortPipeline::UndistortPipeline(const std::vector<Input::VideoReader*>& readers, const PanoDefinition& pano,
                                     const OverrideOutputDefinition& overrideDef)
    : InputPipeline(readers, pano), overrideDef(overrideDef) {
  for (const auto& videoInput : panoDef->getVideoInputs()) {
    transforms.push_back(Transform::create(videoInput));
  }
}

UndistortPipeline::~UndistortPipeline() { deleteAll(transforms); }

Status UndistortPipeline::processInput(videoreaderid_t sourceID, int frame, GPU::Surface& dst,
                                       const std::map<videoreaderid_t, SourceSurface*>& inputSurfaces,
                                       const InputDefinition& inputDef, GPU::Stream& stream) const {
  SourceSurface* surf = inputSurfaces.find(sourceID)->second;
  GPU::Surface* gpuSurf = surf->pimpl->surface;
  // TODO is that needed?
  surf->acquire();
  Status processResult = [&]() -> Status {
    std::unique_ptr<InputDefinition> outputDef{inputDef.clone()};
    overrideDef.applyOverrideSettings(*outputDef);

    FAIL_RETURN(transforms[sourceID]->undistortInput(frame, dst, *gpuSurf, *panoDef, inputDef, *outputDef, stream));
    return Status::OK();
  }();
  surf->release();

  return processResult;
}

Potential<UndistortPipeline> UndistortPipeline::createUndistortPipeline(const std::vector<Input::VideoReader*>& readers,
                                                                        const PanoDefinition& pano,
                                                                        const OverrideOutputDefinition& overrideDef) {
  std::unique_ptr<UndistortPipeline> ret(new UndistortPipeline(readers, pano, overrideDef));
  FAIL_RETURN(ret->init());
  return Potential<UndistortPipeline>{ret.release()};
}

}  // namespace Core
}  // namespace VideoStitch
