// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "inputPipeline.hpp"

#include "core/stitchOutput/stitchOutput.hpp"
#include "common/container.hpp"
#include "gpu/core1/transform.hpp"

#include "libvideostitch/allocator.hpp"
#include "libvideostitch/gpu_device.hpp"

namespace VideoStitch {
namespace Core {

InputPipeline::InputPipeline(const std::vector<Input::VideoReader*>& readers, const PanoDefinition& pano)
    : VideoPipeline(readers, {}, nullptr), panoDef(pano.clone()) {}

InputPipeline::~InputPipeline() { deleteAllValues(processedSurfaces); }

Status InputPipeline::process(mtime_t date, FrameRate frameRate,
                              std::map<readerid_t, Input::PotentialFrame>& inputBuffers,
                              std::vector<ExtractOutput*> extracts) {
  FAIL_RETURN(GPU::useDefaultBackendDevice());

  GPU::Stream extractStream = GPU::Stream::getDefault();

  for (auto extract : extracts) {
    const videoreaderid_t sourceID = extract->pimpl->getSource();
    SourceSurface* surf = processedSurfaces.find(sourceID)->second;
    GPU::Surface* gpuSurf = surf->pimpl->surface;
    FAIL_RETURN(extraction(inputBuffers.find(sourceID)->second, sourceID, *gpuSurf, extractStream));
  }

  FAIL_RETURN(preprocessGroup(processedSurfaces, extractStream));

  // TODO perf: we should synchronize the GPU streams below on the GPU extractStream
  // instead of doing a CPU sync here, but that makes windows_opencl deadlock
  extractStream.synchronize();

  const int frame = frameRate.timestampToFrame(date);

  for (auto extract : extracts) {
    GPU::Stream stream;
    GPU::Surface& rbB = extract->pimpl->acquireFrame(date, stream);
    const videoreaderid_t sourceID = extract->pimpl->getSource();
    const InputDefinition& inputDef = panoDef->getVideoInput(sourceID);

    FAIL_RETURN(processInput(sourceID, frame, rbB, processedSurfaces, inputDef, stream));
    extract->pimpl->pushVideo(date);
  }

  return Status::OK();
}

Status InputPipeline::init() {
  FAIL_RETURN(VideoPipeline::init());

  for (const auto& pair : readers) {
    const Input::VideoReader* r = pair.second;
    Potential<SourceSurface> potSurf =
        OffscreenAllocator::createSourceSurface(r->getWidth(), r->getHeight(), "Processed source");
    FAIL_RETURN(potSurf.status());
    processedSurfaces[pair.first] = potSurf.release();
  }

  return Status::OK();
}

}  // namespace Core
}  // namespace VideoStitch
