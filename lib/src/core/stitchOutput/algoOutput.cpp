// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/surface.hpp"

#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/stitchOutput.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/panoramaDefinitionUpdater.hpp"

namespace VideoStitch {
namespace Core {

/**
 * AlgorithmOutput
 */
AlgorithmOutput::AlgorithmOutput(Util::OnlineAlgorithm* algo, const PanoDefinition& pano, Listener& listener,
                                 Util::OpaquePtr** ctx)
    : algo(algo),
      panoramaUpdater(new PanoramaDefinitionUpdater(pano)),
      listener(listener),
      worker(nullptr),
      ctx(ctx),
      cancelled(false) {}

AlgorithmOutput::~AlgorithmOutput() {
  if (worker) {
    worker->join();
    delete worker;
  }
}

void AlgorithmOutput::onFrame(std::vector<std::pair<int, GPU::Surface&>>& inputFrames, mtime_t date,
                              FrameRate frameRate) {
  worker = new std::thread(asyncJob, this, inputFrames, date, frameRate);
}

void AlgorithmOutput::asyncJob(AlgorithmOutput* that, std::vector<std::pair<videoreaderid_t, GPU::Surface&>> frames,
                               mtime_t date, FrameRate frameRate) {
  Status status = GPU::useDefaultBackendDevice();
  if (!status.ok()) {
    that->listener.onError(status);
    return;
  }
  Potential<Ptv::Value> result = that->algo->onFrame(*(that->panoramaUpdater), frames, date, frameRate, that->ctx);
  if (that->cancelled) {
    return;
  }
  if (result.ok()) {
    that->listener.onPanorama(*(that->panoramaUpdater));
  } else {
    that->listener.onError(result.status());
  }
}

}  // namespace Core
}  // namespace VideoStitch
