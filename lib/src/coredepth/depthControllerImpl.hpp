// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "core/inputControllerImpl.hpp"

#include "libvideostitch/depthController.hpp"

#ifdef _MSC_VER
#pragma warning(push)
// using virtual inheritance of InputController on purpose
#pragma warning(disable : 4250)
#endif

namespace VideoStitch {
namespace Core {

class InputPipeline;

class DepthControllerImpl : public InputControllerImpl, public DepthController {
 public:
  static Potential<DepthController> create(const PanoDefinition& pano, const DepthDefinition& depthDef,
                                           const AudioPipeDefinition& audioPipeDef,
                                           Input::ReaderFactory* readerFactory);

  virtual ~DepthControllerImpl();

  virtual ControllerStatus estimateDepth(std::vector<ExtractOutput*> extracts) final override;

  virtual Potential<ExtractOutput> createAsyncExtractOutput(int sourceID, std::shared_ptr<SourceSurface> surf,
                                                            std::shared_ptr<Output::VideoWriter> writer) final override;

 private:
  DepthControllerImpl(const PanoDefinition& pano, ReaderController* readerController, InputPipeline* videoPipe);

  std::unique_ptr<PanoDefinition> pano;
  InputPipeline* pipe;
};

}  // namespace Core
}  // namespace VideoStitch

#ifdef _MSC_VER
#pragma warning(pop)
#endif
