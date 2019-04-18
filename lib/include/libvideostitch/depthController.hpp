// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "inputController.hpp"

#include "frame.hpp"
#include "input.hpp"
#include "status.hpp"

#include <vector>

// TODODEPTH: does mostly the same as UndistortController
// --> merge implementations, put into InputControllerImpl?
// differenes: UndistortController has additional method createPanoDefWithoutDistortion
// UndistortController uses UndistortPipeline, this uses DepthPipeline
// --> solve with factory or templating

namespace VideoStitch {

namespace Input {
class ReaderFactory;
}

namespace Output {
class VideoWriter;
}

namespace Core {

class AudioPipeDefinition;
class DepthDefinition;
class ExtractOutput;
class PanoDefinition;
class SourceSurface;

class VS_EXPORT DepthController : public virtual InputController {
 public:
  virtual ~DepthController() {}

  virtual ControllerStatus estimateDepth(std::vector<ExtractOutput*> extracts) = 0;

  virtual Potential<ExtractOutput> createAsyncExtractOutput(int sourceID, std::shared_ptr<SourceSurface> surf,
                                                            std::shared_ptr<Output::VideoWriter> writer) = 0;
};

Potential<DepthController> VS_EXPORT createDepthController(const PanoDefinition& pano, const DepthDefinition& depthDef,
                                                           const AudioPipeDefinition& audioPipeDef,
                                                           Input::ReaderFactory* readerFactory);

}  // namespace Core
}  // namespace VideoStitch
