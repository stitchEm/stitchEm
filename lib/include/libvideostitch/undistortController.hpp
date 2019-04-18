// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "inputController.hpp"

#include "frame.hpp"
#include "input.hpp"
#include "status.hpp"

#include <vector>

namespace VideoStitch {

namespace Input {
class ReaderFactory;
}

namespace Output {
class VideoWriter;
}

namespace Core {

class PanoDefinition;
class AudioPipeDefinition;
class ExtractOutput;
class ReaderController;
class SourceSurface;
class InputPipeline;
class OverrideOutputDefinition;

class VS_EXPORT UndistortController : public virtual InputController {
 public:
  virtual ~UndistortController() {}

  virtual ControllerStatus undistort(std::vector<ExtractOutput*> extracts) = 0;

  virtual Potential<ExtractOutput> createAsyncExtractOutput(int sourceID, std::shared_ptr<SourceSurface> surf,
                                                            std::shared_ptr<Output::VideoWriter> writer) = 0;

  virtual Potential<PanoDefinition> createPanoDefWithoutDistortion() = 0;
};

Potential<UndistortController> VS_EXPORT createUndistortController(const PanoDefinition& pano,
                                                                   const AudioPipeDefinition& audioPipeDef,
                                                                   Input::ReaderFactory* readerFactory,
                                                                   const OverrideOutputDefinition& outputDef);

}  // namespace Core
}  // namespace VideoStitch
