// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "core/inputControllerImpl.hpp"

#include "libvideostitch/overrideDef.hpp"
#include "libvideostitch/undistortController.hpp"

#ifdef _MSC_VER
#pragma warning(push)
// using virtual inheritance of InputController on purpose
#pragma warning(disable : 4250)
#endif

namespace VideoStitch {
namespace Core {

class UndistortControllerImpl : public InputControllerImpl, public UndistortController {
 public:
  static Potential<UndistortController> create(const PanoDefinition& panoIn, const AudioPipeDefinition& audioPipeDef,
                                               Input::ReaderFactory* readerFactory,
                                               const OverrideOutputDefinition& outputDef);

  virtual ~UndistortControllerImpl();

  virtual ControllerStatus undistort(std::vector<ExtractOutput*> extracts) final override;

  virtual Potential<ExtractOutput> createAsyncExtractOutput(int sourceID, std::shared_ptr<SourceSurface> surf,
                                                            std::shared_ptr<Output::VideoWriter> writer) final override;

  virtual Potential<PanoDefinition> createPanoDefWithoutDistortion() final override;

 private:
  UndistortControllerImpl(const PanoDefinition& pano, ReaderController* readerController, InputPipeline* videoPipe,
                          const OverrideOutputDefinition& outputDef);

  std::unique_ptr<PanoDefinition> pano;
  InputPipeline* pipe;
  const OverrideOutputDefinition outputDef;
};

}  // namespace Core
}  // namespace VideoStitch

#ifdef _MSC_VER
#pragma warning(pop)
#endif
