// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "exposureScoring.hpp"

#include "exposureScoringProcessor.hpp"

#include "core1/exposureDiffImageMerger.hpp"
#include "parse/json.hpp"
#include "output/discardOutput.hpp"
#include "util/registeredAlgo.hpp"
#include "parallax/noWarper.hpp"
#include "parallax/noFlow.hpp"

#include "libvideostitch/controller.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/output.hpp"

namespace VideoStitch {
namespace Scoring {

namespace {
Util::RegisteredAlgo<ExposureScoringAlgorithm> registered("exposure_scoring");
}

const char* ExposureScoringAlgorithm::docString =
    "An algorithm that computes an exposure score for the stitched image\n";

ExposureScoringAlgorithm::ExposureScoringAlgorithm(const Ptv::Value* config) {
  firstFrame = 0;
  lastFrame = 0;

  if (config != NULL) {
    const Ptv::Value* value;

    value = config->has("first_frame");
    if (value && value->getType() == Ptv::Value::INT) {
      firstFrame = (int)value->asInt();
      if (firstFrame < 0) {
        firstFrame = 0;
      }
    }

    value = config->has("last_frame");
    if (value && value->getType() == Ptv::Value::INT) {
      lastFrame = (int)value->asInt();
      if (lastFrame < firstFrame) {
        lastFrame = firstFrame;
      }
    }
  }
}

ExposureScoringAlgorithm::~ExposureScoringAlgorithm() {}

Potential<Ptv::Value> ExposureScoringAlgorithm::apply(Core::PanoDefinition* pano, ProgressReporter* /* progress */,
                                                      Util::OpaquePtr** /* ctx */) const {
  Core::ExposureDiffImageMerger::Factory imfactory;
  Core::NoFlow::Factory flowfactory;
  Core::NoWarper::Factory warperfactory;

  Input::DefaultReaderFactory* readerFactory = new Input::DefaultReaderFactory(firstFrame, lastFrame);
  if (!readerFactory) {
    return {Origin::ScoringAlgorithm, ErrType::SetupFailure, "Cannot create reader factory"};
  }

  std::unique_ptr<Core::AudioPipeDefinition> audioPipeDef(Core::AudioPipeDefinition::createDefault());
  Core::PotentialController controller =
      Core::createController(*pano, imfactory, warperfactory, flowfactory, readerFactory, *audioPipeDef);
  FAIL_CAUSE(controller.status(), Origin::ScoringAlgorithm, ErrType::SetupFailure, "Cannot create controller");

  Potential<Core::PanoSurface> surf =
      Core::OffscreenAllocator::createPanoSurface(pano->getWidth(), pano->getHeight(), "Exposure scoring algorithm");
  FAIL_RETURN(surf.status());
  Core::StitchOutput* output =
      controller->createBlockingStitchOutput(std::shared_ptr<Core::PanoSurface>(surf.release())).release();

  controller->createStitcher();
  int boundedLastFrame = controller->getLastStitchableFrame();
  if (boundedLastFrame > lastFrame) {
    boundedLastFrame = lastFrame;
  }

  ExposureScoringPostProcessor* postprocessor = ExposureScoringPostProcessor::create();
  controller->setPostProcessor(postprocessor);

  Potential<Ptv::Value> ret(Ptv::Value::emptyObject());

  int curframe = firstFrame;
  while (curframe <= boundedLastFrame) {
    if (controller->stitch(output).ok()) {
      std::array<double, 3> score = postprocessor->getScore();

      Ptv::Value* list = Ptv::Value::emptyObject();
      list->push("diff_red", new Parse::JsonValue(score[0]));
      list->push("diff_green", new Parse::JsonValue(score[1]));
      list->push("diff_blue", new Parse::JsonValue(score[2]));
      ret->asList().push_back(list);
    }
    curframe++;
  }

  controller->deleteStitcher();
  deleteController(controller.release());
  delete output;

  return ret;
}

}  // namespace Scoring
}  // namespace VideoStitch
