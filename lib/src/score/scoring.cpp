// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "scoring.hpp"

#include "scoringProcessor.hpp"

#include "parallax/noWarper.hpp"
#include "parallax/noFlow.hpp"

#include "util/registeredAlgo.hpp"
#include "parse/json.hpp"
#include "core1/noblendImageMerger.hpp"
#include "output/discardOutput.hpp"

#include "libvideostitch/controller.hpp"
#include "libvideostitch/imageMergerFactory.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/output.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/stitchOutput.hpp"
#include "libvideostitch/imageFlowFactory.hpp"

namespace VideoStitch {
namespace Scoring {

namespace {
Util::RegisteredAlgo<ScoringAlgorithm> registered("scoring");
}

const char* ScoringAlgorithm::docString =
    "An algorithm that computes a calibration score and uncovered areas per stitched image\n";

ScoringAlgorithm::ScoringAlgorithm(const Ptv::Value* config) {
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

ScoringAlgorithm::~ScoringAlgorithm() {}

Potential<Ptv::Value> ScoringAlgorithm::apply(Core::PanoDefinition* pano, ProgressReporter* /* progress */,
                                              Util::OpaquePtr** /* ctx */) const {
  Core::NoBlendImageMerger::Factory imfactory;
  Core::NoWarper::Factory warperFactory;
  Core::NoFlow::Factory flowFactory;

  /*Reader for input*/
  Input::DefaultReaderFactory* readerFactory = new Input::DefaultReaderFactory(firstFrame, lastFrame);
  if (!readerFactory) {
    return {Origin::ScoringAlgorithm, ErrType::SetupFailure, "Cannot create reader factory"};
  }

  /*Controller for stitching*/
  std::unique_ptr<Core::AudioPipeDefinition> audioPipeDef(Core::AudioPipeDefinition::createDefault());
  Core::PotentialController controller =
      Core::createController(*pano, imfactory, warperFactory, flowFactory, readerFactory, *audioPipeDef);
  FAIL_CAUSE(controller.status(), Origin::ScoringAlgorithm, ErrType::SetupFailure, "Cannot create controller");

  /*Dumb output writer, we don't need it*/
  std::shared_ptr<Output::VideoWriter> writer(new Output::DiscardVideoWriter(
      "discard", (unsigned)pano->getWidth(), (unsigned)pano->getHeight(), controller->getFrameRate()));
  auto surf = Core::OffscreenAllocator::createPanoSurface(pano->getWidth(), pano->getHeight(), "Scoring Algorithm");
  Core::StitchOutput* output =
      controller->createBlockingStitchOutput(std::shared_ptr<Core::PanoSurface>(surf.release()), writer).release();

  /*Create the stitcher*/
  controller->createStitcher();
  int boundedLastFrame = controller->getLastStitchableFrame();
  if (boundedLastFrame > lastFrame) {
    boundedLastFrame = lastFrame;
  }

  /* Postprocessor */
  ScoringPostProcessor* postprocessor = ScoringPostProcessor::create();
  controller->setPostProcessor(postprocessor);

  /*Create the result*/
  Potential<Ptv::Value> ret(Ptv::Value::emptyObject());

  /*Loop over frames*/
  int curframe = firstFrame;
  while (curframe <= boundedLastFrame) {
    if (controller->stitch(output).ok()) {
      double score, uncovered;

      postprocessor->getScore(score, uncovered);

      Ptv::Value* list = Ptv::Value::emptyObject();
      list->push("frame_number", new Parse::JsonValue(curframe));
      list->push("score", new Parse::JsonValue(score));
      list->push("uncovered_ratio", new Parse::JsonValue(uncovered));
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
