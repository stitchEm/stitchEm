// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/logging.hpp"
#include "libvideostitch/postprocessor.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"

#include <iostream>
#include <memory>

namespace VideoStitch {
namespace Core {

namespace {
/**
 * A PostProcessor that just chains several PostProcessors.
 */
class MultiPostProcessor : public PostProcessor {
 public:
  ~MultiPostProcessor() {
    for (auto postprocessor = postprocessors.begin(); postprocessor != postprocessors.end(); ++postprocessor) {
      delete *postprocessor;
    }
  }

  void add(PostProcessor* postprocessor) { postprocessors.push_back(postprocessor); }

  Status process(GPU::Buffer<uint32_t>& devBuffer, const PanoDefinition& pano, frameid_t frame,
                 GPU::Stream& stream) const {
    for (auto postprocessor = postprocessors.begin(); postprocessor != postprocessors.end(); ++postprocessor) {
      PROPAGATE_FAILURE_STATUS((*postprocessor)->process(devBuffer, pano, frame, stream));
    }
    return Status::OK();
  }

 private:
  std::vector<PostProcessor*> postprocessors;
};
}  // namespace

Potential<PostProcessor> PostProcessor::create(const Ptv::Value& config) {
  // We accept either lists of processors or single objects.
  if (config.getType() == Ptv::Value::LIST) {
    if (config.asList().size() == 0) {
      return {Origin::PostProcessor, ErrType::InvalidConfiguration, "Configuration is empty"};
    } else if (config.asList().size() == 1) {
      return create(*config.asList()[0]);
    } else {
      std::unique_ptr<MultiPostProcessor> multiPostprocessor(new MultiPostProcessor());
      for (auto subConfig = config.asList().begin(); subConfig != config.asList().end(); ++subConfig) {
        Potential<PostProcessor> postprocessor = PostProcessor::create(*(*subConfig));
        FAIL_RETURN(postprocessor.status());
        multiPostprocessor->add(postprocessor.release());
      }
      return Potential<PostProcessor>(multiPostprocessor.release());
    }
  }

  if (!Parse::checkType("PostProcessor", config, Ptv::Value::OBJECT)) {
    return {Origin::PostProcessor, ErrType::InvalidConfiguration, "Invalid 'PostProcessor' configuration type"};
  }

  std::string strType;
  if (Parse::populateString("PostProcessor", config, "type", strType, true) == Parse::PopulateResult_WrongType) {
    return {Origin::PostProcessor, ErrType::InvalidConfiguration,
            "Invalid 'PostProcessor' configuration type, expected string"};
  }

  return {Origin::PostProcessor, ErrType::InvalidConfiguration, "No such PostProcessor: '" + strType + "'"};
}

}  // namespace Core
}  // namespace VideoStitch
