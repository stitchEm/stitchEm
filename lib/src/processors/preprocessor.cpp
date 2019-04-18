// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "exprProcessor.hpp"
#include "gridProcessor.hpp"
#include "tintProcessor.hpp"
#include "maskProcessor.hpp"

#include "libvideostitch/parse.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/preprocessor.hpp"
#include "libvideostitch/ptv.hpp"

#include <memory>

namespace VideoStitch {
namespace Core {

namespace {
/**
 * A PreProcessor that just chains several PreProcessors.
 */
class MultiPreProcessor : public PreProcessor {
 public:
  ~MultiPreProcessor() {
    for (auto preprocessor = preprocessors.begin(); preprocessor != preprocessors.end(); ++preprocessor) {
      delete *preprocessor;
    }
  }

  void add(PreProcessor* preprocessor) { preprocessors.push_back(preprocessor); }

  Status process(frameid_t frame, GPU::Surface& devBuffer, int64_t width, int64_t height, readerid_t inputId,
                 GPU::Stream& stream) const {
    Status s;
    for (auto preprocessor = preprocessors.begin(); preprocessor != preprocessors.end(); ++preprocessor) {
      Status processingStatus = (*preprocessor)->process(frame, devBuffer, width, height, inputId, stream);
      if (processingStatus.ok()) {
        s = processingStatus;
      }
    }
    return s;
  }

  void getDisplayName(std::ostream& os) const {
    for (auto preprocessor = preprocessors.begin(); preprocessor != preprocessors.end(); ++preprocessor) {
      (*preprocessor)->getDisplayName(os);
      os << ", ";
    }
  }

 private:
  std::vector<PreProcessor*> preprocessors;
};
}  // namespace

Potential<PreProcessor> PreProcessor::create(const Ptv::Value& config, Util::OpaquePtr** /*ctx*/) {
  // We accept either lists of processors or single objects.
  if (config.getType() == Ptv::Value::LIST) {
    if (config.asList().size() == 0) {
      return {Origin::PreProcessor, ErrType::InvalidConfiguration, "Configuration is empty"};
    } else if (config.asList().size() == 1) {
      return create(*config.asList()[0]);
    } else {
      std::unique_ptr<MultiPreProcessor> multiPreprocessor(new MultiPreProcessor());
      for (auto subConfig = config.asList().begin(); subConfig != config.asList().end(); ++subConfig) {
        Potential<PreProcessor> preprocessor = PreProcessor::create(*(*subConfig));
        FAIL_RETURN(preprocessor.status());
        multiPreprocessor->add(preprocessor.release());
      }
      return Potential<PreProcessor>(multiPreprocessor.release());
    }
  }

  if (!Parse::checkType("PreProcessor", config, Ptv::Value::OBJECT)) {
    return {Origin::PreProcessor, ErrType::InvalidConfiguration, "Invalid configuration type for 'PreProcessor'"};
  }

  std::string strType;
  if (Parse::populateString("PreProcessor", config, "type", strType, true) == Parse::PopulateResult_WrongType) {
    return {Origin::PreProcessor, ErrType::InvalidConfiguration,
            "Invalid configuration type for 'PreProcessor', expected string"};
  }

  if (strType == "expr") {
    return Potential<PreProcessor>(ExprProcedure::create(config));
  } else if (strType == "tint") {
    return Potential<PreProcessor>(TintPreProcessor::create(config));
  } else if (strType == "mask") {
    return Potential<PreProcessor>(MaskPreProcessor::create(config));
  }
  return {Origin::PreProcessor, ErrType::InvalidConfiguration, "No such pre-processor: '" + strType + "'"};
}

}  // namespace Core
}  // namespace VideoStitch
