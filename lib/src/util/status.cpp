// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/status.hpp"
#include "libvideostitch/logging.hpp"

#include <iostream>
#include <sstream>

namespace VideoStitch {

// Trigger the debugger if a Status is ignored in DEBUG mode

// TODOLATERSTATUS: enable this by default in DEBUG mode
// need to make all tests compatible
//#ifndef NDEBUG
//  #define DEBUG_IGNORED_STATUS
//#endif

#ifdef DEBUG_IGNORED_STATUS

#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif

#include <signal.h>

// an error Status has been created, but the contained information was never accessed
inline void trigger_debugger() {
#ifdef _MSC_VER
  __debugbreak();
#else
  asm("int3");
#endif
}
#endif

static const char* originToString(Origin origin) {
  switch (origin) {
    case Origin::AudioPipeline:
      return "audio pipeline";
    case Origin::AudioPipelineConfiguration:
      return "audio pipeline configuration";
    case Origin::AudioPreProcessor:
      return "audio preprocessor";
    case Origin::BlendingMaskAlgorithm:
      return "blending mask algorithm";
    case Origin::CropAlgorithm:
      return "automatic crop detection";
    case Origin::CalibrationAlgorithm:
      return "calibration algorithm";
    case Origin::ExposureAlgorithm:
      return "exposure algorithm";
    case Origin::ExternalModule:
      return "external module";
    case Origin::GPU:
      return "GPU";
    case Origin::PhotometricCalibrationAlgorithm:
      return "photometric calibration algorithm";
    case Origin::ScoringAlgorithm:
      return "stitching score algorithm";
    case Origin::StabilizationAlgorithm:
      return "stabilization algorithm";
    case Origin::SynchronizationAlgorithm:
      return "synchronization algorithm";
    case Origin::MotionEstimationAlgorithm:
      return "motion estimation algorithm";
    case Origin::MaskInterpolationAlgorithm:
      return "blending mask interpolation algorithm";
    case Origin::EpipolarCurvesAlgorithm:
      return "epipolar curves algorithm";
    case Origin::Stitcher:
      return "stitcher";
    case Origin::PostProcessor:
      return "post-processor";
    case Origin::PreProcessor:
      return "pre-processor";
    case Origin::Input:
      return "input";
    case Origin::ImageFlow:
      return "image flow";
    case Origin::ImageWarper:
      return "image warper";
    case Origin::Output:
      return "output";
    case Origin::PanoramaConfiguration:
      return "panorama configuration";
    case Origin::GeometryProcessingUtils:
      return "geometry processing utils";
    case Origin::Surface:
      return "surface";
    case Origin::Unspecified:
      return "unspecified module";
  }
  assert(false);
  return "Invalid error origin";
}

static const char* typeToString(ErrType type) {
  switch (type) {
    case ErrType::ImplementationError:
      return "Internal error";
    case ErrType::UnsupportedAction:
      return "Unsupported action";
    case ErrType::None:
      return "None";
    case ErrType::OutOfResources:
      return "Out of resources";
    case ErrType::SetupFailure:
      return "Setup failure";
    case ErrType::AlgorithmFailure:
      return "Algorithm execution failure";
    case ErrType::InvalidConfiguration:
      return "Invalid configuration";
    case ErrType::RuntimeError:
      return "Runtime error";
    case ErrType::OperationAbortedByUser:
      return "Operation cancelled by user";
  }
  assert(false);
  return "Invalid error type";
}

/**
 * An object that indicates error status.
 */
class Status::Description {
 public:
  explicit Description(Origin o, ErrType t, const std::string& msg)
      : origin(o), type(t), message(msg), cause(nullptr) {}
  explicit Description(Origin o, ErrType t, const std::string& msg, Status p)
      : origin(o), type(t), message(msg), cause(new Status(p)) {}

  bool hasCause() const { return !!cause; }

  const Status* getCause() const { return cause.get(); }

  Origin getOrigin() const {
#ifdef DEBUG_IGNORED_STATUS
    hasBeenChecked = true;
#endif
    return origin;
  }

  ErrType getType() const {
#ifdef DEBUG_IGNORED_STATUS
    hasBeenChecked = true;
#endif
    return type;
  }

  std::string getCurrentMessageOnly() const {
#ifdef DEBUG_IGNORED_STATUS
    hasBeenChecked = true;
#endif
    return message;
  }

#ifdef DEBUG_IGNORED_STATUS
  ~Description() {
    if (!hasBeenChecked) {
      // an error Status has been created, but the contained information was never accessed
      trigger_debugger();
    }
  }
#endif

  Description(const Description& other)
      : origin(other.getOrigin()),
        type(other.getType()),
        message(other.getCurrentMessageOnly()),
        cause(other.cause ? new Status(*other.cause) : nullptr) {}

 private:
  Origin origin;
  ErrType type;

  std::string message;

  std::unique_ptr<Status> cause;

#ifdef DEBUG_IGNORED_STATUS
  mutable bool hasBeenChecked = false;
#endif
};

const std::string STATUStag("Status");

Status::Status(Origin o, ErrType t, const std::string& message) : description(new Description(o, t, message)) {
  std::stringstream msg;
  msg << typeToString(t) << ": " << message << std::endl;
  Logger::warning(STATUStag, originToString(o)) << msg.str() << std::flush;
  // Should never happen, impossible
  assert(t != ErrType::ImplementationError);
}

Status::Status(Origin o, ErrType t, const std::string& message, Status parent)
    : description(new Description(o, t, message, parent)) {
  std::stringstream msg;
  msg << typeToString(t) << ": " << description->getCurrentMessageOnly() << std::endl;
  Logger::warning(STATUStag, originToString(o)) << msg.str();
}

Status::Status(Origin o, ErrType t, const std::string& tag, const std::string& message)
    : description(new Description(o, t, Logger::concatenateTags(tag) + message)) {
  std::stringstream msg;
  msg << typeToString(t) << ": " << message << std::endl;
  Logger::warning(STATUStag, tag, originToString(o)) << msg.str() << std::flush;
}

Status::Status(Origin o, ErrType t, const std::string& tag, const std::string& message, Status parent)
    : description(new Description(o, t, Logger::concatenateTags(tag) + message, parent)) {
  std::stringstream msg;
  msg << typeToString(t) << ": " << description->getCurrentMessageOnly() << std::endl;
  Logger::warning(STATUStag, tag, originToString(o)) << msg.str() << std::flush;
}

Status::Status() : description(nullptr) {}

Status::Status(const Status& other) : description(other.description ? new Description(*other.description) : nullptr) {}

std::string Status::getErrorMessage() const {
  if (description) {
    return description->getCurrentMessageOnly();
  }
  assert(false);
  return "";
}

Origin Status::getOrigin() const {
  if (description) {
    return description->getOrigin();
  }
  assert(false);
  return Origin::Unspecified;
}

ErrType Status::getType() const {
  if (description) {
    return description->getType();
  }
  assert(false);
  return ErrType::None;
}

std::string Status::getOriginString() const { return originToString(getOrigin()); }

std::string Status::getTypeString() const { return typeToString(getType()); }

bool Status::hasCause() const {
  if (description) {
    if (description->hasCause()) {
      return !description->getCause()->ok();
    }
  }
  return false;
}

const Status& Status::getCause() const {
  if (hasCause()) {
    return *description->getCause();
  }
  return *this;
}

bool Status::hasUnderlyingCause(ErrType t) const {
  if (getType() == t) {
    return true;
  } else if (hasCause()) {
    return description->getCause()->hasUnderlyingCause(t);
  } else {
    return false;
  }
}

Status::~Status() { delete description; }

}  // namespace VideoStitch
