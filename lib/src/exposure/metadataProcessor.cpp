// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "metadataProcessor.hpp"

#include "orah4iResponseCurve.h"

#include "libvideostitch/logging.hpp"

#include <memory>
#include <set>
#include <type_traits>

// Debug option: dump all incoming data to Logger::Info
// #define DUMP_METADATA

namespace VideoStitch {
namespace Exposure {

// allow for a gradual change in values over time
// no sudden jumps between successive frames
const int MIN_INTERPOLATION_LENGTH{10};

// https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
// if the numbers are < 1, just use epsilon for the comparison
// (don't need anything more exact for ev)
// otherwise, scale epsilon with the max abs value
bool almostEqual(double a, double b) {
  double epsilon = std::numeric_limits<double>::epsilon();

  double maxAbs = std::max(fabs(a), fabs(b));
  if (maxAbs > 1.) {
    epsilon *= maxAbs;
  }

  return fabs(a - b) <= epsilon;
}

// enforce the @value at @frame, whether it is currently
// a key frame (spline endpoint) or not
void enforceValue(Core::Curve& curve, int frame, double value) {
  if (!almostEqual(curve.at(frame), value)) {
    curve.splitAt(frame);
    Core::Spline* s = curve.splines();
    while (s) {
      if (s->end.t == frame) {
        s->end.v = value;

        auto next = s->next;
        // patch the type if splitAt created a line
        if (next && next->getType() == Core::Spline::LineType) {
          next->makeCubic();
        }
        break;
      }
      s = s->next;
    }
  }

  assert(almostEqual(curve.at(frame), value));
}

// Insert values into a curve while live processing at currentFrame.
// Guaranteed safety: the curve's value at currentFrame will stay the same.
//
// @param insertAtFrame: proposed time where value should be added
// @return: frame: where value was actually inserted, potentially later
//
// Motivation: there are many different states of the current curve imaginable,
// as the interval between values can vary, as can latency (insertFrame - currentFrame).
// Trying to cover all possible situations creates spaghetti code that is likely
// to forget some specific state.
//
// This solution inserts the new value into the curve with little regard to whether
// it might affect the current spline. It then patches up the value at the current frame,
// if it was broken.
//
// It is still possible that the current slope changes, if a value is inserted closely
// to the currentFrame. But the current value itself can not jump.
int curveSafeInsert(Core::Curve& curve, int insertAtFrame, double value, int currentFrame) {
  auto whereToAdd = std::max(insertAtFrame, currentFrame + MIN_INTERPOLATION_LENGTH);
  auto currentCurveValue = curve.at(currentFrame);

  {
    // will extend cubic to point in the future
    // (a) if it is covered by the current curve, will be dropped
    // (b) if it was in the past, will be inserted, may be dropped in next pruning
    Core::Curve point(Core::Spline::point(whereToAdd, value));
    curve.extend(&point);

    // in case (a) we may want to override the curve's previous value at insertAtFrame
    // for cases in which we are replacing 'fake' values in the future: delayed
    // placeholder values from interpolation with real, measured ones
    enforceValue(curve, whereToAdd, value);
  }

  // repair if we botched the value at the current stitching frame
  enforceValue(curve, currentFrame, currentCurveValue);

  return whereToAdd;
}

void updateGlobalExposure(const std::set<frameid_t>& updateGlobal, Core::PanoDefinition& pano,
                          frameid_t currentStitchingFrame) {
  Core::Curve* globalExposure = pano.getExposureValue().clone();
  for (frameid_t toUpdate : updateGlobal) {
    double sum{0.};
    for (const Core::InputDefinition& input : pano.getVideoInputs()) {
      sum += input.getExposureValue().at(toUpdate);
    }
    double avg = sum / pano.numVideoInputs();
    double globalCompensation = -avg;

    curveSafeInsert(*globalExposure, toUpdate, globalCompensation, currentStitchingFrame);
  }

  assert(almostEqual(globalExposure->at(currentStitchingFrame), pano.getExposureValue().at(currentStitchingFrame)) &&
         "Safe insert should not cause jumps at the current frame");
  pano.replaceExposureValue(globalExposure);
}

void insertExposureMetadata(const std::vector<std::map<videoreaderid_t, Metadata::Exposure>>& exposureMetadata,
                            FrameRate frameRate, Core::PanoDefinition& pano, frameid_t currentStitchingFrame) {
  std::set<frameid_t> updateGlobal;

  // step 1: insert the exposure measurements into the exposure curves, without causing jumps
  for (const std::map<videoreaderid_t, Metadata::Exposure>& exposureMeasure : exposureMetadata) {
    for (const auto& pair : exposureMeasure) {
      videoreaderid_t inputID = pair.first;
      if (inputID >= pano.numVideoInputs()) {
        Logger::get(Logger::Warning) << "[MetadataProcessor] Received data for input ID " << inputID
                                     << ", which is out of bounds (have " << pano.numVideoInputs() << " video inputs)"
                                     << std::endl;
        continue;
      }

      Core::InputDefinition& input = pano.getVideoInput(inputID);

      Metadata::Exposure data = pair.second;
      mtime_t timestamp = data.timestamp;

#ifdef DUMP_METADATA
      {
        std::stringstream msg;
        msg << "[MetadataProcessor] #" << inputID << ": " << data << std::endl;
        Logger::get(Logger::Info) << msg.str() << std::endl;
      }
#endif  // DUMP_METADATA

      auto frameid = frameRate.timestampToFrame(timestamp);

      if (!data.isValid()) {
        Logger::get(Logger::Warning) << "[MetadataProcessor] Received invalid exposure data: " << data << std::endl;
        continue;
      }

      double exposureCompensation = -(data.computeEv());

      Core::Curve* exposureCurve = input.getExposureValue().clone();

      if (frameid < currentStitchingFrame) {
        Logger::get(Logger::Warning) << "[MetadataProcessor] Received data for frame " << frameid
                                     << " while stitching frame " << currentStitchingFrame << ", too late!"
                                     << std::endl;
      }

      auto addedAt = curveSafeInsert(*exposureCurve, frameid, exposureCompensation, currentStitchingFrame);
      assert(
          almostEqual(exposureCurve->at(currentStitchingFrame), input.getExposureValue().at(currentStitchingFrame)) &&
          "Safe insert should not cause jumps at the current frame");

      updateGlobal.insert(addedAt);

      input.replaceExposureValue(exposureCurve);
    }
  }

  // step 2: update the global exposure in all places where there were new measurements
  updateGlobalExposure(updateGlobal, pano, currentStitchingFrame);
}

Core::Curve* pruneCurve(const Core::Curve& curve, frameid_t currentStitchingFrame) {
  Core::Curve* pruned = curve.clone();

  while (const Core::Spline* firstSpline = pruned->splines()) {
    const Core::Spline* secondSpline = firstSpline->next;
    if (secondSpline && secondSpline->next && secondSpline->next->end.t < currentStitchingFrame) {
      pruned->mergeAt(firstSpline->end.t);
    } else {
      break;
    }
  }

  assert(almostEqual(pruned->at(currentStitchingFrame), curve.at(currentStitchingFrame)) &&
         "Pruning should not cause jumps at the current frame");
  return pruned;
}

void pruneExposureCurves(Core::InputDefinition& input, frameid_t currentStitchingFrame) {
  input.replaceExposureValue(pruneCurve(input.getExposureValue(), currentStitchingFrame));
}

void pruneExposureCurves(Core::PanoDefinition& pano, frameid_t currentStitchingFrame) {
  // in live, we are going forward only
  // prune the curve history to not accumulate too much memory over time
  for (Core::InputDefinition& input : pano.getVideoInputs()) {
    pruneExposureCurves(input, currentStitchingFrame);
  }
  pano.replaceExposureValue(pruneCurve(pano.getExposureValue(), currentStitchingFrame));
}

void MetadataProcessor::pruneToneCurves(frameid_t currentStitchingFrame, FrameRate frameRate) {
  for (auto& byReader : toneCurves) {
    ToneCurveByTime& byTime = byReader.second;

    mtime_t canDeleteBelow = std::numeric_limits<mtime_t>::min();
    for (auto reverseIt = byTime.rbegin(); reverseIt != byTime.rend(); ++reverseIt) {
      if (frameRate.timestampToFrame(reverseIt->first) <= currentStitchingFrame) {
        canDeleteBelow = reverseIt->first;
        break;
      }
    }

    ToneCurveByTime::iterator it = byTime.begin();
    while (it != byTime.end()) {
      if (it->first < canDeleteBelow) {
        it = byTime.erase(it);
      } else {
        ++it;
      }
    }
  }
}

template <typename T>
float lookupValueLinearInterpolation(const std::array<T, 256>& curve, float index8f) {
  assert(index8f >= 0.f && index8f < (float)curve.size());

  const int lowerIndex8i = std::max(0, std::min((int)index8f, (int)curve.size() - 1));
  const int upperIndex8i = std::min(lowerIndex8i + 1, (int)curve.size() - 1);

  const float x = index8f - (float)lowerIndex8i;
  float val = (1.0f - x) * curve[lowerIndex8i] + x * curve[upperIndex8i];

  return val;
}

Metadata::ToneCurve applyOrah4iResponse(const Metadata::ToneCurve& metadataCurve) {
  Metadata::ToneCurve newValues{metadataCurve};
  const std::array<uint16_t, 256> metadataValues = metadataCurve.curveAsArray();
  for (int i = 0; i < 256; i++) {
    // lookup camera response
    float f = lookupValueLinearInterpolation(orah4iCurve, (float)i) / 1023.f;
    assert(f >= 0.f);
    assert(f <= 1.f);
    // lookup metadata response
    f = lookupValueLinearInterpolation(metadataValues, f * 255.f);
    uint16_t v = (uint16_t)roundf(f);
    assert(v <= 1023);
    // write new value
    newValues.curve[i] = v;
  }
  return newValues;
}

void MetadataProcessor::insertToneCurveMetadata(std::vector<std::map<videoreaderid_t, Metadata::ToneCurve>> newData) {
  for (const std::map<videoreaderid_t, Metadata::ToneCurve>& tc : newData) {
    for (const auto& kv : tc) {
      const Metadata::ToneCurve& val = kv.second;
      const Metadata::ToneCurve applied = applyOrah4iResponse(val);
      toneCurves[kv.first][val.timestamp] = applied;
    }
  }
}

bool inputDefNeedsToneCurveUpdate(const Core::InputDefinition& videoInput, const Metadata::ToneCurve& toneCurve) {
  if (videoInput.getPhotoResponse() != Core::InputDefinition::PhotoResponse::CurveResponse) {
    return true;
  }

  if (!videoInput.getValueBasedResponseCurve()) {
    return true;
  }

  const std::array<uint16_t, 256>& currentResponse = *videoInput.getValueBasedResponseCurve();
  const uint16_t(&curve)[256] = toneCurve.curve;

#if (!_MSC_VER || _MSC_VER >= 1900)
  static_assert(std::remove_reference<decltype(currentResponse)>::type().size() == sizeof(curve) / sizeof(curve[0]),
                "Metadata ToneCurve needs to be the same size as InputDef photo response curve size");
#endif

  for (size_t index = 0; index < currentResponse.size(); index++) {
    if (currentResponse[index] != curve[index]) {
      return true;
    }
  }

  return false;
}

void MetadataProcessor::createUpdatedPanoForCurrentFrame(std::unique_ptr<Core::PanoDefinition>& potentialNewPano,
                                                         const Core::PanoDefinition& currentPano, FrameRate frameRate,
                                                         frameid_t currentStitchingFrame) const {
  mtime_t currentStitcherDate = frameRate.frameToTimestamp(currentStitchingFrame);

  auto getNewPano = [&potentialNewPano, &currentPano]() -> Core::PanoDefinition& {
    // late init, we may not need to copy the pano if there are no updates
    if (!potentialNewPano) {
      potentialNewPano.reset(currentPano.clone());
    }
    return *potentialNewPano;
  };

  for (const auto& kv : toneCurves) {
    videoreaderid_t id = kv.first;
    const Core::InputDefinition& videoInput = currentPano.getVideoInput(id);
    const ToneCurveByTime& byTime = kv.second;

    for (auto reverseIt = byTime.rbegin(); reverseIt != byTime.rend(); ++reverseIt) {
      mtime_t time = reverseIt->first;

      if (time <= currentStitcherDate) {
        const Metadata::ToneCurve& curve = reverseIt->second;

        if (inputDefNeedsToneCurveUpdate(videoInput, curve)) {
          getNewPano().getVideoInput(id).setValueBasedResponseCurve(curve.curveAsArray());
        }

        break;
      }
    }
  }
}

std::unique_ptr<Core::PanoDefinition> MetadataProcessor::createUpdatedPano(const Input::MetadataChunk& metadata,
                                                                           const Core::PanoDefinition& currentPano,
                                                                           FrameRate frameRate,
                                                                           frameid_t currentStitchingFrame) {
  std::unique_ptr<Core::PanoDefinition> pano;

  if (metadata.hasExposureData()) {
    pano.reset(currentPano.clone());

    pruneExposureCurves(*pano, currentStitchingFrame);
    insertExposureMetadata(metadata.exposure, frameRate, *pano, currentStitchingFrame);

    pruneToneCurves(currentStitchingFrame, frameRate);
    insertToneCurveMetadata(metadata.toneCurve);
  }

  createUpdatedPanoForCurrentFrame(pano, currentPano, frameRate, currentStitchingFrame);
  return pano;
}

}  // namespace Exposure
}  // namespace VideoStitch
