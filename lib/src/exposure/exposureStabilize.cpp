// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "exposureStabilize.hpp"

#include "core/controller.hpp"
#include "core/photoTransform.hpp"

#include "libvideostitch/controller.hpp"
#include "libvideostitch/curves.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/profile.hpp"
//#define RANSAC_EXPERIMENT
#ifdef RANSAC_EXPERIMENT
#include "ransac.hpp"
#else
#include "util/lmfit/lmmin.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <random>

#define GLOBAL_EXPOSURE_REFERENCE

namespace VideoStitch {
namespace Util {

namespace {
/**
 * Updates first and last splines by adding a point.
 * @param first First spline.
 * @param last Last spline
 * @param time Time where to add @a value
 * @param value Value to add.
 */
void updateSplines(Core::Spline*& first, Core::Spline*& last, int time, double value) {
  if (last) {
    Core::Spline* spline = last->cubicTo(time, value);
    last = spline;
  } else {
    assert(first == NULL);
    Core::Spline* spline = Core::Spline::point(time, value);
    first = spline;
    last = spline;
  }
}
}  // namespace

class ExposureStabilizationProblemBase::ParameterSet {
 public:
  /**
   * @param pano Panorama definition
   */
  ParameterSet(const Core::PanoDefinition& pano, readerid_t anchor)
      : pano(pano),
        anchoringType(anchor < 0 || anchor >= pano.numInputs() ? CenterParamsAnchoring : SingleInputAnchoring),
        videoAnchor(convertAnchorInputIndexToVideoInputIndex(pano, anchoringType, anchor)),
        numFloatingInputs(pano.numVideoInputs() - 1),
        evFirstSplines(pano.numVideoInputs()),
        evLastSplines(pano.numVideoInputs()) {}

  virtual ~ParameterSet() {}

  videoreaderid_t getVideoParamIndex(videoreaderid_t k) const {
    assert(k != videoAnchor);
    if (k < videoAnchor) {
      return k;
    } else {
      return k - 1;
    }
  }

  videoreaderid_t getVideoInputIndex(videoreaderid_t paramK) const {
    assert(paramK < numFloatingInputs);
    if (paramK < videoAnchor) {
      return paramK;
    } else {
      return paramK + 1;
    }
  }

  virtual int numParams() const = 0;
  virtual void computeInitialGuess(std::vector<double>& params, int time) const = 0;

  virtual float3 getVideoColorMult(const double* params, videoreaderid_t k, int time) const = 0;

  const Core::PanoDefinition& getPano() const { return pano; }

  virtual void constantControlPoint(std::vector<double>& params) = 0;

  virtual void saveControlPoint(std::vector<double>& params, int time) = 0;

  virtual bool injectSavedControlPoints(Core::PanoDefinition* pano, bool preserveOutside, int firstFrame,
                                        int lastFrame) const = 0;

  videoreaderid_t getVideoAnchor() const { return videoAnchor; }
  ExposureStabilizationProblemBase::AnchoringType getAnchoringType() const { return anchoringType; }

  virtual double getMinValidParamValue() const = 0;
  virtual double getMaxValidParamValue() const = 0;

  bool isValid(const double* params) const {
    for (int par = 0; par < numParams(); par++) {
      if (params[par] < getMinValidParamValue() || params[par] > getMaxValidParamValue()) {
        return false;
      }
    }
    return true;
  }

 protected:
  template <Core::Curve* (Core::InputDefinition::*curveDisplacer)(Core::Curve*)>
  void createBoundaryKeyframes(Core::PanoDefinition* pano, int firstFrame, int lastFrame) const {
    for (videoreaderid_t k = 0; k < pano->numVideoInputs(); ++k) {
      Core::Curve* tmpCurve = new Core::Curve(0.0);
      tmpCurve = (pano->getVideoInput(k).*curveDisplacer)(tmpCurve);
      if (firstFrame > 0) {
        tmpCurve->splitAt(firstFrame - 1);
      }
      tmpCurve->splitAt(lastFrame + 1);
      delete (pano->getVideoInput(k).*curveDisplacer)(tmpCurve);
    }
  }

  bool injectEvControlPoints(Core::PanoDefinition* pano) const {
    for (videoreaderid_t k = 0; k < pano->numVideoInputs(); ++k) {
      if (!evFirstSplines[k]) {
        return false;
      }
    }
    for (videoreaderid_t k = 0; k < pano->numVideoInputs(); ++k) {
      Core::Curve* curve = new Core::Curve(evFirstSplines[k]);
      curve->extend(&pano->getVideoInput(k).getExposureValue());
      pano->getVideoInput(k).replaceExposureValue(curve);
    }
    return true;
  }

  static videoreaderid_t convertAnchorInputIndexToVideoInputIndex(
      const Core::PanoDefinition& pano, const ExposureStabilizationProblemBase::AnchoringType anchoringType,
      readerid_t anchor) {
    anchor = anchor < 0 || anchor >= (int)pano.numInputs() ? 0 : anchor;
    if (anchoringType == CenterParamsAnchoring) {
      return 0;
    } else {
      return pano.convertInputIndexToVideoInputIndex(anchor);
    }
  }

  const Core::PanoDefinition& pano;
  const ExposureStabilizationProblemBase::AnchoringType anchoringType;
  const videoreaderid_t videoAnchor;
  const videoreaderid_t numFloatingInputs;
  std::vector<Core::Spline*> evFirstSplines;
  std::vector<Core::Spline*> evLastSplines;
};

/**
 * ParameterSet for ev-only exposure stabilization.
 */
class EvParameterSetImpl : public ExposureStabilizationProblemBase::ParameterSet {
 public:
  EvParameterSetImpl(const Core::PanoDefinition& pano, int anchor) : ParameterSet(pano, anchor) {}

  ~EvParameterSetImpl() {}

  int numParams() const { return (int)numFloatingInputs; }

  void computeInitialGuess(std::vector<double>& params, int time) const {
    // The parameter is the ev value.
    params.clear();
    params.reserve(numParams());
    for (videoreaderid_t paramK = 0; paramK < numFloatingInputs; ++paramK) {
      params.push_back(pano.getVideoInput(getVideoInputIndex(paramK)).getExposureValue().at(time));
    }
  }

  void constantControlPoint(std::vector<double>& params) {
    for (videoreaderid_t paramK = 0; paramK < numFloatingInputs; ++paramK) {
      const videoreaderid_t k = getVideoInputIndex(paramK);
      evFirstSplines[k] = Core::Spline::point(0, params[paramK]);
      evLastSplines[k] = evFirstSplines[k];
    }
    evFirstSplines[videoAnchor] = Core::Spline::point(0, getAnchorEv(params.data(), 0));
    evLastSplines[videoAnchor] = evFirstSplines[videoAnchor];
  }

  void saveControlPoint(std::vector<double>& params, int time) {
    for (videoreaderid_t paramK = 0; paramK < numFloatingInputs; ++paramK) {
      const videoreaderid_t k = getVideoInputIndex(paramK);
      updateSplines(evFirstSplines[k], evLastSplines[k], time, params[paramK]);
    }
    updateSplines(evFirstSplines[videoAnchor], evLastSplines[videoAnchor], time, getAnchorEv(params.data(), time));
  }

  bool injectSavedControlPoints(Core::PanoDefinition* pano, bool preserveOutside, int firstFrame, int lastFrame) const {
    if (preserveOutside) {
      createBoundaryKeyframes<&Core::InputDefinition::displaceExposureValue>(pano, firstFrame, lastFrame);
    }
    return injectEvControlPoints(pano);
  }

  double getMinValidParamValue() const { return -12.0; }

  double getMaxValidParamValue() const { return 12.0; }

 private:
  /**
   * Returns the ev for the anchor given the param values.
   */
  double getAnchorEv(const double* params, int time) const {
    switch (anchoringType) {
      case ExposureStabilizationProblemBase::SingleInputAnchoring:
        return pano.getVideoInput(videoAnchor).getExposureValue().at(time);
      case ExposureStabilizationProblemBase::CenterParamsAnchoring:
        double ev = 0.0;
        // Anchor parameters so that the sum of parameters is zero.
        for (int i = 0; i < numParams(); ++i) {
          ev -= params[i];
        }
        return ev;
    }
    assert(false);
    return 0.0;
  }

  float3 getVideoColorMult(const double* params, videoreaderid_t k, int time) const {
    const double ev = (k == videoAnchor) ? getAnchorEv(params, time) : params[getVideoParamIndex(k)];
    // The parameter is the ev value.
    return Core::PhotoTransform::ColorCorrectionParams((float)ev, pano.getVideoInput(k).getRedCB().at(time),
                                                       pano.getVideoInput(k).getGreenCB().at(time),
                                                       pano.getVideoInput(k).getBlueCB().at(time))
#ifdef GLOBAL_EXPOSURE_REFERENCE
        .computeColorMultiplier(pano.getExposureValue().at(time), pano.getRedCB().at(time), pano.getGreenCB().at(time),
                                pano.getBlueCB().at(time));
#else
        .computeColorMultiplier(0.0, 1.0, 1.0, 1.0);
#endif
  }
};

/**
 * ParameterSet for rgb exposure stabilization.
 */
class WBParameterSetImpl : public ExposureStabilizationProblemBase::ParameterSet {
 public:
  WBParameterSetImpl(const Core::PanoDefinition& pano, int anchor)
      : ParameterSet(pano, anchor),
        redCBFirstSplines(pano.numVideoInputs()),
        redCBLastSplines(pano.numVideoInputs()),
        blueCBFirstSplines(pano.numVideoInputs()),
        blueCBLastSplines(pano.numVideoInputs()) {}

  ~WBParameterSetImpl() {}

  int numParams() const { return (int)(3 * numFloatingInputs); }

  void computeInitialGuess(std::vector<double>& params, int time) const {
    // The parameters are color multipliers.
    params.clear();
    params.reserve(numParams());
    for (videoreaderid_t paramK = 0; paramK < numFloatingInputs; ++paramK) {
      const Core::InputDefinition& im = pano.getVideoInput(getVideoInputIndex(paramK));
      const float3 mult =
          Core::PhotoTransform::ColorCorrectionParams(im.getExposureValue().at(time), im.getRedCB().at(time),
                                                      im.getGreenCB().at(time), im.getBlueCB().at(time))
#ifdef GLOBAL_EXPOSURE_REFERENCE
              .computeColorMultiplier(pano.getExposureValue().at(time), pano.getRedCB().at(time),
                                      pano.getGreenCB().at(time), pano.getBlueCB().at(time));
#else
              .computeColorMultiplier(0.0, 1.0, 1.0, 1.0);
#endif
      params.push_back(mult.x);
      params.push_back(mult.y);
      params.push_back(mult.z);
    }
  }

  void constantControlPoint(std::vector<double>& params) {
    for (videoreaderid_t k = 0; k < pano.numVideoInputs(); ++k) {
      float3 mult;
      if (k == videoAnchor) {
        mult = getAnchorMult(params.data(), 0);
      } else {
        const videoreaderid_t paramK = getVideoParamIndex(k);
        mult.x = (float)params[3 * paramK];
        mult.y = (float)params[3 * paramK + 1];
        mult.z = (float)params[3 * paramK + 2];
      }
      Core::PhotoTransform::ColorCorrectionParams colorCorr =
          Core::PhotoTransform::ColorCorrectionParams::canonicalFromMultiplier(
              mult,
#ifdef GLOBAL_EXPOSURE_REFERENCE
              pano.getExposureValue().at(0), pano.getRedCB().at(0), pano.getGreenCB().at(0), pano.getBlueCB().at(0));
#else
              0.0, 1.0, 1.0, 1.0);
#endif
      evFirstSplines[k] = Core::Spline::point(0, colorCorr.ev);
      evLastSplines[k] = evFirstSplines[k];
      redCBFirstSplines[k] = Core::Spline::point(0, colorCorr.redCB);
      redCBLastSplines[k] = redCBFirstSplines[k];
      blueCBFirstSplines[k] = Core::Spline::point(0, colorCorr.blueCB);
      blueCBLastSplines[k] = blueCBFirstSplines[k];
    }
  }

  void saveControlPoint(std::vector<double>& params, int time) {
    for (videoreaderid_t k = 0; k < pano.numVideoInputs(); ++k) {
      float3 mult;
      if (k == videoAnchor) {
        mult = getAnchorMult(params.data(), time);
      } else {
        const videoreaderid_t paramK = getVideoParamIndex(k);
        mult.x = (float)params[3 * paramK];
        mult.y = (float)params[3 * paramK + 1];
        mult.z = (float)params[3 * paramK + 2];
      }
      Core::PhotoTransform::ColorCorrectionParams colorCorr =
          Core::PhotoTransform::ColorCorrectionParams::canonicalFromMultiplier(
              mult,
#ifdef GLOBAL_EXPOSURE_REFERENCE
              pano.getExposureValue().at(time), pano.getRedCB().at(time), pano.getGreenCB().at(time),
              pano.getBlueCB().at(time));
#else
              0.0, 1.0, 1.0, 1.0);
#endif
      updateSplines(evFirstSplines[k], evLastSplines[k], time, colorCorr.ev);
      updateSplines(redCBFirstSplines[k], redCBLastSplines[k], time, colorCorr.redCB);
      updateSplines(blueCBFirstSplines[k], blueCBLastSplines[k], time, colorCorr.blueCB);
    }
  }

  bool injectSavedControlPoints(Core::PanoDefinition* pano, bool preserveOutside, int firstFrame, int lastFrame) const {
    if (preserveOutside) {
      createBoundaryKeyframes<&Core::InputDefinition::displaceRedCB>(pano, firstFrame, lastFrame);
      createBoundaryKeyframes<&Core::InputDefinition::displaceBlueCB>(pano, firstFrame, lastFrame);
    }
    for (videoreaderid_t k = 0; k < pano->numVideoInputs(); ++k) {
      if (!(redCBFirstSplines[k] && blueCBFirstSplines[k])) {
        return false;
      }
    }
    if (!injectEvControlPoints(pano)) {
      return false;
    }
    for (videoreaderid_t k = 0; k < pano->numVideoInputs(); ++k) {
      Core::Curve* curve = new Core::Curve(redCBFirstSplines[k]);
      curve->extend(&pano->getVideoInput(k).getRedCB());
      pano->getVideoInput(k).replaceRedCB(curve);
      curve = new Core::Curve(blueCBFirstSplines[k]);
      curve->extend(&pano->getVideoInput(k).getBlueCB());
      pano->getVideoInput(k).replaceBlueCB(curve);
    }
    return true;
  }

  double getMinValidParamValue() const { return 1.0 / 128.0; }

  double getMaxValidParamValue() const { return 128.0; }

 private:
  /**
   * Returns the multiplier for the anchor given the param values.
   */
  float3 getAnchorMult(const double* params, int time) const {
    switch (anchoringType) {
      case ExposureStabilizationProblemBase::SingleInputAnchoring:
        return Core::PhotoTransform::ColorCorrectionParams(pano.getVideoInput(videoAnchor).getExposureValue().at(time),
                                                           pano.getVideoInput(videoAnchor).getRedCB().at(time),
                                                           pano.getVideoInput(videoAnchor).getGreenCB().at(time),
                                                           pano.getVideoInput(videoAnchor).getBlueCB().at(time))
#ifdef GLOBAL_EXPOSURE_REFERENCE
            .computeColorMultiplier(pano.getExposureValue().at(time), pano.getRedCB().at(time),
                                    pano.getGreenCB().at(time), pano.getBlueCB().at(time));
#else
            .computeColorMultiplier(0.0, 1.0, 1.0, 1.0);
#endif
      case ExposureStabilizationProblemBase::CenterParamsAnchoring:
        double logMult[] = {0.0, 0.0, 0.0};
        // Anchor parameters so that the sum of log2(parameters) is zero.
        for (videoreaderid_t i = 0; i < numFloatingInputs; ++i) {
          logMult[0] += log2(params[3 * i]);
          logMult[1] += log2(params[3 * i + 1]);
          logMult[2] += log2(params[3 * i + 2]);
        }
        return make_float3((float)pow(2.0, -logMult[0]), (float)pow(2.0, -logMult[1]), (float)pow(2.0, -logMult[2]));
    }
    assert(false);
    return make_float3(0.0f, 0.0f, 0.0f);
  }

  float3 getVideoColorMult(const double* params, videoreaderid_t k, int time) const {
    if (k == videoAnchor) {
      return getAnchorMult(params, time);
    } else {
      // The parameters are the multipliers themselves.
      const videoreaderid_t paramK = getVideoParamIndex(k);
      return make_float3((float)params[3 * paramK], (float)params[3 * paramK + 1], (float)params[3 * paramK + 2]);
    }
  }

  std::vector<Core::Spline*> redCBFirstSplines;
  std::vector<Core::Spline*> redCBLastSplines;
  std::vector<Core::Spline*> blueCBFirstSplines;
  std::vector<Core::Spline*> blueCBLastSplines;
};

ExposureStabilizationProblemBase::ExposureStabilizationProblemBase(const Core::PanoDefinition& pano, int anchor,
                                                                   ParameterSetType parameterSetType)
    : parameterSet(parameterSetType == EvParameterSet
                       ? static_cast<ParameterSet*>(new EvParameterSetImpl(pano, anchor))
                       : static_cast<ParameterSet*>(new WBParameterSetImpl(pano, anchor))),
      time(0) {}

ExposureStabilizationProblemBase::~ExposureStabilizationProblemBase() {}

int ExposureStabilizationProblemBase::numParams() const { return parameterSet->numParams(); }

void ExposureStabilizationProblemBase::computeInitialGuess(std::vector<double>& params) const {
  return parameterSet->computeInitialGuess(params, time);
}

const Core::PanoDefinition& ExposureStabilizationProblemBase::getPano() const { return parameterSet->getPano(); }

float3 ExposureStabilizationProblemBase::getVideoColorMult(const double* params, videoreaderid_t k) const {
  return parameterSet->getVideoColorMult(params, k, time);
}

videoreaderid_t ExposureStabilizationProblemBase::getVideoParamIndex(int k) const {
  return parameterSet->getVideoParamIndex(k);
}

videoreaderid_t ExposureStabilizationProblemBase::getVideoInputIndex(int paramK) const {
  return parameterSet->getVideoInputIndex(paramK);
}

void ExposureStabilizationProblemBase::constantControlPoint(std::vector<double>& params) {
  parameterSet->constantControlPoint(params);
}

void ExposureStabilizationProblemBase::saveControlPoint(std::vector<double>& params) {
  parameterSet->saveControlPoint(params, time);
}

bool ExposureStabilizationProblemBase::injectSavedControlPoints(Core::PanoDefinition* pano, bool preserveOutside,
                                                                int firstFrame, int lastFrame) {
  return parameterSet->injectSavedControlPoints(pano, preserveOutside, firstFrame, lastFrame);
}

videoreaderid_t ExposureStabilizationProblemBase::getVideoAnchor() const {
  assert(getAnchoringType() == SingleInputAnchoring);
  return parameterSet->getVideoAnchor();
}

ExposureStabilizationProblemBase::AnchoringType ExposureStabilizationProblemBase::getAnchoringType() const {
  return parameterSet->getAnchoringType();
}

double ExposureStabilizationProblemBase::getMinValidParamValue() const { return parameterSet->getMinValidParamValue(); }

bool ExposureStabilizationProblemBase::isValid(const double* params) const { return parameterSet->isValid(params); }

double ExposureStabilizationProblemBase::getMaxValidParamValue() const { return parameterSet->getMaxValidParamValue(); }

}  // namespace Util
}  // namespace VideoStitch
