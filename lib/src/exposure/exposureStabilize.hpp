// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/config.hpp"
#include "libvideostitch/panoDef.hpp"
#include "util/lmfit/lmmin.hpp"
#include "gpu/vectorTypes.hpp"

#include <memory>

namespace VideoStitch {
namespace Util {
/**
 * A base problem for exposure compensation.
 */
class ExposureStabilizationProblemBase : public SolverProblem {
 public:
  class ParameterSet;

  /**
   * The parameter set to optimize: Only Ev or WB.
   */
  enum ParameterSetType {
    EvParameterSet,
    WBParameterSet,
  };

  /**
   * The type of anchoring:
   */
  enum AnchoringType {
    SingleInputAnchoring,
    CenterParamsAnchoring,
  };

  /**
   * ctor.
   * @param pano Pano definition
   * @param anchor anchor input, or -1
   * @param parameterSetType What parameters to stabilize
   */
  ExposureStabilizationProblemBase(const Core::PanoDefinition& pano, int anchor, ParameterSetType parameterSetType);

  virtual ~ExposureStabilizationProblemBase();

  /**
   * Returns the anchoring type.
   */
  AnchoringType getAnchoringType() const;

  /**
   * Returns the video input id of the anchor. Only meaningful if getAnchoringType() == SingleInputAnchoring.
   */
  videoreaderid_t getVideoAnchor() const;

  int numParams() const;

  virtual int getNumInputSamples() const = 0;

  int getNumValuesPerSample() const {
    return 3;  // (R, G, B) for each sample
  }

  /**
   * k is a video input index
   */
  float3 getVideoColorMult(const double* params, videoreaderid_t k) const;

  int getNumAdditionalValues() const { return 0; }

  /**
   * Returns the minimum acceptable parameter value.
   */
  double getMinValidParamValue() const;

  /**
   * Returns the maximum acceptable parameter value.
   */
  double getMaxValidParamValue() const;

  /**
   * Returns true if the parameter set consists of only valid values.
   */
  bool isValid(const double* params) const;

  /**
   * Sets the time to be used when querying time-dependant panorama or input properties.
   * @param newTime The time to set.
   */
  void setTime(int newTime) { time = newTime; }

  /**
   * Computes the initial guess for parameters.
   * @param params Vector of parameters.
   */
  void computeInitialGuess(std::vector<double>& params) const;

  /**
   * Returns the pano definition.
   */
  const Core::PanoDefinition& getPano() const;

  /**
   * Computes the index of video input @a k in the renumbered video inputs (without the anchor).
   * @param k Pano video input id.
   * @returns params video input id.
   */
  videoreaderid_t getVideoParamIndex(int k) const;

  /**
   * Computes the index of video input @a k from the renumbered video parameter id..
   * @param paramK params input id.
   * @returns pano video input id.
   */
  videoreaderid_t getVideoInputIndex(int paramK) const;

  /**
   * Save a constant curve point.
   * @param params Vector of parameters.
   */
  void constantControlPoint(std::vector<double>& params);

  /**
   * Save a curve point at the current time.
   * @param params Vector of parameters.
   */
  void saveControlPoint(std::vector<double>& params);

  /**
   * Inject the new splines in place of the old ones.
   * @param pano Output pano def.
   * @param preserveOutside If true, add one keyframe on each side to preserve exterior values.
   * @param firstFrame First stabilizatiuon frame.
   * @param lastFrame Last stabilizatiuon frame.
   * @return false if there are no saved control points.
   */
  bool injectSavedControlPoints(Core::PanoDefinition* pano, bool preserveOutside, int firstFrame, int lastFrame);

 private:
  const std::unique_ptr<ParameterSet> parameterSet;
  int time;
};

}  // namespace Util
}  // namespace VideoStitch
