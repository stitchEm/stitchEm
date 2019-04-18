// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/config.hpp"
#include "libvideostitch/inputDef.hpp"

#include <gpu/hostBuffer.hpp>

#include <unordered_map>
#include <functional>

namespace VideoStitch {
namespace Util {

class SampledExposureStabilizationProblem;

/**
 * An algorithm that stabilizes the exposure/WB spacially and temporally.
 */
class SampledStabilizationBase {
 public:
  explicit SampledStabilizationBase(const Ptv::Value* config);
  virtual ~SampledStabilizationBase() {}

 protected:
  void sample(Core::PanoDefinition*, std::vector<GPU::HostBuffer<uint32_t>>& frames,
              SampledExposureStabilizationProblem&) const;

  Potential<SampledExposureStabilizationProblem> createProblem(Core::PanoDefinition* pano) const;

  int maxSampledPoints;
  int minPointsPerInput;
  int neighbourhoodSize;
  int anchor;
  bool stabilizeWB;
};

class SampledStabilizationOnlineAlgorithm : public OnlineAlgorithm, public SampledStabilizationBase {
 public:
  /**
   * The algo docstring.
   */
  static const char* docString;
  explicit SampledStabilizationOnlineAlgorithm(const Ptv::Value* config);
  virtual ~SampledStabilizationOnlineAlgorithm() {}

  Potential<Ptv::Value> onFrame(Core::PanoDefinition&, std::vector<std::pair<videoreaderid_t, GPU::Surface&>>& frames,
                                mtime_t date, FrameRate frameRate, Util::OpaquePtr** ctx) override;

 private:
  Status processFrames(const std::vector<std::pair<videoreaderid_t, GPU::Surface&>>& frames, Core::PanoDefinition& pano,
                       const Potential<SampledExposureStabilizationProblem>& problem);

  std::unordered_map<std::string, std::vector<Core::Spline*>> preserveCurves(const Core::PanoDefinition& panorama,
                                                                             frameid_t frame);
  void updateInputCurves(Core::PanoDefinition& panorama,
                         std::unordered_map<std::string, std::vector<Core::Spline*>> preservedCurves,
                         frameid_t algorithmFinishFrame, frameid_t interpolationFinishFrame);

  static const std::unordered_map<std::string, std::pair<const Core::Curve& (Core::InputDefinition::*)(void)const,
                                                         void (Core::InputDefinition::*)(Core::Curve*)>>
      functionMap;
  int interpolationFixationFrames;
  mtime_t interpolationDuration;                         // microseconds
  static const mtime_t InterpolationDurationMultiplier;  // microseconds
};

class SampledStabilizationAlgorithm : public Algorithm, public SampledStabilizationBase {
 public:
  /**
   * The algo docstring.
   */
  static const char* docString;
  explicit SampledStabilizationAlgorithm(const Ptv::Value* config);
  virtual ~SampledStabilizationAlgorithm() {}

  Potential<Ptv::Value> apply(Core::PanoDefinition* pano, ProgressReporter* progress,
                              OpaquePtr** = NULL) const override;

 private:
  int firstFrame;
  int lastFrame;
  int timeStep;
  bool temporalStabilization;
  bool preserveOutside;
  bool returnPointSet;
};

}  // namespace Util
}  // namespace VideoStitch
