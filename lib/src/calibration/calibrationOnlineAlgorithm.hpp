// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "calibrationAlgorithmBase.hpp"

#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/config.hpp"
#include "libvideostitch/panoDef.hpp"

namespace VideoStitch {
namespace Calibration {

/**
@brief Instance of OnlineAlgorithm
*/
class VS_EXPORT CalibrationOnlineAlgorithm : public Util::OnlineAlgorithm, public CalibrationAlgorithmBase {
 public:
  static const char* docString;

  explicit CalibrationOnlineAlgorithm(const Ptv::Value* config);
  virtual ~CalibrationOnlineAlgorithm();

  Potential<Ptv::Value> onFrame(Core::PanoDefinition&, std::vector<std::pair<videoreaderid_t, GPU::Surface&>>& frames,
                                mtime_t, FrameRate, Util::OpaquePtr** ctx) override;

 private:
  /*Load images in Host memory*/
  Status retrieveImages(RigCvImages& rig, const Core::PanoDefinition& pano,
                        const std::vector<std::pair<videoreaderid_t, GPU::Surface&>>& frames) const;
};
}  // namespace Calibration
}  // namespace VideoStitch
