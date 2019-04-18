// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "autoCropConfig.hpp"

#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/status.hpp"

#include <vector>
#include <unordered_map>
#include <memory>

namespace VideoStitch {
namespace AutoCrop {

/**
 * @brief Auto-crop for circular fisheye camera
 */
class VS_EXPORT AutoCropAlgorithm : public Util::Algorithm {
 public:
  static const char* docString;
  explicit AutoCropAlgorithm(const Ptv::Value* config);
  virtual ~AutoCropAlgorithm();

 public:
  /**
  Specialization of Algorithm::apply
  @param pano the input/output panorama definition
  @param progress a callback object to give information about the progress of calibration algorithm
  @param ctx An optional context object instance
  @return a description of the result
  */
  virtual Potential<Ptv::Value> apply(Core::PanoDefinition* pano, ProgressReporter* progress,
                                      Util::OpaquePtr** ctx = nullptr) const override;

 private:
  AutoCropConfig autoCropConfig;
};

}  // namespace AutoCrop
}  // namespace VideoStitch
