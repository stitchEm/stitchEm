// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "mergerMaskConfig.hpp"
#include "mergerMask.hpp"

#include "gpu/uniqueBuffer.hpp"

#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/status.hpp"
#include "libvideostitch/stereoRigDef.hpp"

#include <vector>
#include <unordered_map>
#include <memory>

namespace VideoStitch {
namespace Core {
class StereoRigDefinition;
}

namespace MergerMask {

/**
 * @brief Optimize for the mask and blending order of the output panorama
 */
class VS_EXPORT MergerMaskAlgorithm : public Util::Algorithm {
 public:
  static const char* docString;
  explicit MergerMaskAlgorithm(const Ptv::Value* config);
  virtual ~MergerMaskAlgorithm();

 public:
  /**
   * Specialization of Algorithm::apply
   * @param pano the input/output panorama definition
   * @param progress a callback object to give information about the progress of calibration algorithm
   * @param ctx An optional context object instance
   * @return a description of the result
   */
  virtual Potential<Ptv::Value> apply(Core::PanoDefinition* pano, ProgressReporter* progress,
                                      Util::OpaquePtr** ctx = NULL) const override;

 protected:
  /**
   * @param rig the output images
   * @param pano the input panorama definition
   * @param progress the progress reporter
   * @return a status
   */
 private:
  mutable std::map<readerid_t, Input::VideoReader*> readers;
  MergerMaskConfig mergerMaskConfig;
  const Core::StereoRigDefinition* const rigDef;
};

}  // namespace MergerMask
}  // namespace VideoStitch
