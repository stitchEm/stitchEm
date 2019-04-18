// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "backend/common/core/transformPhotoParam.hpp"
#include "gpu/vectorTypes.hpp"

#include "libvideostitch/preprocessor.hpp"

#include <string>

namespace VideoStitch {

class ThreadSafeOstream;

namespace Core {

class InputDefinition;
class DevicePhotoTransform;

/**
 * @brief A processor that does pre-stitching photo correction.
 */
class PhotoCorrPreProcessor : public PreProcessor {
 public:
  PhotoCorrPreProcessor(const InputDefinition& inputDef, const float3& colorMult,
                        const DevicePhotoTransform& transform);

  Status process(frameid_t frame, GPU::Surface& devBuffer, int64_t width, int64_t height, readerid_t inputId,
                 GPU::Stream& stream) const;

  void getDisplayName(std::ostream& os) const;

 private:
  const InputDefinition& inputDef;
  // Photometry parameters
  const float rMult;                       // Red multiplier
  const float gMult;                       // Green multiplier
  const float bMult;                       // Blue multiplier
  const float inverseDemiDiagonalSquared;  // inverse of the image demi-diagonal length
  const TransformPhotoParam devicePhotoParam;
};

}  // namespace Core
}  // namespace VideoStitch
