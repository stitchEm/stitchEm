// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <gpu/buffer.hpp>
#include "gpu/surface.hpp"
#include <gpu/stream.hpp>

#include <stdint.h>
#include "libvideostitch/inputDef.hpp"

struct TransformPhotoParam;

namespace VideoStitch {
namespace Core {

Status linearPhotoCorrection(GPU::Surface& buffer, const int width, const int height, const float rMult,
                             const float gMult, const float bMult, const float vigCenterX, const float vigCenterY,
                             const float inverseDemiDiagonalSquared, const float vigCoeff0, const float vigCoeff1,
                             const float vigCoeff2, const float vigCoeff3, const TransformPhotoParam& photoParam,
                             GPU::Stream stream);

Status gammaPhotoCorrection(GPU::Surface& buffer, const int width, const int height, const float rMult,
                            const float gMult, const float bMult, const float vigCenterX, const float vigCenterY,
                            const float inverseDemiDiagonalSquared, const float vigCoeff0, const float vigCoeff1,
                            const float vigCoeff2, const float vigCoeff3, const TransformPhotoParam& photoParam,
                            GPU::Stream stream);

Status emorPhotoCorrection(GPU::Surface& buffer, const int width, const int height, const float rMult,
                           const float gMult, const float bMult, const float vigCenterX, const float vigCenterY,
                           const float inverseDemiDiagonalSquared, const float vigCoeff0, const float vigCoeff1,
                           const float vigCoeff2, const float vigCoeff3, const TransformPhotoParam& photoParam,
                           GPU::Stream stream);
}  // namespace Core
}  // namespace VideoStitch
