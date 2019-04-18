// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "photoCorrProcessor.hpp"

#include "gpu/buffer.hpp"
#include "gpu/processors/photoCorr.hpp"
#include "core/photoTransform.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/inputDef.hpp"

#include <iostream>

namespace VideoStitch {
namespace Core {

PhotoCorrPreProcessor::PhotoCorrPreProcessor(const InputDefinition& inputDef, const float3& colorMult,
                                             const DevicePhotoTransform& transform)
    : inputDef(inputDef),
      rMult(colorMult.x),
      gMult(colorMult.y),
      bMult(colorMult.z),
      inverseDemiDiagonalSquared((float)transform.inverseDemiDiagonalSquared),
      devicePhotoParam(transform.getDevicePhotoParam()) {}

Status PhotoCorrPreProcessor::process(frameid_t /*frame*/, GPU::Surface& devBuffer, int64_t /*width*/,
                                      int64_t /*height*/, readerid_t /*inputId*/, GPU::Stream& stream) const {
  switch (inputDef.getPhotoResponse()) {
    case InputDefinition::PhotoResponse::LinearResponse:
      return linearPhotoCorrection(devBuffer, (int)inputDef.getWidth(), (int)inputDef.getHeight(), rMult, gMult, bMult,
                                   (float)inputDef.getVignettingCenterX(), (float)inputDef.getVignettingCenterY(),
                                   (float)inverseDemiDiagonalSquared, (float)inputDef.getVignettingCoeff0(),
                                   (float)inputDef.getVignettingCoeff1(), (float)inputDef.getVignettingCoeff2(),
                                   (float)inputDef.getVignettingCoeff3(), devicePhotoParam, stream);
    case InputDefinition::PhotoResponse::GammaResponse:
      return gammaPhotoCorrection(devBuffer, (int)inputDef.getWidth(), (int)inputDef.getHeight(), rMult, gMult, bMult,
                                  (float)inputDef.getVignettingCenterX(), (float)inputDef.getVignettingCenterY(),
                                  (float)inverseDemiDiagonalSquared, (float)inputDef.getVignettingCoeff0(),
                                  (float)inputDef.getVignettingCoeff1(), (float)inputDef.getVignettingCoeff2(),
                                  (float)inputDef.getVignettingCoeff3(), devicePhotoParam, stream);
    case InputDefinition::PhotoResponse::EmorResponse:
    case InputDefinition::PhotoResponse::InvEmorResponse:
      return emorPhotoCorrection(devBuffer, (int)inputDef.getWidth(), (int)inputDef.getHeight(), rMult, gMult, bMult,
                                 (float)inputDef.getVignettingCenterX(), (float)inputDef.getVignettingCenterY(),
                                 (float)inverseDemiDiagonalSquared, (float)inputDef.getVignettingCoeff0(),
                                 (float)inputDef.getVignettingCoeff1(), (float)inputDef.getVignettingCoeff2(),
                                 (float)inputDef.getVignettingCoeff3(), devicePhotoParam, stream);
    case InputDefinition::PhotoResponse::CurveResponse:
      return Status{Origin::PreProcessor, ErrType::UnsupportedAction,
                    "Photo correction pre-processing with a curve camera response is not implemented"};
  }
  assert(false);
  return Status{Origin::PreProcessor, ErrType::ImplementationError, "Invalid photo response"};
}

void PhotoCorrPreProcessor::getDisplayName(std::ostream& os) const { os << "Procedural(P): PhotoCorr"; }

}  // namespace Core
}  // namespace VideoStitch
