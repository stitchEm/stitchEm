// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/processors/photoCorr.hpp"

#include "backend/common/core/transformPhotoParam.hpp"
#include "../context.hpp"
#include "../kernel.hpp"
#include "../surface.hpp"

#include <gpu/util.hpp>

#include "libvideostitch/inputDef.hpp"

namespace VideoStitch {
namespace Core {

template <InputDefinition::PhotoResponse>
const char* photoCorrKernelName();

template <>
const char* photoCorrKernelName<InputDefinition::PhotoResponse::LinearResponse>() {
  return KERNEL_STR(preStitchPhotoCorrectionKernel_linear);
}

template <>
const char* photoCorrKernelName<InputDefinition::PhotoResponse::EmorResponse>() {
  return KERNEL_STR(preStitchPhotoCorrectionKernel_emor);
}

template <>
const char* photoCorrKernelName<InputDefinition::PhotoResponse::GammaResponse>() {
  return KERNEL_STR(preStitchPhotoCorrectionKernel_gamma);
}

template <InputDefinition::PhotoResponse photoCorrStr>
Status launchPhotoKernel(GPU::Surface& /*buffer*/, const int /*width*/, const int /*height*/, const float /*rMult*/,
                         const float /*gMult*/, const float /*bMult*/, const float /*vigCenterX*/,
                         const float /*vigCenterY*/, const float /*inverseDemiDiagonalSquared*/,
                         const float /*vigCoeff0*/, const float /*vigCoeff1*/, const float /*vigCoeff2*/,
                         const float /*vigCoeff3*/, const float /*photoParamFloat*/, const cl_mem /*photoParamArray*/,
                         GPU::Stream /*stream*/) {
  //  std::string kernelName=photoCorrKernelName<photoCorrStr> ();
  //  auto kernel2D = GPU::Kernel::get(PROGRAM(photoCorr), kernelName).setup2D(stream, width, height);
  //  return kernel2D.enqueueWithKernelArgs(buffer.get(), width, height, rMult, gMult, bMult, vigCenterX, vigCenterY,
  //  inverseDemiDiagonalSquared, vigCoeff0, vigCoeff1, vigCoeff2, vigCoeff3, photoParamFloat, photoParamArray);
  return {Origin::GPU, ErrType::UnsupportedAction, "[OpenCL] photo correction on the inputs is not supported"};
}

Status linearPhotoCorrection(GPU::Surface& buffer, const int width, const int height, const float rMult,
                             const float gMult, const float bMult, const float vigCenterX, const float vigCenterY,
                             const float inverseDemiDiagonalSquared, const float vigCoeff0, const float vigCoeff1,
                             const float vigCoeff2, const float vigCoeff3, const TransformPhotoParam& photoParam,
                             GPU::Stream stream) {
  return launchPhotoKernel<InputDefinition::PhotoResponse::LinearResponse>(
      buffer, width, height, rMult, gMult, bMult, vigCenterX, vigCenterY, inverseDemiDiagonalSquared, vigCoeff0,
      vigCoeff1, vigCoeff2, vigCoeff3, photoParam.floatParam, (cl_mem) nullptr, stream);
}

Status gammaPhotoCorrection(GPU::Surface& buffer, const int width, const int height, const float rMult,
                            const float gMult, const float bMult, const float vigCenterX, const float vigCenterY,
                            const float inverseDemiDiagonalSquared, const float vigCoeff0, const float vigCoeff1,
                            const float vigCoeff2, const float vigCoeff3, const TransformPhotoParam& photoParam,
                            GPU::Stream stream) {
  return launchPhotoKernel<InputDefinition::PhotoResponse::GammaResponse>(
      buffer, width, height, rMult, gMult, bMult, vigCenterX, vigCenterY, inverseDemiDiagonalSquared, vigCoeff0,
      vigCoeff1, vigCoeff2, vigCoeff3, photoParam.floatParam, (cl_mem) nullptr, stream);
}

Status emorPhotoCorrection(GPU::Surface& buffer, const int width, const int height, const float rMult,
                           const float gMult, const float bMult, const float vigCenterX, const float vigCenterY,
                           const float inverseDemiDiagonalSquared, const float vigCoeff0, const float vigCoeff1,
                           const float vigCoeff2, const float vigCoeff3, const TransformPhotoParam& photoParam,
                           GPU::Stream stream) {
  return launchPhotoKernel<InputDefinition::PhotoResponse::EmorResponse>(
      buffer, width, height, rMult, gMult, bMult, vigCenterX, vigCenterY, inverseDemiDiagonalSquared, vigCoeff0,
      vigCoeff1, vigCoeff2, vigCoeff3, 0.f, (cl_mem)photoParam.transformData, stream);
}

}  // namespace Core
}  // namespace VideoStitch
