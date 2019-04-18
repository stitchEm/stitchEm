// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/context.hpp"

#include "context.hpp"

namespace VideoStitch {
namespace GPU {

namespace Context {

Status compileAllKernels(const bool aheadOfTime, Util::Algorithm::ProgressReporter* pReporter) {
  const auto& potContext = getContext();
  FAIL_RETURN(potContext.status());
  return potContext.value().compileAllPrograms(pReporter, aheadOfTime);
}

/**
 * Compile all the OpenCL programs for the selected device.
 * @param device The device delected
 * @param aheadOfTime if true, only compile the programs that are marked as likelyUsed
 * @param progress if non-null, used as progress indicator.
 */
Status compileAllKernelsOnSelectedDevice(int device, const bool aheadOfTime,
                                         Util::Algorithm::ProgressReporter* pReporter) {
  FAIL_RETURN(VideoStitch::GPU::setDefaultBackendDeviceVS(device));
  FAIL_RETURN(GPU::Context::compileAllKernels(aheadOfTime, pReporter));
  return Status::OK();
}

Status destroy() { return destroyOpenCLContext(); }

Status setDefaultBackendDeviceAndCheck(const int vsDeviceID) {
  FAIL_RETURN(setDefaultBackendDeviceVS(vsDeviceID));
  return getContext().status();
}

}  // namespace Context

}  // namespace GPU
}  // namespace VideoStitch
