// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "algorithm.hpp"
#include "status.hpp"

namespace VideoStitch {
namespace GPU {

namespace Context {

/** Compile all GPU kernels for the thread-local device */
Status compileAllKernels(const bool aheadOfTime, Util::Algorithm::ProgressReporter* pReporter);

/**
 * Compile all the OpenCL programs for the selected device.
 * @param device The device delected
 * @param aheadOfTime if true, only compile the programs that are marked as likelyUsed
 * @param progress if non-null, used as progress indicator.
 */
VS_EXPORT Status compileAllKernelsOnSelectedDevice(int device, const bool aheadOfTime,
                                                   Util::Algorithm::ProgressReporter* pReporter = nullptr);

/** Destroy current GPU context immediately.
 *  This is usually called before program exit to flush profiling data.
 */
VS_EXPORT Status destroy();

// check that sharing between OpenGL and cuda or OpenCL is OK.
// If it doesn't return OK, it probably means that the selected GPU doesn't display
VS_EXPORT Status setDefaultBackendDeviceAndCheck(const int vsDeviceID);

}  // namespace Context
}  // namespace GPU
}  // namespace VideoStitch
