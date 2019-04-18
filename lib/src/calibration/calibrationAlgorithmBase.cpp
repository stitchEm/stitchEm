// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "calibrationAlgorithmBase.hpp"

#include "calibrationProgress.hpp"

#include "gpu/memcpy.hpp"
#include "gpu/uniqueBuffer.hpp"
#include "image/unpack.hpp"

namespace VideoStitch {
namespace Calibration {

CalibrationAlgorithmBase::CalibrationAlgorithmBase(const Ptv::Value* config) : calibConfig(config) {}

CalibrationAlgorithmBase::~CalibrationAlgorithmBase() {}

Status CalibrationAlgorithmBase::loadInputImage(std::shared_ptr<CvImage>& result, const GPU::Surface& input, int width,
                                                int height) const {
  // Allocate colorspace conversion result on the Device
  auto potFrameBuffer = GPU::Buffer2D::allocate(width, height, "Frame Loading");
  FAIL_RETURN(potFrameBuffer.status());
  auto frameBuffer = potFrameBuffer.value();

  // Allocate colorspace conversion result on the host
  auto hostFrame = GPU::HostBuffer<unsigned char>::allocate(width * height, "Frame Loading");
  FAIL_RETURN(hostFrame.status());

  // Perform colorspace conversion from RGBA to grayscale
  Status gpuStatus = [&result, &frameBuffer, &hostFrame, &input, width, height]() -> Status {
    auto stream = GPU::Stream::getDefault();
    FAIL_RETURN(Image::unpackGrayscale(frameBuffer, input, width, height, stream));
    FAIL_RETURN(GPU::memcpyBlocking(hostFrame.value(), frameBuffer));

    // Create OpenCV Wrapper for image, takes ownership of hostFrame
    result = std::shared_ptr<CvImage>(new Calibration::CvImage(hostFrame.value(), width, height));
    return Status::OK();
  }();

  if (!gpuStatus.ok()) {
    hostFrame.value().release();
  }

  frameBuffer.release();
  return gpuStatus;
}

double CalibrationAlgorithmBase::getProgressUnits(const int numVideoInputs, const int numFramesTuples) const {
  const double numFrames = static_cast<double>(numFramesTuples);
  const double numCameras = static_cast<double>(numVideoInputs);
  double totalProgressUnits = 0.;

  if (!calibConfig.isApplyingPresetsOnly()) {
    // Seek, detection and matching
    totalProgressUnits += (numFrames * numCameras * (CalibrationProgress::seek + CalibrationProgress::kpDetect));
    totalProgressUnits += (numFrames * (numCameras - 1) * (0.5 * numCameras) * CalibrationProgress::kpMatch);
  }
  if (calibConfig.isFovDefined()) {
    // FOV iterations
    totalProgressUnits += CalibrationProgress::fovIterate;
  }
  if (calibConfig.isInDeshuffleMode()) {
    // Deshuffling progress
    totalProgressUnits += CalibrationProgress::deshuffle;
  }
  if (!calibConfig.isApplyingPresetsOnly() && !calibConfig.isInDeshuffleModeOnly()) {
    // Optimization progress
    totalProgressUnits += CalibrationProgress::filter + CalibrationProgress::initGeometry + CalibrationProgress::optim +
                          CalibrationProgress::optim_done;
  }
  return totalProgressUnits;
}

}  // namespace Calibration
}  // namespace VideoStitch
