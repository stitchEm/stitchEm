// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/parse.hpp"

#include <memory>

namespace VideoStitch {
namespace AutoCrop {

/**
 * @brief Configuration used by the AutoCropAlgorithm
 */
class VS_EXPORT AutoCropConfig {
 public:
  explicit AutoCropConfig(const Ptv::Value* config);
  ~AutoCropConfig() = default;
  AutoCropConfig(const AutoCropConfig&);

  bool isValid() const { return isConfigValid; }

  size_t getGaussianBlurKernelSize() const { return gaussianBlurKernelSize; }

  float getGaussianBlurSigma() const { return gaussianBlurSigma; }

  size_t getNeighborThreshold() const { return neighborThreshold; }

  size_t getDifferenceThreshold() const { return differenceThreshold; }

  size_t getConvexHullSampledCount() const { return convexHullSampledCount; }

  size_t getFineTuneMarginSize() const { return fineTuneMarginSize; }

  bool dumpCircleImage() const { return circleImage; }

  bool dumpOriginalImage() const { return originalImage; }

  double getScaleRadius() const { return scaleRadius; }

 private:
  bool isConfigValid;
  size_t gaussianBlurKernelSize;
  float gaussianBlurSigma;
  size_t convexHullSampledCount;
  size_t neighborThreshold;
  size_t differenceThreshold;
  size_t fineTuneMarginSize;
  bool circleImage;
  bool originalImage;
  double scaleRadius;
};

}  // namespace AutoCrop
}  // namespace VideoStitch
