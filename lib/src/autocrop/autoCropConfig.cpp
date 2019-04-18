// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "autoCropConfig.hpp"

#include <memory>

namespace VideoStitch {
namespace AutoCrop {

AutoCropConfig::AutoCropConfig(const Ptv::Value* config) : isConfigValid(true) {
  if (!config) {
    isConfigValid = false;
    return;
  }

  // Find gaussian blur's kernel size
  const Ptv::Value* val_gaussianBlurKernelSize = config->has("gaussianBlurKernelSize");
  gaussianBlurKernelSize = 21;
  if (val_gaussianBlurKernelSize) {
    if (val_gaussianBlurKernelSize->getType() == Ptv::Value::INT) {
      gaussianBlurKernelSize = (size_t)val_gaussianBlurKernelSize->asInt();
    } else {
      isConfigValid = false;
      return;
    }
  }

  // Find Gaussian blur sigma
  const Ptv::Value* val_gaussianBlurSigma = config->has("gaussianBlurSigma");
  gaussianBlurSigma = 30.0f;
  if (val_gaussianBlurSigma) {
    if (val_gaussianBlurSigma->getType() == Ptv::Value::DOUBLE) {
      gaussianBlurSigma = (float)val_gaussianBlurSigma->asDouble();
    } else {
      isConfigValid = false;
      return;
    }
  }

  // Find binary thresholding value
  const Ptv::Value* val_neighborThreshold = config->has("neighborThreshold");
  neighborThreshold = 40;
  if (val_neighborThreshold) {
    if (val_neighborThreshold->getType() == Ptv::Value::INT) {
      neighborThreshold = (size_t)val_neighborThreshold->asInt();
    } else {
      isConfigValid = false;
      return;
    }
  }

  // Find binary thresholding value
  const Ptv::Value* val_differenceThreshold = config->has("differenceThreshold");
  differenceThreshold = 500;
  if (val_differenceThreshold) {
    if (val_differenceThreshold->getType() == Ptv::Value::INT) {
      differenceThreshold = (size_t)val_differenceThreshold->asInt();
    } else {
      isConfigValid = false;
      return;
    }
  }

  // Find the number of sampled points
  const Ptv::Value* val_convexHullSampledCount = config->has("convexHullSampledCount");
  convexHullSampledCount = 1000;
  if (val_convexHullSampledCount) {
    if (val_convexHullSampledCount->getType() == Ptv::Value::INT) {
      convexHullSampledCount = (size_t)val_convexHullSampledCount->asInt();
    } else {
      isConfigValid = false;
      return;
    }
  }

  // The search range for finding the fine scale points
  const Ptv::Value* val_fineTuneMarginSize = config->has("fineTuneMarginSize");
  fineTuneMarginSize = 150;
  if (val_fineTuneMarginSize) {
    if (val_fineTuneMarginSize->getType() == Ptv::Value::INT) {
      fineTuneMarginSize = (size_t)val_fineTuneMarginSize->asInt();
    } else {
      isConfigValid = false;
      return;
    }
  }

  // Check if a circle image need to be dumped
  const Ptv::Value* val_circleImage = config->has("circleImage");
  circleImage = false;
  if (val_circleImage) {
    if (val_circleImage->getType() == Ptv::Value::BOOL) {
      circleImage = (bool)val_circleImage->asBool();
    } else {
      isConfigValid = false;
      return;
    }
  }

  // Check if the original image need to be dumped
  const Ptv::Value* val_originalImage = config->has("originalImage");
  originalImage = false;
  if (val_originalImage) {
    if (val_originalImage->getType() == Ptv::Value::BOOL) {
      originalImage = (bool)val_originalImage->asBool();
    } else {
      isConfigValid = false;
      return;
    }
  }

  // A parameter to scale the final radius, in order to get rid of the black transition area
  const Ptv::Value* val_scaleRadius = config->has("scaleRadius");
  scaleRadius = 0.985;
  if (val_scaleRadius) {
    if (val_scaleRadius->getType() == Ptv::Value::DOUBLE) {
      scaleRadius = (double)val_scaleRadius->asDouble();
    } else {
      isConfigValid = false;
      return;
    }
  }
}

AutoCropConfig::AutoCropConfig(const AutoCropConfig& other)
    : isConfigValid(other.isConfigValid),
      gaussianBlurKernelSize(other.gaussianBlurKernelSize),
      gaussianBlurSigma(other.gaussianBlurSigma),
      convexHullSampledCount(other.convexHullSampledCount),
      neighborThreshold(other.neighborThreshold),
      differenceThreshold(other.differenceThreshold),
      fineTuneMarginSize(other.fineTuneMarginSize),
      circleImage(other.circleImage),
      originalImage(other.originalImage),
      scaleRadius(other.scaleRadius) {}

}  // namespace AutoCrop
}  // namespace VideoStitch
