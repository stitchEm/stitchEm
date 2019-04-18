// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/parse.hpp"

#include <memory>

namespace VideoStitch {
namespace MergerMask {

/**
 * @brief Configuration used by the MergerMaskAlgorithm
 */
class MergerMaskConfig {
 public:
  explicit MergerMaskConfig(const Ptv::Value* config);
  ~MergerMaskConfig() = default;

  MergerMaskConfig(const MergerMaskConfig&);

  bool isValid() const { return isConfigValid; }

  /* @return The max overlapping width of two inputs in the output space*/
  int getMaxOverlappingWidth() const { return maxOverlappingWidth; }

  /* @return The maximum size of the panoramic image used for optimization */
  size_t getSizeThreshold() const { return sizeThreshold; }

  /* @return The maximum distortion value that a pixel in the output panorama can have (higher value indicates a more
   * distorted pixel) */
  unsigned char getDistortionThreshold() const { return distortionThreshold; }

  /* @return List of frames used for the optimization */
  std::vector<unsigned int> getFrames() const { return frames; }

  /* @return Size of the kernel used for min-pooling in the image difference metric */
  int getKernelSize() const { return kernelSize; }

  /* @return A parameter used to transform the distortion value*/
  float getDistortionParam() const { return distortionParam; }

  /* @return Whether to use/not use the blending order */
  bool useBlendingOrder() const { return blendingOrder; }

  /* @return Whether to use/not use the seam*/
  bool useSeam() const { return seam; }

  /* @return The feathering size along seam*/
  int getSeamFeatheringSize() const { return seamFeatheringSize; }

  int getInputScaleFactor() const { return inputScaleFactor; }

 private:
  bool isConfigValid;
  int maxOverlappingWidth;
  bool blendingOrder;
  bool seam;
  size_t sizeThreshold;
  unsigned char distortionThreshold;
  float distortionParam;
  int kernelSize;
  int seamFeatheringSize;
  int inputScaleFactor;
  std::vector<unsigned int> frames;
};

}  // namespace MergerMask
}  // namespace VideoStitch
