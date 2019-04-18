// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "mergerPair.hpp"

#include "libvideostitch/status.hpp"

#include <memory>

namespace VideoStitch {
namespace Core {

/**
 * This class is used to warp image from the original pano mapping
 * to the more compatible pano mapping (so that the seam is less visible in the pano space).
 * (See description of class MergerPair for more infos.)
 */
class ImageWarper {
 public:
  enum class ImageWarperAlgorithm {
    NoWarper = 0
#ifndef VS_OPENCL
    ,
    LinearFlowWarper = 1
#endif

  };
  static ImageWarperAlgorithm getDefaultImageWarper();

  /**
   * This function is only used for debugging purpose in the Visualizer.
   */
  static Potential<ImageWarper> factor(const ImageWarperAlgorithm e, std::shared_ptr<MergerPair> mergerPair,
                                       std::map<std::string, float> parameters, GPU::Stream gpuStream);

  /**
   * Warp input buffer to parallax-tolerant buffer in pano space
   * @param warpedBuffer The output warped buffer.
   * @param inputBuffer Input image buffer in "input space".
   * @param flowRect The bounding rectangle of the flow.
   * @param flow The input flow buffer.
   * @param lookupOffsetX The left coordinate of the lookup buffer
   * @param lookupOffsetY The top coordinate of the lookup buffer
   * @param debug A debug buffer
   * @param gpuStream CUDA stream for the operation.
   */
  virtual Status warp(GPU::Buffer<uint32_t> warpedBuffer, const GPU::Buffer<const uint32_t> inputBuffer,
                      const Rect& flowRect, const GPU::Buffer<const float2> flow, const int lookupOffsetX,
                      const int lookupOffsetY, GPU::Buffer<float4> debug, GPU::Buffer<uint32_t> flowWarpedBuffer,
                      GPU::Stream gpuStream);

  /**
   * Init the merger pair use to transform pano <--> intermediate <--> input coordinate
   * @mergerPair The precomputed coordinate mapping of an image pair
   */
  Status init(std::shared_ptr<MergerPair>& mergerPair);

  virtual Status setupCommon(GPU::Stream gpuStream);

  /**
   * Check whether the current warper need image flow, or not
   */
  virtual bool needImageFlow() const = 0;

  virtual ImageWarperAlgorithm getWarperAlgorithm() const = 0;

  virtual ~ImageWarper();

 protected:
  explicit ImageWarper(const std::map<std::string, float>& parameters_);

  std::map<std::string, float> parameters;
  std::shared_ptr<const MergerPair> mergerPair;
};
}  // namespace Core
}  // namespace VideoStitch
