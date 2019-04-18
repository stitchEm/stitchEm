// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "core/pyramid.hpp"
#include "core/rect.hpp"
#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"

#include "libvideostitch/status.hpp"
#include "libvideostitch/ptv.hpp"

#include <vector>
#include <map>
#include <memory>

namespace VideoStitch {
namespace Core {

class MergerPair;
class FlowSequence;

/**
 * This class is used to compute image flow between to input buffers, in the "intermediate" space .
 * (See description of class MergerPair for more infos.)
 * This class takes the precomputed coordinate mapping from "MergerPair" and
 * computes flow either in a single scale or multi-scale manner.
 */
class ImageFlow {
 public:
  enum class ImageFlowAlgorithm {
    NoFlow = 0
#ifndef VS_OPENCL
    ,
    SimpleFlow = 1
#endif
  };

  /**
   * This function is only used for debugging purpose in the Visualizer.
   */
  static Potential<ImageFlow> factor(const ImageFlowAlgorithm e, std::shared_ptr<MergerPair>& mergerPair,
                                     const std::map<std::string, float>& parameters);

  /**
   * Init the merger pair use to transform pano <--> intermediate <--> input coordinate
   * @mergerPair The precomputed coordinate mapping of an image pair
   */
  Status init(std::shared_ptr<MergerPair>& mergerPair);

  virtual Status findExtrapolatedImageFlow(const int2& offset0, const int2& size0,
                                           const GPU::Buffer<const uint32_t>& image0,
                                           const GPU::Buffer<const float2>& inputFlow0, const int2& offset1,
                                           const int2& size1, const GPU::Buffer<const uint32_t>& image1,
                                           const int2& outputOffset, const int2& outputSize,
                                           GPU::Buffer<float2> outputFlow0, GPU::Stream gpuStream);

  /**
   * Find optical flow between two image in the intermediate space.
   * @param offset0 Offset of the first buffer.
   * @param size0 Size of the first buffer.
   * @param image0 The first image in LAB color space in "intermediate space".
   * @param offset1 Offset of the second buffer.
   * @param size1 Size of the second buffer.
   * @param image1 The second image in LAB color space in "intermediate space".
   * @param outputFlow The resulting flow.
   * @param gpuStream CUDA stream for the operation.
   */
  virtual Status findSingleScaleImageFlow(const int2& offset0, const int2& size0,
                                          const GPU::Buffer<const uint32_t>& image0, const int2& offset1,
                                          const int2& size1, const GPU::Buffer<const uint32_t>& image1,
                                          GPU::Buffer<float2> outputFlow, GPU::Stream gpuStream) = 0;

  virtual Status findTemporalCoherentFlow(const frameid_t frame, const int2& size, GPU::Buffer<float2> outputFlow,
                                          GPU::Stream gpuStream);

  /**
   * Find multi-scale optical flow between two image in the intermediate space.
   * The resulting flow will be written to @ImageFlow::finalFlow.
   * @param frameIndex Index of the input buffers' frames.
   * @param level The level of pyramid (as in @ImageFlow::mergerPair) to find the flow.
   * @param size0 Size of the first input buffer in "Input space".
   * @param buffer0 The first buffer in "Input space".
   * @param size1 Size of the second input buffer in "Input space".
   * @param buffer1 The second buffer in "Input space".
   * @param gpuStream CUDA stream for the operation.
   */
  virtual Status findMultiScaleImageFlow(const frameid_t frame, const int level, const int2& bufferSize0,
                                         const GPU::Buffer<const uint32_t>& buffer0, const int2& bufferSize1,
                                         const GPU::Buffer<const uint32_t>& buffer1, GPU::Stream gpuStream);

  /**
   * Find multi-scale optical flow between two image in the intermediate space.
   * @param frameIndex Index of the input buffers' frames.
   * @param level The level of pyramid (as in @ImageFlow::mergerPair) to find the flow.
   * @param size0 Size of the first input buffer in "Input space".
   * @param buffer0 The first buffer in "Input space".
   * @param size1 Size of the second input buffer in "Input space".
   * @param buffer1 The second buffer in "Input space".
   * @param outputFlow The resulting flow.
   * @param gpuStream CUDA stream for the operation.
   */
  virtual Status findMultiScaleImageFlow(const frameid_t frame, const int level, const int2& bufferSize0,
                                         const GPU::Buffer<const uint32_t>& buffer0, const int2& bufferSize1,
                                         const GPU::Buffer<const uint32_t>& buffer1, GPU::Buffer<float2> outputFlow,
                                         GPU::Stream gpuStream);

  /**
   * Upsample flow field based on a lower resolution. New resolution size is NewSize = 2 * Old Size.
   * @param size0 Size of the first input buffer in "Intermediate space".
   * @param offset0 Offset of the first input buffer in "Intermediate space".
   * @param image0 The first buffer in "Intermediate space".
   * @param size1 Size of the second input buffer in "Intermediate space".
   * @param offset1 Offset of the second input buffer in "Intermediate space".
   * @param image1 The second buffer in "Intermediate space".
   * @param inputFlow The input flow.
   * @param upsampledFlow The up-sampled flow.
   * @param gpuStream CUDA stream for the operation.
   */
  virtual Status upsampleFlow(const int2& size0, const int2& offset0, const GPU::Buffer<const uint32_t>& image0,
                              const int2& size1, const int2& offset1, const GPU::Buffer<const uint32_t>& image1,
                              const GPU::Buffer<const float2>& inputFlow, GPU::Buffer<float2> upsampledFlow,
                              GPU::Stream gpuStream) = 0;

  const MergerPair* getMergerPair() const;

  const GPU::Buffer<const float2> getFinalFlowBuffer() const;
  const GPU::Buffer<const float2> getFinalExtrapolatedFlowBuffer() const;
  Rect getExtrapolatedFlowRect(const int level) const;
  Rect getFlowRect(const int level) const;
  int2 getLookupOffset(const int level) const;

  virtual Status cacheFlowSequence(const frameid_t keyframe, const int level, const int2& bufferSize0,
                                   const GPU::Buffer<const uint32_t>& buffer0, const int2& bufferSize1,
                                   const GPU::Buffer<const uint32_t>& buffer1, GPU::Stream gpuStream) const;

  virtual ~ImageFlow();

  virtual ImageFlowAlgorithm getFlowAlgorithm() const = 0;

#ifndef NDEBUG
  // This is used for debugging purpose
  virtual Status dumpDebugImages(const int width0, const int height0, const GPU::Buffer<const uint32_t>& buffer0,
                                 const int width1, const int height1, const GPU::Buffer<const uint32_t>& buffer1,
                                 GPU::Stream gpuStream) const;
#endif

 protected:
  /**
   * Basic construction of image flow.
   * @param parameters The set of parameters used for image flow algorithm.
   */
  explicit ImageFlow(const std::map<std::string, float>& parameters);

  virtual Status allocMemory();

  std::shared_ptr<const MergerPair> mergerPair;
  std::map<std::string, float> parameters;

  std::unique_ptr<LaplacianPyramid<float2>> flowLaplacianPyramid;

  GPU::UniqueBuffer<uint32_t> image0;
  GPU::UniqueBuffer<uint32_t> image1;
  GPU::UniqueBuffer<uint32_t> extrapolatedImage1;

  GPU::UniqueBuffer<float2> finalFlow;

  // Store the extrapolated flow
  std::unique_ptr<LaplacianPyramid<float2>> extrapolatedFlowLaplacianPyramid;
  std::vector<Rect> extrapolatedFlowRects;
};

}  // namespace Core
}  // namespace VideoStitch
