// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "imageFlow.hpp"

#include "flowSequence.hpp"

#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"

#include "libvideostitch/status.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/imageFlowFactory.hpp"

#include <vector>

namespace VideoStitch {
namespace Core {

class SimpleFlow : public ImageFlow {
 public:
  class Factory : public ImageFlowFactory {
   public:
    static Potential<ImageFlowFactory> parse(const Ptv::Value& value);
    explicit Factory(const int flowSize, const int windowSize, const float flowMagnitudeWeight,
                     const float gradientWeight, const float confidenceTransformThreshold,
                     const float confidenceTransformGamma, const float confidenceTransformClampedValue,
                     const int interpolationKernelSize, const float interpolationSigmaSpace,
                     const float interpolationSigmaImage, const float interpolationSigmaConfidence,
                     const int extrapolationKernelSize, const int upsamplingJitterSize, const int leftOffset,
                     const int rightOffset, const float interpolationSigmaTime);
    virtual ~Factory() {}

    virtual Potential<ImageFlow> create() const override;
    virtual std::string getImageFlowName() const override;
    virtual Ptv::Value* serialize() const override;
    virtual bool needsInputPreProcessing() const override;
    virtual std::string hash() const override;
    virtual ImageFlowFactory* clone() const override;

   private:
    // Parameters used for finding flow
    const int flowSize;
    const int windowSize;
    const float flowMagnitudeWeight;
    const float gradientWeight;

    // Parameters used to transform confidence value
    const float confidenceTransformThreshold;
    const float confidenceTransformGamma;
    const float confidenceTransformClampedValue;

    // Parameters used to perform confidence aware interpolation
    const int interpolationKernelSize;
    const float interpolationSigmaSpace;
    const float interpolationSigmaImage;
    const float interpolationSigmaConfidence;

    // Parameters for flow extrapolation
    const int extrapolationKernelSize;

    // Parameters for upsampling
    const int upsamplingJitterSize;

    // Parameters for temporal coherent flow
    const int leftOffset;
    const int rightOffset;
    const float interpolationSigmaTime;
  };

 public:
  static Potential<ImageFlow> create(const std::map<std::string, float>& parameters);

  static std::string getName();

  virtual ImageFlowAlgorithm getFlowAlgorithm() const override;

  virtual Status findTemporalCoherentFlow(const frameid_t frame, const int2& size, GPU::Buffer<float2> outputFlow,
                                          GPU::Stream gpuStream) override;

  virtual Status cacheFlowSequence(const frameid_t frame, const int level, const int2& bufferSize0,
                                   const GPU::Buffer<const uint32_t>& buffer0, const int2& size0,
                                   const GPU::Buffer<const uint32_t>& image0, GPU::Stream gpuStream) const override;

  virtual Status findSingleScaleImageFlow(const int2& offset0, const int2& size0,
                                          const GPU::Buffer<const uint32_t>& image0, const int2& offset1,
                                          const int2& size1, const GPU::Buffer<const uint32_t>& image1,
                                          GPU::Buffer<float2> outputFlow, GPU::Stream gpuStream) override;

  virtual Status findMultiScaleImageFlow(const frameid_t frame, const int level, const int2& bufferSize0,
                                         const GPU::Buffer<const uint32_t>& buffer0, const int2& bufferSize1,
                                         const GPU::Buffer<const uint32_t>& buffer1, GPU::Stream gpuStream) override;

  virtual Status findMultiScaleImageFlow(const frameid_t frame, const int level, const int2& bufferSize0,
                                         const GPU::Buffer<const uint32_t>& buffer0, const int2& bufferSize1,
                                         const GPU::Buffer<const uint32_t>& buffer1, GPU::Buffer<float2> outputFlow,
                                         GPU::Stream gpuStream) override;

  virtual Status findExtrapolatedImageFlow(const int2& offset0, const int2& size0,
                                           const GPU::Buffer<const uint32_t>& image0,
                                           const GPU::Buffer<const float2>& inputFlow0, const int2& offset1,
                                           const int2& size1, const GPU::Buffer<const uint32_t>& image1,
                                           const int2& outputOffset, const int2& outputSize,
                                           GPU::Buffer<float2> outputFlow0, GPU::Stream gpuStream) override;

  virtual Status upsampleFlow(const int2& size0, const int2& offset0, const GPU::Buffer<const uint32_t>& image0,
                              const int2& size1, const int2& offset1, const GPU::Buffer<const uint32_t>& image1,
                              const GPU::Buffer<const float2>& inputFlow, GPU::Buffer<float2> upsampledFlow,
                              GPU::Stream gpuStream) override;

  Status performFlowUpsample22(const int2& size0, const int2& offset0, const GPU::Buffer<const uint32_t>& image0,
                               const int2& size1, const int2& offset1, const GPU::Buffer<const uint32_t>& image1,
                               const GPU::Buffer<const float2>& inputFlow, GPU::Buffer<float2> tmpFlow,
                               GPU::Buffer<float2> upsampledFlow, GPU::Stream gpuStream);

  /**
   * This function is only used for debugging purpose.
   */
  Status upsampleImageFlow(const int level, const int width0, const int height0,
                           const GPU::Buffer<const uint32_t>& buffer0, const int width1, const int height1,
                           const GPU::Buffer<const uint32_t>& buffer1, const int widthFlow, const int heightFlow,
                           const GPU::Buffer<float2>& inputFlow, GPU::Buffer<float2> finalFlow, GPU::Stream gpuStream);

 private:
  static Status findForwardFlow(const int flowSize, const int windowSize, const float flowMagnitudeWeight,
                                const float gradientWeight, const int2 size0, const int2 offset0,
                                const GPU::Buffer<const uint32_t> inputBuffer0,
                                const GPU::Buffer<const float> inputGradientBuffer0, const int2 size1,
                                const int2 offset1, const GPU::Buffer<const uint32_t> inputBuffer1,
                                const GPU::Buffer<const float> inputGradientBuffer1, GPU::Buffer<float2> flow,
                                GPU::Buffer<float> confidence, GPU::Stream gpuStream);

  static Status findOffsetCost(const int2 flowOffset, const int flowSize, const float flowMagnitudeWeight,
                               const float gradientWeight, const int2 size0, const int2 offset0,
                               const GPU::Buffer<const uint32_t> inputBuffer0,
                               const GPU::Buffer<const float> inputGradientBuffer0, const int2 size1,
                               const int2 offset1, const GPU::Buffer<const uint32_t> inputBuffer1,
                               const GPU::Buffer<const float> inputGradientBuffer1, GPU::Buffer<float2> cost,
                               GPU::Stream gpuStream);

  static Status updateBestCost(const int2 flowOffset, const int2 size0, const GPU::Buffer<const float2> cost,
                               GPU::Buffer<float> bestCost, GPU::Buffer<float2> bestOffset, GPU::Stream gpuStream);

  static Status findBackwardFlow(const int flowSize, const int windowSize, const float flowMagnitudeWeight,
                                 const float gradientWeight, const int2 size0, const int2 offset0,
                                 const GPU::Buffer<const uint32_t> inputBuffer0,
                                 const GPU::Buffer<const float> inputGradientBuffer0, const int2 size1,
                                 const int2 offset1, const GPU::Buffer<const uint32_t> inputBuffer1,
                                 const GPU::Buffer<const float> inputGradientBuffer1, GPU::Buffer<float2> flow,
                                 GPU::Buffer<float> confidence, GPU::Stream gpuStream);

  static Status findConfidence(const int windowSize, const float gradientWeight, const int2 size0,
                               const GPU::Buffer<const uint32_t> input0, const GPU::Buffer<const float> gradient0,
                               GPU::Buffer<const float2> forwardFlow0, const int2 size1,
                               const GPU::Buffer<const uint32_t> input1, const GPU::Buffer<const float> gradient1,
                               GPU::Buffer<float> confidence, GPU::Stream gpuStream);

  static Status findBackwardAndForwardFlowAgreementConfidence(
      const int flowSize, const int2 size0, const int2 offset0, const GPU::Buffer<const float2> flow0,
      const GPU::Buffer<const float> confidence0, const int2 size1, const int2 offset1,
      const GPU::Buffer<const float2> flow1, const GPU::Buffer<const float> confidence1,
      GPU::Buffer<float> agreementConfidence0, GPU::Stream gpuStream);

  static Status performConfidenceTransform(const int width, const int height, const float threshold, const float gamma,
                                           const float clampedValue, const GPU::Buffer<const float> inputConfidence,
                                           GPU::Buffer<float> outputConfidence, GPU::Stream gpuStream);

  static Status performConfidenceAwareFlowInterpolation(const bool extrapolation, const int2 size, const int kernelSize,
                                                        const float sigmaSpace, const float sigmaImage,
                                                        const float sigmaConfidence,
                                                        const GPU::Buffer<const uint32_t> inputImage,
                                                        const GPU::Buffer<const float2> inputFlow,
                                                        const GPU::Buffer<const float> inputConfidence,
                                                        GPU::Buffer<float2> outputFlow, GPU::Stream gpuStream);

  static Status performTemporalAwareFlowInterpolation(
      const bool extrapolation, const frameid_t frameId, const int2 size, const int kernelSize, const float sigmaSpace,
      const float sigmaImage, const float sigmaTime, const GPU::Buffer<const float> frames,
      const GPU::Buffer<const uint32_t> inputImages, const GPU::Buffer<const float2> inputFlows,
      const GPU::Buffer<const float> inputConfidences, GPU::Buffer<float2> outputFlow, GPU::Stream gpuStream);

  static Status performFlowJittering(const int jitterSize, const int windowSize, const float flowMagnitudeWeight,
                                     const float gradientWeight, const int2 size0, const int2 offset0,
                                     const GPU::Buffer<const uint32_t> inputBuffer0,
                                     const GPU::Buffer<const float> inputGradientBuffer0, const int2 size1,
                                     const int2 offset1, const GPU::Buffer<const uint32_t> inputBuffer1,
                                     const GPU::Buffer<const float> inputGradientBuffer1,
                                     const GPU::Buffer<const float2> inputFlow, GPU::Buffer<float2> outputFlow,
                                     GPU::Stream gpuStream);

#ifndef NDEBUG
  virtual Status dumpDebugImages(const int width0, const int height0, const GPU::Buffer<const uint32_t>& buffer0,
                                 const int width1, const int height1, const GPU::Buffer<const uint32_t>& buffer1,
                                 GPU::Stream gpuStream) const override;
#endif
 protected:
  virtual Status allocMemory() override;

  GPU::UniqueBuffer<uint32_t> imageLab0;
  GPU::UniqueBuffer<uint32_t> imageLab1;

  GPU::UniqueBuffer<float2> backwardFlowImage1;
  GPU::UniqueBuffer<float2> extrapolatedBackwardFlowImage1;
  GPU::UniqueBuffer<float2> extrapolatedFlowTmp;

  GPU::UniqueBuffer<float> gradient0;
  GPU::UniqueBuffer<float> gradient1;

  GPU::UniqueBuffer<float> confidence;
  GPU::UniqueBuffer<float> transformedConfidence;

  GPU::UniqueBuffer<float2> flow;
  GPU::UniqueBuffer<float2> interpolatedFlow;

  GPU::UniqueBuffer<float2> upsampledFlow;
  GPU::UniqueBuffer<float> kernelWeight;

  GPU::UniqueBuffer<float4> debugInfo;

 private:
  explicit SimpleFlow(const std::map<std::string, float>& parameters);
  std::unique_ptr<FlowSequence> flowSequence;
  friend class ImageFlow;
  friend class PatchMatchFlow;
};

}  // namespace Core
}  // namespace VideoStitch
