// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"
#include "libvideostitch/status.hpp"
#include "libvideostitch/ptv.hpp"
#include "core/rect.hpp"
#include <vector>

namespace VideoStitch {
namespace Util {

class OpticalFlow {
 public:
  static Status putOverOriginalFlow(const int2 inputOffset, const int2 inputSize,
                                    const GPU::Buffer<const float2> inputFlow, const int2 outputOffset,
                                    const int2 outputSize, GPU::Buffer<float2> outputFlow, GPU::Stream gpuStream);

  static Status backwardCoordLookup(const int2 inputOffset, const int2 inputSize,
                                    const GPU::Buffer<const float2> inputCoordBuffer, const int2 outputOffset,
                                    const int2 outputSize, GPU::Buffer<float2> outputCoordBuffer,
                                    GPU::Stream gpuStream);

  static Status forwardCoordLookup(const int2 inputOffset, const int2 inputSize,
                                   const GPU::Buffer<const float2> inputCoordBuffer, const int2 originalOffset,
                                   const int2 originalSize, const GPU::Buffer<const float2> originalCoordBuffer,
                                   const int2 outputOffset, const int2 outputSize,
                                   GPU::Buffer<float2> outputCoordBuffer, GPU::Stream gpuStream);

  /**
   * @brief This function is used for debugging purpose.
   */
  static Status outwardCoordLookup(const int2 offset1, const int2 size1, const GPU::Buffer<const float2> coordBuffer,
                                   const int2 offset0, const int2 size0, const GPU::Buffer<const uint32_t> inputBuffer,
                                   GPU::Buffer<uint32_t> outputBuffer, GPU::Stream gpuStream);

  /**
   * @brief Lookup RGBA image pixel from a flow buffer
   */
  static Status coordLookup(const int outputWidth, const int outputHeight, const GPU::Buffer<const float2> coordBuffer,
                            const int inputWidth, const int inputHeight, const GPU::Buffer<const uint32_t> inputBuffer,
                            GPU::Buffer<uint32_t> outputBuffer, GPU::Stream gpuStream);

  /**
   * @brief Generate identity flow buffer
   */
  static Status generateIdentityFlow(const int2 size, GPU::Buffer<float2> coordBuffer, GPU::Stream gpuStream,
                                     const bool normalizedFlow = false);

  /**
   * Transform an offset to flow field.
   * @param size0 Size of the buffer
   * @param offset0 Offset of the first buffer
   * @param offset1 Offset of the second buffer.
   * @param buffer The offset buffer --> The output flow buffer.
   * @param gpuStream CUDA stream for the operation.
   */
  static Status transformOffsetToFlow(const int2 size0, const int2 offset0, const int2 offset1,
                                      GPU::Buffer<float2> buffer, GPU::Stream gpuStream);

  /**
   * Transform an offset field to flow field.
   * @param size0 Size of the buffer.
   * @param offset0 Offset of the first buffer.
   * @param offset1 Offset of the second buffer.
   * @param inputBuffer The input offset buffer.
   * @param outputBuffer The output flow buffer.
   * @param gpuStream CUDA stream for the operation.
   */
  static Status transformOffsetToFlow(const int2 size0, const int2 offset0, const int2 offset1,
                                      const GPU::Buffer<const float2> inputBuffer, GPU::Buffer<float2> outputBuffer,
                                      GPU::Stream gpuStream);

  /**
   * Transform a flow field to an offset field.
   * @param size0 Size of the buffer.
   * @param offset0 Offset of the first buffer.
   * @param offset1 Offset of the second buffer.
   * @param inputBuffer The input flow buffer.
   * @param outputBuffer The output offset buffer.
   * @param gpuStream CUDA stream for the operation.
   */
  static Status transformFlowToOffset(const int2 size0, const int2 offset0, const int2 offset1,
                                      const GPU::Buffer<const float2> inputBuffer, GPU::Buffer<float2> outputBuffer,
                                      GPU::Stream gpuStream);

  /**
   * @brief Multiply a flow buffer with a constant. Do nothing on pixel with invalid flow.
   */
  static Status mulFlowOperator(GPU::Buffer<float2> dst, const float2 toMul, std::size_t size, GPU::Stream stream);

  /**
   * @brief Multiply two flow buffers. Do nothing on pixel with invalid flow value.
   */
  static Status mulFlowOperator(GPU::Buffer<float2> dst, GPU::Buffer<const float2> src, const float2 toMul,
                                std::size_t size, GPU::Stream stream);

  /**
   * @brief Perform flow upsampling using bilinear interpolation.
   * Make sure invalid flow value is not propagated into the interpolate function.
   */
  static Status upsampleFlow22(GPU::Buffer<float2> dst, GPU::Buffer<const float2> src, std::size_t dstWidth,
                               std::size_t dstHeight, bool wrap, unsigned blockSize, GPU::Stream stream);

  /**
   * @brief This function is used for debugging purpose in the Visualizer
   */
  static Status interCoordLookup(const int warpWidth, const int interOffsetX, const int interOffsetY,
                                 const int interWidth, const int interHeight,
                                 const GPU::Buffer<const uint32_t> inputBuffer, const int coordWidth,
                                 const int coordHeight, const GPU::Buffer<const float2> coordBuffer,
                                 GPU::Buffer<uint32_t> output, GPU::Stream gpuStream);

  /**
   * @brief Convert the flow buffer into a RGBA buffer for dumping
   */
  static Status convertFlowToRGBA(const int2 size, const GPU::Buffer<const float2> src, const int2 maxFlowValue,
                                  GPU::Buffer<uint32_t> dst, GPU::Stream stream);

  /**
   * @brief Given a RGBA "colorBuffer" and a "flowBuffer" of the same size
   * if a pixel's alpha value is 0, set the correspondent flow value to INVALID_FLOW_VALUE
   */
  static Status setAlphaToFlowBuffer(const int2 size, const GPU::Buffer<const uint32_t> colorBuffer,
                                     GPU::Buffer<float2> flowBuffer, GPU::Stream gpuStream);
};

}  // namespace Util
}  // namespace VideoStitch
