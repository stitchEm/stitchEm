// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"
#include "gpu/uniqueBuffer.hpp"

#include "libvideostitch/status.hpp"
#include "libvideostitch/ptv.hpp"

#include "core/rect.hpp"
#include "core1/textureTarget.hpp"

namespace VideoStitch {
namespace Core {
class StereoRigDefinition;
}
namespace Util {

class ImageProcessingGPU {
 public:
  /**
   * @brief Down sample an image by a certain "levelCount" using nearest ("isNearest" == true) or bilinear interpolation
   * ("isNearest" == false)
   */
  template <typename T>
  static Status downSampleImages(const int levelCount, int& bufferWidth, int& bufferHeight, GPU::Buffer<T> buffer,
                                 GPU::Stream stream, const bool isNearest = false);

  /**
   * @brief Calculate sum of all elements ("output") and count the total of non-zero entries ("mask")
   */
  static Status calculateSum(const int numElement, const GPU::Buffer<const float> buffer, const unsigned blockSize,
                             GPU::Stream stream, float& output, float& mask);

  /**
   * @brief Convert RGB-gradient to normalizedLAB-gradient (3 bytes-1 byte).
   * Note that pixels with value Image::RGBA::pack(1, 2, 3, 0) represent black color with alpha = 0
   */
  static Status convertRGBandGradientToNormalizedLABandGradient(const int2 bufferSize,
                                                                GPU::Buffer<uint32_t> colorBuffer, GPU::Stream stream);

  /**
   * @brief Convert RGB210 buffer to RGB-gradient (3 bytes-1 byte).
   * Note that in the final buffer, original pixels with alpha = 0 are set to value Image::RGBA::pack(1, 2, 3, 0)
   */
  static Status convertRGB210ToRGBandGradient(const int2 bufferSize, const GPU::Buffer<const uint32_t> inputBuffer,
                                              GPU::Buffer<uint32_t> colorBuffer, GPU::Stream stream);

  /**
   * @brief Convert RGB210 buffer to RGBA buffer
   */
  static Status convertRGB210ToRGBA(const int2 bufferSize, GPU::Buffer<uint32_t> buffer, GPU::Stream stream);

  /**
   * @brief normalizedLAB-gradient (3 bytes-1 byte) to RGBA.
   * Note that pixels with value Image::RGBA::pack(1, 2, 3, 0) represent black color with alpha = 0
   */
  static Status convertNormalizedLABandGradientToRGBA(const int2 bufferSize, GPU::Buffer<uint32_t> colorBuffer,
                                                      GPU::Stream stream);

  /**
   * @brief Extract a buffer channel from a RGBA buffer, used for debugging purpose
   * Note: "inputBuffer" stored as Image::RGBA format
   */
  static Status extractChannel(const int2 bufferSize, const GPU::Buffer<const uint32_t> inputBuffer,
                               const int channelIndex, GPU::Buffer<unsigned char> outputBuffer, GPU::Stream stream);

  /**
   * Convert a RGB buffer into normalized LAB buffer
   * @param size The size of input buffer.
   * @param inputRGBBuffer Input RGB buffer.
   * @param outputNormalizedLABBuffer The output buffer in normalized LAB color space.
   * @param stream CUDA stream for the operation.
   */
  static Status convertRGBToNormalizedLAB(const int2 size, const GPU::Buffer<const uint32_t> inputRGBBuffer,
                                          GPU::Buffer<uint32_t> outputNormalizedLABBuffer, GPU::Stream gpuStream);

  /**
   * Convert a RGB buffer into luminance
   * @param size The size of input buffer.
   * @param inputBuffer Input RGB buffer.
   * @param outputLuminanceBuffer The output luminance buffer.
   * @param stream CUDA stream for the operation.
   */
  static Status findLuminance(const int2 size, const GPU::Buffer<const uint32_t> inputBuffer,
                              GPU::Buffer<float> outputLuminanceBuffer, GPU::Stream gpuStream);

  /**
   * Find gradient of an RGB buffer
   * @param size The size of input buffer.
   * @param inputBuffer Input RGB buffer.
   * @param outputGradientBuffer The output gradient buffer.
   * @param stream CUDA stream for the operation.
   */
  static Status findGradient(const int2 size, const GPU::Buffer<const uint32_t> inputBuffer,
                             GPU::Buffer<float> outputGradientBuffer, GPU::Stream gpuStream);

  /**
   * Alpha-aware blending of two input buffers.
   * @param dstRect The bounding rect of the dst buffer.
   * @param weight0 Weight of the first buffer.
   * @param src0Rect The bounding rect of the first buffer.
   * @param src0 The first buffer.
   * @param weight1 Weight of the second buffer.
   * @param src1Rect The bounding rect of the second buffer.
   * @param src1 The second buffer.
   */
  static Status buffer2DRGBACompactBlendOffsetOperator(const Core::Rect& dstRect, GPU::Buffer<uint32_t> dst,
                                                       const float weight0, const Core::Rect& src0Rect,
                                                       const GPU::Buffer<const uint32_t> src0, const float weight1,
                                                       const Core::Rect& src1Rect,
                                                       const GPU::Buffer<const uint32_t> src1, GPU::Stream gpuStream);

  /**
   * Convert a buffer into binary
   * @param size The size of input buffer.
   * @param inputMask Input buffer.
   * @param binarizedMask The output buffer, 1 if the input value is not 0.
   * @param stream CUDA stream for the operation.
   */
  static Status binarizeMask(const int2 size, const GPU::Buffer<const uint32_t> inputMask,
                             GPU::Buffer<uint32_t> binarizedMask, GPU::Stream gpuStream);

  /**
   * Find bounding box of on non-zero value in @maskBuffer.
   * @param width The width of input buffer.
   * @param height The height of input buffer.
   * @param maskBuffer Input buffer.
   * @param boundingRect The output bounding rect of @maskBuffer.
   * @param stream CUDA stream for the operation.
   */
  static Status findBBox(Core::TextureTarget, const bool canWarp, const Core::StereoRigDefinition* rigDef,
                         const int width, const int height, const GPU::Buffer<const uint32_t>& maskBuffer,
                         Core::Rect& boundingRect, GPU::Stream gpuStream);

  /**
   * Down sample coordinate mapping buffer and its weight buffer.
   * @param inputWidth Width of buffer.
   * @param inputHeight Height of buffer.
   * @param levelCount The level of down sampling.
   * @param coordBuffer The input coordinate buffer.
   * @param weightBuffer The input weight buffer.
   * @param stream CUDA stream for the operation.
   */
  static Status downSampleCoordImage(const int inputWidth, const int inputHeight, const int levelCount,
                                     GPU::UniqueBuffer<float2>& coordBuffer, GPU::UniqueBuffer<uint32_t>& weightBuffer,
                                     GPU::Stream stream);

  /**
   * Perform pixel-wise AND operator of two input buffers.
   * @param warpWidth Width of the canvas.
   * @param boundingRect0 The bounding rect of the first buffer.
   * @param buffer0 The first buffer.
   * @param boundingRect1 The bounding rect of the second buffer.
   * @param buffer1 The second buffer.
   * @param stream CUDA stream for the operation.
   */
  static Status onBothBufferOperator(const int warpWidth, const Core::Rect boundingRect0,
                                     const GPU::Buffer<const uint32_t> buffer0, const Core::Rect boundingRect1,
                                     const GPU::Buffer<const uint32_t> buffer1, const Core::Rect boundingRectBuffer,
                                     GPU::Buffer<uint32_t> buffer, GPU::Stream stream);

  /**
   * Given two buffers, compute a tight bounding box of the overlapping area.
   * @param warpWidth Width of the canvas.
   * @param boundingRect0 The bounding rect of the first buffer.
   * @param buffer0 The first buffer.
   * @param boundingRect1 The bounding rect of the second buffer.
   * @param buffer1 The second buffer.
   * @param stream CUDA stream for the operation.
   */
  static Status computeTightOverlappingRect(Core::TextureTarget, const int warpWidth, const Core::Rect& boundingRect0,
                                            const GPU::Buffer<const uint32_t>& buffer0, const Core::Rect& boundingRect1,
                                            const GPU::Buffer<const uint32_t>& buffer1, Core::Rect& overlappingRect,
                                            GPU::Stream stream);

  /**
   * @brief Set a buffer to a constant value
   */
  template <typename T>
  static Status setConstantBuffer(const int2 size, GPU::Buffer<T> buffer, const T value, GPU::Stream gpuStream);

  /**
   * @brief Pack a buffer inside another buffer.
   */
  template <typename T>
  static Status packBuffer(const int warpWidth, const T invalidValue, const Core::Rect inputRect,
                           const GPU::Buffer<const T> inputBuffer, const Core::Rect outputRect,
                           GPU::Buffer<T> outputBuffer, GPU::Stream gpuStream);

  /**
   * @brief Thresholding an input buffer, set value < thresholdValue to minBoundValue
   *                                                                and maxBoundValue otherwise
   */
  template <typename T>
  static Status thresholdingBuffer(const int2 size, const T thresholdValue, const T minBoundValue,
                                   const T maxBoundValue, GPU::Buffer<T> inputBuffer, GPU::Stream gpuStream);

 private:
  /**
   * @brief Down sample an image by 1 level using nearest ("isNearest" == true) or bilinear interpolation ("isNearest"
   * == false)
   */
  template <typename T>
  static Status subsampleImage(GPU::Buffer<T> dst, GPU::Buffer<const T> src, std::size_t srcWidth,
                               std::size_t srcHeight, GPU::Stream gpuStream, const bool isNearest = false);
};

}  // namespace Util
}  // namespace VideoStitch
