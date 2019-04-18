// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "mergerMaskConfig.hpp"
#include "mergerMaskProgress.hpp"

#include "core/rect.hpp"
#include "gpu/core1/transform.hpp"
#include "gpu/uniqueBuffer.hpp"
#include "gpu/sharedBuffer.hpp"

#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/status.hpp"
#include "libvideostitch/stereoRigDef.hpp"

#include <vector>
#include <memory>

namespace VideoStitch {
namespace Core {
class ImageMapping;
class ImageMerger;
class InputDefinition;
class PanoDefinition;
class PanoRemapper;
class StereoRigDefinition;
class PreProcessor;
}  // namespace Core

namespace Input {
class VideoReader;
}

namespace MergerMask {

typedef std::map<std::pair<size_t, size_t>, GPU::UniqueBuffer<uint32_t>> FrameBuffers;
typedef std::vector<GPU::UniqueBuffer<uint32_t>> MappedFramesBuffer;
typedef std::vector<Core::Rect> FrameRects;

/**
 * @brief The main algorithm to optimize for the blending mask and order.
 */
class MergerMask {
 public:
  /**
   * Creates a MergerMask that optimizes for a given panorama and rig setting.
   * @param readers The video inputs
   * @param pano The input panorama definition
   * @param rigDef The rig's definition
   * @param config Algorithm configuration
   */
  static Potential<MergerMask> create(const std::map<readerid_t, Input::VideoReader*>&,
                                      const Core::PanoDefinition& pano, const Core::StereoRigDefinition* rigDef,
                                      const MergerMaskConfig& config, const MergerMaskProgress& progress);

  ~MergerMask();

  /**
   * @brief Compute the results, put the inputs map into the first param and the blending order into the second param.
   */
  Status getMergerMasks(GPU::Buffer<uint32_t> inputIndexPixelBuffer, std::vector<size_t>& masksOrder);

  /**
   * @brief From the precomputed lookup coordinate in the output space, find the mask in the input space
   */
  static Status getInputMaskFromOutputIndices(const videoreaderid_t imId, const int scaleFactor, const int2 outputSize,
                                              const GPU::Buffer<const uint32_t> maskBuffer, const int2 inputSize,
                                              const GPU::Buffer<const float2> inputCoordBuffer,
                                              GPU::Buffer<unsigned char> inputMask, GPU::Stream stream);

  /**
   * @brief From the mask in the input space, find mask in the output space
   */
  static Status getOutputIndicesFromInputMask(const videoreaderid_t imId, const int scaleFactor, const int2 inputSize,
                                              const GPU::Buffer<const unsigned char> inputMask, const int2 outputSize,
                                              const GPU::Surface& coordBuffer, GPU::Buffer<uint32_t> maskBuffer,
                                              GPU::Stream stream);

  /**
   * @brief Take a blending mask stored in the output space,
   * construct the encoded polylines of the mask in the input spaces
   */
  static Status transformMasksFromOutputToEncodedInputSpace(const Core::PanoDefinition& pano,
                                                            const std::map<readerid_t, Input::VideoReader*>& readers,
                                                            const GPU::Buffer<const uint32_t>& maskOutputSpaces,
                                                            std::map<videoreaderid_t, std::string>& maskInputSpaces);

  /**
   * @brief Take the encoded polylines of the mask in the input spaces,
   * find the final blending mask in the output space
   */
  static Status transformMasksFromEncodedInputToOutputSpace(
      const Core::PanoDefinition& pano, const std::map<readerid_t, Input::VideoReader*>& readers,
      const std::map<videoreaderid_t, std::string>& maskInputSpaces, GPU::Buffer<uint32_t> maskOutputSpaces);

 protected:
  explicit MergerMask(const Core::PanoDefinition& pano, const Core::StereoRigDefinition* rigDef,
                      const MergerMaskConfig& config, const MergerMaskProgress& progress);

 private:
  Status releaseMergerMaskMemory();
  /**
   * @brief Setup all the data used for the optimization
   */
  Status setup(const std::map<readerid_t, Input::VideoReader*>&);

  /**
   * @brief Get inputs-map and blending order in the original resolution
   */
  Status getOriginalMaskFromOrder(const std::vector<int>& normalizedOverlappingWidths,
                                  const std::vector<size_t>& maskOrder, GPU::Buffer<uint32_t> inputIndexPixelBuffer);

  /**
   * @brief Read all the images from inputs.
   */
  static Status retrieveImages(const std::vector<unsigned int>& frames, FrameBuffers& frameBuffers,
                               const Core::PanoDefinition& pano);

  /**
   * @brief The number of levels needed to downsample the original image to the bounded resolution
   */
  static int getDownLevel(const Core::PanoDefinition& pano, const int sizeThreshold);

  /**
   * @brief The down sampled size
   */
  int getDownSize(const int size) const;

  /**
   * @brief The down sampled coordinate
   */
  int getDownCoord(const int coord) const;

  // Setup data for the optimization
  Status setupOverlappingPair();
  Status setupMappings(const std::map<readerid_t, Input::VideoReader*>&);
  Status setupTransform(const std::map<readerid_t, Input::VideoReader*>&);
  Status setupDistortionMap();
  Status setupMappedRect();
  Status setupFrames();

  // Update the inputs map
  Status updateIndexMask(const videoreaderid_t camId, const int maxOverlappingCount,
                         const GPU::Buffer<const char> cameraIndices, const int2 distortionBufferSize,
                         const int2 distortionBufferOffset, const GPU::Buffer<const unsigned char> distortionBuffer,
                         const int2 inputSize, GPU::Buffer<uint32_t> inputIndexBuffer, GPU::Buffer<unsigned char> mask,
                         const GPU::Buffer<const uint32_t> srcMap, GPU::Stream stream);

  /**
   * @brief Update the distortion mask to current camId.
   * Replace old pixel by new pixel if the original distortion value is bigger than a threshold
   * and the new distortion value is smaller.
   */
  Status updateDistortionFromMask(const videoreaderid_t camId, const int2 distortionBufferSize,
                                  const int2 distortionBufferOffset, GPU::Buffer<unsigned char> distortionBuffer,
                                  const int2 inputSize, const GPU::Buffer<const uint32_t> srcMap, GPU::Stream stream);

  Status updateInputIndexByDistortionMap(const videoreaderid_t camId, const int2 inputSize,
                                         const GPU::Buffer<const uint32_t> inputNonOverlappingIndexBuffer,
                                         const GPU::Buffer<const unsigned char> inputDistortionBuffer,
                                         GPU::Buffer<uint32_t> nextNonOverlappingIndexBuffer,
                                         GPU::Buffer<unsigned char> nextDistortionBuffer, GPU::Stream stream,
                                         const bool original = false);

  Status initializeMasks(const int2 inputSize, const videoreaderid_t camId,
                         GPU::Buffer<uint32_t> inputNonOverlappingIndexBuffer,
                         GPU::Buffer<unsigned char> inputDistortionBuffer, GPU::Stream stream,
                         const bool original = false);

  Status updateOverlappingMap(const std::vector<int>& normalizedMaxOverlappingWidths, const int2& size,
                              const std::vector<size_t>& allCam,
                              const GPU::Buffer<const uint32_t>& inputNonOverlappingIndexBuffer,
                              GPU::Buffer<uint32_t> inputIndexBuffer, GPU::Stream stream, const bool original = false);

  /**
   * @brief Update the cost of stitching, accumulate it to the "cost" buffer
   */
  Status updateStitchingCost(const int2 inputSize, const int kernelSize,
                             const GPU::Buffer<const uint32_t> inputIndexBuffer,
                             const GPU::Buffer<const uint32_t> mappedOffset,
                             const GPU::Buffer<const int2> mappedRectOffset,
                             const GPU::Buffer<const int2> mappedRectSize,
                             const GPU::Buffer<const uint32_t> mappedBuffer, GPU::Buffer<float> cost,
                             GPU::Buffer<uint32_t> debugBuffer0, GPU::Buffer<uint32_t> debugBuffer1,
                             GPU::Stream stream);

  /**
   * @brief Transform the distortion function.
   */
  Status transformDistortion(const int2 inputSize, GPU::Buffer<unsigned char> distortionBuffer, GPU::Stream stream);

  /**
   * @brief Compute the blending cost of current camera configuration
   */
  Status getBlendingCost(const std::vector<int>& normalizedMaxOverlappingWidthgetmergerms, const int2& size,
                         const std::vector<size_t>& allCam,
                         const GPU::Buffer<const unsigned char>& inputDistortionBuffer,
                         const GPU::Buffer<const uint32_t>& inputNonOverlappingIndexBuffer,
                         GPU::Buffer<unsigned char> nextDistortionBuffer,
                         GPU::Buffer<uint32_t> nextNonOverlappingIndexBuffer,
                         GPU::Buffer<uint32_t> nextInputIndexBuffer, float& blendingCost);

  Status performSeamOptimization(const std::vector<size_t>& maskOrder, GPU::Buffer<uint32_t> inputIndexBuffer);

  Status updateIndexMaskAfterSeam(const videoreaderid_t id0, const videoreaderid_t id1, const int2 bufferSize,
                                  const GPU::Buffer<const unsigned char> seamBuffer, GPU::Buffer<uint32_t> indexBuffer,
                                  GPU::Stream stream);

  static Status lookupColorBufferFromInputIndex(const int wrapWidth, const GPU::Buffer<const unsigned char> camBuffer,
                                                const GPU::Buffer<const int2> mappedRectOffsets,
                                                const GPU::Buffer<const int2> mappedRectSizes,
                                                const GPU::Buffer<const uint32_t> mappedOffsets,
                                                const GPU::Buffer<const uint32_t> mappedBuffers, const int2 bufferSize,
                                                const GPU::Buffer<const uint32_t> inputIndexBuffer,
                                                GPU::Buffer<uint32_t> outputBuffer, GPU::Stream stream);

  static Status updateSeamMask(const videoreaderid_t id, const int2 size,
                               const GPU::Buffer<const uint32_t> originalInputIndexBuffer,
                               const GPU::Buffer<const unsigned char> distanceBuffer,
                               GPU::Buffer<uint32_t> seamOuputIndexBuffer, GPU::Stream stream);

  static Status extractLayerFromIndexBuffer(const videoreaderid_t id, const int2 bufferSize,
                                            const GPU::Buffer<const uint32_t> inputIndexBuffer,
                                            GPU::Buffer<uint32_t> extractedBuffer, GPU::Stream stream);

 private:
  std::map<videoreaderid_t, Core::ImageMapping*> imageMappings;
  std::map<videoreaderid_t, Core::Transform*> transforms;

  GPU::UniqueBuffer<uint32_t> inputsMapBuffer;
  GPU::UniqueBuffer<uint32_t> inputsMapOriginalBuffer;

  std::map<videoreaderid_t, GPU::Buffer<unsigned char>> distortionMaps;
  std::map<videoreaderid_t, GPU::Buffer<unsigned char>> originalDistortionMaps;

  std::vector<std::vector<int>> isOverlapping;
  MappedFramesBuffer cachedMappedFrames;
  FrameRects cachedMappedRects;
  GPU::UniqueBuffer<uint32_t> mappedOffset;
  GPU::UniqueBuffer<int2> mappedRectOffset;
  GPU::UniqueBuffer<int2> mappedRectSize;

  FrameRects cachedOriginalMappedRects;
  GPU::UniqueBuffer<uint32_t> originalMappedOffset;
  GPU::UniqueBuffer<int2> originalMappedRectOffset;
  GPU::UniqueBuffer<int2> originalMappedRectSize;

  GPU::SharedBuffer<uint32_t> work1;
  GPU::SharedBuffer<uint32_t> work2;
  GPU::SharedBuffer<unsigned char> workMask;
  GPU::UniqueBuffer<float> workCost;

  const Core::PanoDefinition& pano;
  const Core::StereoRigDefinition* const rigDef;
  const MergerMaskConfig& mergerMaskConfig;
  const int downsamplingLevelCount;

  MergerMaskProgress progress;
};

}  // namespace MergerMask
}  // namespace VideoStitch
