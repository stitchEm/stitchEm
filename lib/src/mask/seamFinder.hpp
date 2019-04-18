// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "dijkstraShortestPath.hpp"
#include "mergerMaskConstant.hpp"

#include "core/rect.hpp"
#include "gpu/uniqueBuffer.hpp"
#include "gpu/sharedBuffer.hpp"
#include "gpu/vectorTypes.hpp"
#include "gpu/stream.hpp"

#include "libvideostitch/status.hpp"

#include <vector>

namespace VideoStitch {
namespace MergerMask {

/**
 * @brief The main algorithm used to find the seam in the overlapping area
 */
class SeamFinder {
 public:
  static Potential<SeamFinder> create(const int seamFeatheringSize, const int kernelSize, const int wrapWidth,
                                      const Core::Rect& rect0, GPU::Buffer<const uint32_t> input0Buffer,
                                      const Core::Rect& rect1, GPU::Buffer<const uint32_t> input1Buffer,
                                      GPU::Stream& stream,
                                      GPU::SharedBuffer<unsigned char> workMask = GPU::SharedBuffer<unsigned char>(),
                                      GPU::SharedBuffer<uint32_t> work1 = GPU::SharedBuffer<uint32_t>(),
                                      GPU::SharedBuffer<uint32_t> work2 = GPU::SharedBuffer<uint32_t>());

  /**
   * @brief Find the optimal seam to blend the two input images
   */
  Status findSeam();

  /**
   * @brief Check whether a seam is found
   */
  bool seamFound() const;

  GPU::Buffer<const unsigned char> getOutputsMap() const { return outputsMapBuffer.borrow_const(); }

  GPU::Buffer<const unsigned char> getInputsMap() const { return inputsMapBuffer.borrow_const(); }

  /**
   * @brief Blend the two images with the computed seam
   */
  Status blendImages(GPU::Buffer<uint32_t> outputBuffer);

  /**
   * @brief A very special function to replace buffers. The old and new buffers must have the same size
   */
  Status replaceBuffers(GPU::Buffer<const uint32_t> newBuffer0, GPU::Buffer<const uint32_t> newBuffer1);

  /**
   * @brief Find the connected component after a seam was computed.
   * Break the overlapping into parts to decide which part belong to the first or second image after using the computed
   * seams.
   */
  Status findConnectedComponentsAfterCuts(const int bufferIndex, std::vector<int>& components);

  /**
   * @brief Save seam image to a png file
   */
  Status saveSeamImage(const std::string& filename, const int bufferIndex);

  /**
   * @brief Save seam image to a buffer
   */
  Status saveSeamToBuffer(const int bufferIndex, std::vector<unsigned char>& data);

 protected:
  /**
   * @brief Compute the cost of the seam, given a 4-connected neighboring pixels
   */
  Status prepareSeamCostBuffer(const Core::Rect rect, GPU::Buffer<float> costsBuffer);

  /**
   * @brief Find a buffer that contains border of the valid pixels
   */
  Status findBordersBuffer(const Core::Rect rect, const GPU::Buffer<const unsigned char> mapBuffer,
                           GPU::Buffer<unsigned char> bordersBuffer, const int directionCount = 8);

  /**
   * @brief Update the inputsMask after a seam is used to cut components into small parts
   */
  Status updateMaskAfterCut(const int bufferIndex, const std::vector<char>& components,
                            std::vector<unsigned char>& outputsMap);

  /**
   * @brief Find the valid part of the image pair
   */
  Status findValidMask(const Core::Rect rect, const GPU::Buffer<const uint32_t> inputBuffer,
                       std::vector<unsigned char>& mask, GPU::Stream stream);

  Status findFeatheringMask(const int2 size, const GPU::Buffer<const unsigned char> inputBuffer,
                            GPU::Buffer<uint32_t> outputBuffer, GPU::Stream stream);

  /**
   * @brief Find the InputsMap of the current pair
   */
  Status findInputsMap();

 private:
  /**
   * @brief Pass the two image pair to find seam
   */
  SeamFinder(const int seamFeatheringSize, const int kernelSize, const int wrapWidth, const Core::Rect& rect0,
             GPU::Buffer<const uint32_t> input0Buffer, const Core::Rect& rect1,
             GPU::Buffer<const uint32_t> input1Buffer, GPU::Stream& stream, GPU::SharedBuffer<unsigned char> workMask,
             GPU::SharedBuffer<uint32_t> work1, GPU::SharedBuffer<uint32_t> work2);

  /**
   * @brief Setup temp memory for seam computation
   */
  Status setup();

  /**
   * @brief Setup up the cost function map
   */
  Status findCostMap();

  /**
   * @brief Find the start and end points from which seams are computed
   */
  Status findIntersectingSegments(const unsigned char borderIndex);

  /**
   * @brief
   */
  Status getRidOfOnePixelWidthInputsMap(const unsigned char borderIndex);

  /**
   * @brief isConnectedByOnePixel
   * @param size Size of the inputsMap
   * @param inputsMap Store the component values, different values indicate different component
   * @param value Value of the connected component
   * @param coord The current position to be check
   * @param connectedPoint The connected point of "coord" with "value", if exists
   * @return true if in the 8-neighbor pixels of "coord" contains only one pixel with component "value"
   *         false otherwise
   */
  bool isConnectedByOnePixel(const int2& size, const std::vector<unsigned char>& inputsMap, const unsigned char value,
                             const int2& coord, int2& connectedPoint);

  /**
   * @brief findDisconnectedComponent Find the connected component at a point.
   *                                  The connected component are wrapped around the vertical direction.
   */
  static void findDisconnectedComponent(const int wrapWidth, const int componentIndex, const int2& startPoint,
                                        const int2& offset, const int2& size, const std::vector<bool>& edges,
                                        const int2& inputsOffset, const int2& inputsSize,
                                        const std::vector<unsigned char>& inputsMap, std::vector<int>& components);

  /**
   * @brief Find the output camera index map
   */
  Status findOutputsMap(const int bufferIndex, const std::vector<int>& components);

  /**
   * @brief Modify the cost function, enforces the path to go near a prefined set of pixels
   */
  Status addDistanceCost(const int supportDistance, const int2& size, const std::vector<int2>& cachedPoints,
                         const std::vector<float>& inputCost, std::vector<float>& outputCost);

  /**
   * @brief Modify the cost function, enforces the path to not go near the feathering area
   */
  Status addFeatheringCost(const int2& size, const std::vector<float>& inputCost, std::vector<float>& outputCost);

  /**
   * @brief Find the connected component at a point
   */
  static bool getCurveEnds(const int wrapWidth, const int2& startPoint, const unsigned char value, const int2& size,
                           const std::vector<unsigned char>& borders, std::vector<bool>& visited,
                           std::vector<std::vector<int2>>& curveCachedPoints,
                           std::vector<std::vector<int2>>& curveRootPoints, int& distanceCost);

  // All input buffers
  const int wrapWidth;
  const videoreaderid_t id0;
  const Core::Rect rect0;
  GPU::Buffer<const uint32_t> input0Buffer;
  const videoreaderid_t id1;
  const Core::Rect rect1;
  GPU::Buffer<const uint32_t> input1Buffer;
  GPU::Stream& stream;

  // Seam parameter
  const int kernelSize;
  const int seamFeatheringSize;

  std::vector<int2> startPoints;                           // Starting point of the computed seam
  std::vector<int2> endPoints;                             // Ending points of the computed seam
  std::vector<std::vector<unsigned char>> all_directions;  // Directions of the seam
  std::vector<int> distanceCosts;  // Whether the seam was constrained to stay close to the correspondent cachePoints
  std::vector<std::vector<int2>> all_cachePoints;  // The cached points, or the original border
  std::vector<bool> wrapPaths;                     // Whether the seam is allowed to wrap around the right border

  Core::Rect rect;
  // Intermediate buffers
  GPU::UniqueBuffer<unsigned char> outputsMapBuffer;
  GPU::UniqueBuffer<unsigned char> inputsMapBuffer;

  // Cost buffer in CPU memory
  std::vector<float> costs;

  // Temp buffer, ideally borrowed from other class
  GPU::SharedBuffer<unsigned char> workMask;
  GPU::SharedBuffer<uint32_t> work1;
  GPU::SharedBuffer<uint32_t> work2;
  GPU::UniqueBuffer<uint32_t> outputBuffer;
};

}  // namespace MergerMask
}  // namespace VideoStitch
