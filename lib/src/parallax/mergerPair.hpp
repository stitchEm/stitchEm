// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "core/rect.hpp"
#include "core/pyramid.hpp"
#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"
#include "gpu/memcpy.hpp"
#include "gpu/uniqueBuffer.hpp"

#include "libvideostitch/status.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/matrix.hpp"

#include <vector>

namespace VideoStitch {
namespace Core {

class PanoDefinition;
class ImageMapping;
class StereoRigDefinition;

template <class T>
class LaplacianPyramid;

/**
 * This class is used to pre-compute coordinate mappings.
 * Mapping is constructed in a multi-resolution manner.
 * It deals with 3 different coordinate spaces:
 *  1) Pano space or Output space: the output space of the stitched panorama.
 *  2) Intermediate space: the space used to calculate image flow.
 *     Intermediate space is constructed by rotating the original pole @oldCoord to the new pole @newCoord.
 *     It make sure that the two input buffer are mapped to the equator to minimize distortion.
 *  3) Input space: the space used for capturing the input videos
 */

class MergerPair {
 public:
  /**
   * This construction is used to construct coordinate mapping between an input image pair of certain size
   * It is only used for debugging purpose when the input pair will "NOT" be transformed into the intermediate space
   * @param boundingFirstLevelSize Bounding size of the first (largest) level.
   * @param boundingLastLevelSize Bounding size of the last (smallest) level.
   * @param width0 Width of the first image.
   * @param height0 Height of the first image.
   * @param offset0X Left coordinate of the first image.
   * @param offset0Y Top coordinate the first image.
   * @param width1 Width of the second image.
   * @param height1 Height of the second image.
   * @param offset1X Left coordinate of the first image.
   * @param offset1Y Top coordinate the first image.
   * @param useInterToPano This flag is turned off in debug mode, where no actually space transformation is done
   */
  MergerPair(const int boundingFirstLevelSize, const int boundingLastLevelSize, const int width0, const int height0,
             const int offset0X, const int offset0Y, const GPU::Buffer<const uint32_t>& buffer0, const int width1,
             const int height1, const int offset1X, const int offset1Y, const GPU::Buffer<const uint32_t>& buffer1,
             GPU::Stream stream);

  /**
   * Create a coordinate mapping between two image Ids from panoDef
   * @param panoDef The pano definition.
   * @param rigDef The rig definition.
   * @param boundingFirstLevelSize Bounding size of the first (largest) level.
   * @param boundingLastLevelSize Bounding size of the last (smallest) level.
   * @param id0 First image id.
   * @param id1 Second image id.
   * @param inBoundingPanoRect0 Bounding rect of the first image in pano space. Taken from ImageMapping of @id0.
   * @param inBoundingPanoRect1 Bounding rect of the second image in pano space. Taken from ImageMapping of @id1.
   * @param stream Where to do the computations.
   */
  static Potential<MergerPair> create(const PanoDefinition& panoDef, const StereoRigDefinition* rigDef,
                                      const int boundingFirstLevelSize, const int boundingLastLevelSize,
                                      const std::vector<videoreaderid_t>& id0s, const videoreaderid_t id1,
                                      const Rect& inBoundingPanoRect0, const Rect& inBoundingPanoRect1,
                                      GPU::Stream stream);

 private:
  MergerPair(const int boundingFirstLevelSize, const int boundingLastLevelSize,
             const std::vector<videoreaderid_t>& id0s, const videoreaderid_t id1);

 public:
  const Rect getBoundingInterRect(const int index, const int level) const;

  const LaplacianPyramid<float2>* getInterToInputSpaceCoordMappingLaplacianPyramid(const int index) const;

  int getWrapWidth() const;

  int getWrapHeight() const;

  bool doesOverlap() const;

  std::vector<Rect> getBoundingInterRect1s() const;

  GPU::Buffer<float2> getInterToLookupSpaceCoordMappingBufferLevel(const int index, const int level) const;

  const Rect getBoundingPanoRect(const int index) const;

  int2 getInput1Size() const;

  GPU::Buffer<float2> getPanoToInputSpaceCoordMapping(const int index) const;

  GPU::Buffer<float2> getPanoToInterSpaceCoordMapping(const int index) const;

  Rect getBoundingPanosIRect() const;

#ifndef VS_OPENCL
  /**
   * Get average spherical coordinate of the input image pair.
   * @param panoDef The pano definition.
   * @param id0 First image id.
   * @param id1 Second image id.
   */
  static Vector3<double> getAverageSphericalCoord(const PanoDefinition& panoDef, const videoreaderid_t id0,
                                                  const videoreaderid_t id1);

  static Vector3<double> getAverageSphericalCoord(
      const PanoDefinition& panoDef, const std::vector<videoreaderid_t>& id0s, const std::vector<videoreaderid_t>& id1s,
      const Rect& boundingPanoRect0, const GPU::Buffer<const float2>& panoToInputSpaceCoordMapping0,
      const GPU::Buffer<const uint32_t>& maskBuffer0, const Rect& boundingPanoRect1,
      const GPU::Buffer<const float2>& panoToInputSpaceCoordMapping1, const GPU::Buffer<const uint32_t>& maskBuffer1);

  /**
   * Set up a mask in the output space where (pixel value at x & (1 << id)) > 0
   * indicates that image id is projected to x in the output space
   */
  Status setupPairMappingMask(GPU::Buffer<uint32_t> devMask, GPU::Stream gpuStream) const;

  Status debugMergerPair(const int2 panoSize, const GPU::Buffer<const uint32_t> panoBuffer, const int2 bufferSize0,
                         const GPU::Buffer<const uint32_t> buffer0, const int2 bufferSize1,
                         const GPU::Buffer<const uint32_t> buffer1, GPU::Stream gpuStream) const;

  /**
   * Find coordinate mapping from one space (either pano or intermediate) to input space.
   * @param panoDef The pano definition.
   * @param rigDef The rig definition.
   * @param id Buffer Id.
   * @param oldCoord The original pole.
   * @param newCoord The new pole.
   * @param toInputMapping The output coordinate mapping buffer.
   * @param weight The output weight buffer, set to 1 only at valid pixels.
   * @param boundingRect The output bounding rect of @weight (and @toInputMapping).
   *                     Use the input rect  if @usePassedBoundingRect is true.
   * @param stream CUDA stream for the operation.
   * @param usePassedBoundingRect Specify whether @boundingRect is used as input or output.
   */
  static Status findMappingToInputSpace(const PanoDefinition& panoDef, const StereoRigDefinition* rigDef,
                                        const std::vector<videoreaderid_t>& ids, const Vector3<double>& oldCoord,
                                        const Vector3<double>& newCoord, GPU::UniqueBuffer<float2>& toInputMapping,
                                        GPU::UniqueBuffer<uint32_t>& weight, Rect& boundingRect, GPU::Stream stream,
                                        const bool usePassedBoundingRect = false);

  std::string getImIdString(const int index) const;

  bool UseInterToPano() const;

 private:
  Status init(const PanoDefinition& panoDef, const StereoRigDefinition* rigDef, const Rect& inBoundingPanoRect0,
              const Rect& inBoundingPanoRect1, GPU::Stream stream);

  /**
   * Find coordinate mapping from pano space to intermediate space.
   * Intermediate space is constructed by rotating the original pole @oldCoord to the new pole @newCoord.
   * @param panoDef The pano definition.
   * @param downRatio The bounding rect of the first buffer.
   * @param id Buffer Id.
   * @param oldCoord The original pole.
   * @param newCoord The new pole.
   * @param panoToInputSpaceCoordMapping Buffer that contain coordinate mapping from pano to input space.
   * @param boundingPanoRect The bounding rect of @panoToInterSpaceCoordMapping in pano space.
   * @param panoToInterSpaceCoordMapping The output mapping from pano to intermediate space.
   * @param stream CUDA stream for the operation.
   */
  static Status findMappingFromPanoToInterSpace(const PanoDefinition& panoDef, const float downRatio,
                                                const std::vector<videoreaderid_t>& ids,
                                                const Vector3<double>& oldCoord, const Vector3<double>& newCoord,
                                                const GPU::Buffer<const float2>& panoToInputSpaceCoordMapping,
                                                const GPU::Buffer<const uint32_t>& panoToInputSpaceMask,
                                                const Rect& boundingPanoRect,
                                                GPU::UniqueBuffer<float2>& panoToInterSpaceCoordMapping,
                                                GPU::Stream stream);

  static Status findMappingFromInterToPanoSpace(const PanoDefinition& panoDef, const std::vector<videoreaderid_t>& ids,
                                                const GPU::Buffer<const float2>& interToInputSpaceCoordMapping,
                                                const GPU::Buffer<const uint32_t>& interToInputSpaceMask,
                                                const Rect& boundingInterRect,
                                                GPU::UniqueBuffer<float2>& interToPanoSpaceCoordMapping,
                                                GPU::Stream stream);

  /**
   * Given the input buffer pair, calculate pyramid info,
   * so that the first and last level follow the bounding size of @boundingFirstLevelSize and @boundingLastLevelSize
   * @param downRatio The level of down sampling.
   * @param coord0Mapping Coordinate mapping of the first buffer. Store as the down-grade results as well
   * @param weight0 Weight of the first buffer. Store as the down-grade result as well.
   * @param boundingRect0 The output bounding box of the first buffer.
   * @param coord1Mapping Coordinate mapping of the second buffer. Store as the down-grade results as well
   * @param weight1 Weight of the second buffer. Store as the down-grade result as well.
   * @param boundingRect1 The output bounding box of the second buffer.
   * @param stream CUDA stream for the operation.
   */
  Status calculateLaplacianPyramidsInfo(float& downRatio, GPU::UniqueBuffer<float2>& coord0Mapping,
                                        GPU::UniqueBuffer<uint32_t>& weight0, Rect& boundingRect0,
                                        GPU::UniqueBuffer<float2>& coord1Mapping, GPU::UniqueBuffer<uint32_t>& weight1,
                                        Rect& boundingRect1, GPU::Stream stream);

  /**
   * Build the pyramid to map intermediate space to input space.
   * @param downRatio The level of down sampling.
   * @param coord0Mapping Coordinate mapping of the first buffer. Store as the down-grade results as well
   * @param weight0 Weight of the first buffer. Store as the down-grade result as well.
   * @param boundingRect0 The output bounding box of the first buffer.
   * @param coord1Mapping Coordinate mapping of the second buffer. Store as the down-grade results as well
   * @param weight1 Weight of the second buffer. Store as the down-grade result as well.
   * @param boundingRect1 The output bounding box of the second buffer.
   * @param stream CUDA stream for the operation.
   */
  Status buildLaplacianPyramids(const PanoDefinition& panoDef, float& downRatio,
                                GPU::UniqueBuffer<float2>& coord0Mapping, GPU::UniqueBuffer<uint32_t>& weight0,
                                Rect& boundingRect0, GPU::UniqueBuffer<float2>& coord1Mapping,
                                GPU::UniqueBuffer<uint32_t>& weight1, Rect& boundingRect1, GPU::Stream stream);

  static Status packCoordBuffer(const int warpWidth, const Core::Rect& inputRect,
                                const GPU::Buffer<const float2>& inputBuffer,
                                const GPU::Buffer<const uint32_t>& inputWeight, const Core::Rect& outputRect,
                                GPU::Buffer<float2> outputBuffer, GPU::Buffer<uint32_t> outputWeight,
                                GPU::Stream gpuStream);
#endif
 private:
  /**
   * Need a mapping from inter space to pano space to read result from previous warp step
   */
  // A mapping from the intermediate space to the pano space
  std::unique_ptr<LaplacianPyramid<float2>> interToPanoSpaceCoordMappingLaplacianPyramid0;
  std::unique_ptr<LaplacianPyramid<uint32_t>> interToPanoSpaceWeightLaplacianPyramid0;
  std::vector<Rect> boundingInterToPanoRect0s;

  /**
   * Intermediate to input space pyramid
   */
  std::unique_ptr<LaplacianPyramid<float2>> interToInputSpaceCoordMappingLaplacianPyramid0;
  std::unique_ptr<LaplacianPyramid<uint32_t>> interToInputSpaceWeightLaplacianPyramid0;
  std::vector<Rect> boundingInterRect0s;

  std::unique_ptr<LaplacianPyramid<float2>> interToInputSpaceCoordMappingLaplacianPyramid1;
  std::unique_ptr<LaplacianPyramid<uint32_t>> interToInputSpaceWeightLaplacianPyramid1;
  std::vector<Rect> boundingInterRect1s;

  const std::vector<videoreaderid_t> id0s;
  const videoreaderid_t id1;
  const float extendedRatio;
  const bool overlappedAreaOnly;
  const int boundingFirstLevelSize;
  const int boundingLastLevelSize;
  uint64_t wrapWidth;
  uint64_t wrapHeight;

  int2 input1Size;

  GPU::UniqueBuffer<float2> panoToInputSpaceCoordMapping0;
  GPU::UniqueBuffer<float2> panoToInterSpaceCoordMapping0;
  Rect boundingPanoRect0;

  GPU::UniqueBuffer<float2> panoToInputSpaceCoordMapping1;
  GPU::UniqueBuffer<float2> panoToInterSpaceCoordMapping1;
  GPU::UniqueBuffer<float2> interToInputCoord;
  Rect boundingPanoRect1;

  GPU::UniqueBuffer<float2> inputToPanoCoordMapping0;
  GPU::UniqueBuffer<float2> panoToInputCoordMapping0;
  const bool useInterToPano;
};
}  // namespace Core
}  // namespace VideoStitch
