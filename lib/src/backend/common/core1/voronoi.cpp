// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "voronoiKernel.hpp"

// #define DEBUGMASKS
#ifdef DEBUGMASKS
#ifndef _MSC_VER
static const std::string DEBUG_FOLDER = "/tmp/voronoi/";
#else
static const std::string DEBUG_FOLDER = "";
#endif
#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif
#include "util/debugUtils.hpp"
#include <sstream>
#endif

namespace VideoStitch {
namespace Core {

namespace {

/**
 * Returns the largest power of two smaller than @a v.
 */
unsigned largestPowerOfTwoLessThan(unsigned v) {
  unsigned res = 1;
  while (res < v) {
    res *= 2;
  }
  return res / 2;
}

}  // namespace

// returns pointer to destination of last step
PotentialValue<GPU::Buffer<uint32_t>> distanceMapErect(GPU::Buffer<uint32_t> src, GPU::Buffer<uint32_t> work,
                                                       PanoRegion region, bool hWrap, GPU::Stream stream,
                                                       unsigned blockSize = 16) {
  GPU::Buffer<uint32_t> tmpSrc = src;
  GPU::Buffer<uint32_t> tmpDst = work;

  for (unsigned step = largestPowerOfTwoLessThan((unsigned)std::max(region.viewWidth, region.viewHeight)); step > 0;
       step /= 2) {
    FAIL_RETURN(voronoiComputeErect(tmpDst, tmpSrc, region, step, hWrap, blockSize, stream));
    std::swap(tmpDst, tmpSrc);
  }

  return tmpSrc;
}

PotentialValue<GPU::Buffer<uint32_t>> distanceMap(GPU::Buffer<uint32_t> src, GPU::Buffer<uint32_t> work,
                                                  std::size_t width, std::size_t height, bool hWrap, GPU::Stream stream,
                                                  unsigned blockSize = 16) {
  GPU::Buffer<uint32_t> tmpSrc = src;
  GPU::Buffer<uint32_t> tmpDst = work;

  for (unsigned step = largestPowerOfTwoLessThan((unsigned)std::max(width, height)); step > 0; step /= 2) {
    FAIL_RETURN(voronoiComputeEuclidean(tmpDst, tmpSrc, width, height, step, hWrap, blockSize, stream));
    std::swap(tmpDst, tmpSrc);
  }

  return tmpSrc;
}

Status voronoiCompute(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> src, GPU::Buffer<uint32_t> work,
                      std::size_t width, std::size_t height, uint32_t fromIdMask, uint32_t toIdMask, bool hWrap,
                      unsigned blockSize, GPU::Stream stream) {
  FAIL_RETURN(voronoiInit(src, width, height, toIdMask, fromIdMask, blockSize, stream));

  auto tmpDst = distanceMap(src, work, width, height, hWrap, stream, blockSize);
  FAIL_RETURN(tmpDst.status());

  return voronoiMakeMask(dst, tmpDst.value(), width, height, blockSize, stream);
}

Status computeMask(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> src, GPU::Buffer<uint32_t> workBuffer1,
                   GPU::Buffer<uint32_t> workBuffer2, const PanoRegion& region, uint32_t fromIdMask, uint32_t toIdMask,
                   bool hWrap, float maxTransitionDistance, float power, GPU::Stream stream) {
  const auto blackWork = workBuffer1;
  const auto whiteWork = workBuffer2;

  // Extract base distance maps.
  FAIL_RETURN(
      initForMaskComputation(blackWork, src, region.viewWidth, region.viewHeight, fromIdMask, toIdMask, stream));
  FAIL_RETURN(
      initForMaskComputation(whiteWork, src, region.viewWidth, region.viewHeight, toIdMask, fromIdMask, stream));

#ifdef DEBUGMASKS
  {
    stream.synchronize();
    const std::string prefix =
        DEBUG_FOLDER + "computeMask-" + std::to_string(fromIdMask) + "-" + std::to_string(toIdMask) + "-";
    Debug::dumpRGBADeviceBuffer((prefix + "blackInit.png").c_str(), blackWork.as_const(), region.viewWidth,
                                region.viewHeight);
    Debug::dumpRGBADeviceBuffer((prefix + "whiteInit.png").c_str(), whiteWork.as_const(), region.viewWidth,
                                region.viewHeight);
  }
#endif  // DEBUGMASKS

  // Process black.
  const auto blackResult = distanceMapErect(blackWork, src, region, hWrap, stream);
  FAIL_RETURN(blackResult.status());

  const auto workBuffer = (blackResult.value() == blackWork) ? src : blackWork;

  // Process white.
  const auto whiteResult = distanceMapErect(whiteWork, workBuffer, region, hWrap, stream);
  FAIL_RETURN(whiteResult.status());

#ifdef DEBUGMASKS
  {
    stream.synchronize();
    const std::string prefix =
        DEBUG_FOLDER + "computeMask-" + std::to_string(fromIdMask) + "-" + std::to_string(toIdMask) + "-";
    Debug::dumpRGBADeviceBuffer((prefix + "blackResult.png").c_str(), blackResult.value(), region.viewWidth,
                                region.viewHeight);
    Debug::dumpRGBADeviceBuffer((prefix + "whiteResult.png").c_str(), whiteResult.value(), region.viewWidth,
                                region.viewHeight);
  }
#endif  // DEBUGMASKS

  return makeMaskErect(dst, blackResult.value(), whiteResult.value(), region, hWrap, maxTransitionDistance, power,
                       stream);
}

Status computeEuclideanDistanceMap(GPU::Buffer<unsigned char> dst, GPU::Buffer<const uint32_t> src,
                                   GPU::Buffer<uint32_t> work1, GPU::Buffer<uint32_t> work2, std::size_t width,
                                   std::size_t height, uint32_t fromIdMask, uint32_t toIdMask, bool hWrap,
                                   float maxTransitionDistance, float power, GPU::Stream stream) {
  auto whiteWork = work2;

  // Extract base distance maps.
  FAIL_RETURN(initForMaskComputation(whiteWork, src, width, height, toIdMask, fromIdMask, stream));
  auto workBuffer = work1;
  // Process white.
  auto whiteResult = distanceMap(whiteWork, workBuffer, width, height, hWrap, stream);
  FAIL_RETURN(whiteResult.status());

  return extractEuclideanDist(dst, whiteResult.value(), width, height, hWrap, maxTransitionDistance, power, stream);
}

Status computeEuclideanDistanceMap(GPU::Buffer<unsigned char> dst, GPU::Buffer<const uint32_t> src,
                                   GPU::Buffer<uint32_t> work1, GPU::Buffer<uint32_t> work2, std::size_t width,
                                   std::size_t height, uint32_t idMask, bool hWrap, float maxTransitionDistance,
                                   float power, GPU::Stream stream) {
  return computeEuclideanDistanceMap(dst, src, work1, work2, width, height, 0, idMask, hWrap, maxTransitionDistance,
                                     power, stream);
}

}  // namespace Core
}  // namespace VideoStitch
