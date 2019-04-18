// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/core1/voronoi.hpp"

#include "../kernel.hpp"

namespace VideoStitch {
namespace Core {

namespace {
#include "voronoi.xxd"
}

INDIRECT_REGISTER_OPENCL_PROGRAM(voronoi, true);

Status setInitialImageMask(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> src, std::size_t width,
                           std::size_t height, uint32_t fromIdMask, GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(voronoi), KERNEL_STR(edtReflexiveKernel))
                      .setup2D(stream, (unsigned)width, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(dst, src, (unsigned)width, (unsigned)height, fromIdMask);
}

Status voronoiInit(GPU::Buffer<uint32_t> buffer, std::size_t width, std::size_t height, uint32_t blackMask,
                   uint32_t whiteMask, unsigned blockSize, GPU::Stream stream) {
  auto voronoiInit = GPU::Kernel::get(PROGRAM(voronoi), KERNEL_STR(voronoiInitKernel))
                         .setup2D(stream, (unsigned)width, (unsigned)height, blockSize, blockSize);
  return voronoiInit.enqueueWithKernelArgs(buffer.get(), (unsigned)width, (unsigned)height, blackMask, whiteMask);
}

Status voronoiComputeEuclidean(GPU::Buffer<uint32_t> dst, GPU::Buffer<uint32_t> src, std::size_t width,
                               std::size_t height, uint32_t step, bool hWrap, unsigned blockSize, GPU::Stream stream) {
  std::string voronoiComputeVariant;

  if (hWrap) {
    voronoiComputeVariant = KERNEL_STR(voronoiCompute_Wrap_distSqr);
  } else {
    voronoiComputeVariant = KERNEL_STR(voronoiCompute_NoWrap_distSqr);
  }

  PanoRegion region;
  region.panoDim = {-1};
  region.viewLeft = 0;
  region.viewTop = 0;
  region.viewWidth = (int32_t)width;
  region.viewHeight = (int32_t)height;

  auto voronoiCompute = GPU::Kernel::get(PROGRAM(voronoi), voronoiComputeVariant)
                            .setup2D(stream, (unsigned)width, (unsigned)height, blockSize);
  return voronoiCompute.enqueueWithKernelArgs(dst, src, region, step);
}

Status voronoiComputeErect(GPU::Buffer<uint32_t> dst, GPU::Buffer<uint32_t> src, const PanoRegion& region,
                           uint32_t step, bool hWrap, unsigned blockSize, GPU::Stream stream) {
  std::string voronoiComputeVariant;

  if (hWrap) {
    voronoiComputeVariant = KERNEL_STR(voronoiCompute_Wrap_distSphere);
  } else {
    voronoiComputeVariant = KERNEL_STR(voronoiCompute_NoWrap_distSphere);
  }

  auto voronoiCompute = GPU::Kernel::get(PROGRAM(voronoi), voronoiComputeVariant)
                            .setup2D(stream, region.viewWidth, region.viewHeight, blockSize);
  return voronoiCompute.enqueueWithKernelArgs(dst, src, region, step);
}

Status voronoiMakeMask(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> src, std::size_t width, std::size_t height,
                       unsigned blockSize, GPU::Stream stream) {
  auto voronoiMakeMask = GPU::Kernel::get(PROGRAM(voronoi), KERNEL_STR(voronoiMakeMaskKernel))
                             .setup2D(stream, (unsigned)width, (unsigned)height, blockSize, blockSize);
  return voronoiMakeMask.enqueueWithKernelArgs(dst.get(), src, (unsigned)width, (unsigned)height);
}

Status initForMaskComputation(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> buf, std::size_t width,
                              std::size_t height, uint32_t mask, uint32_t otherMask, GPU::Stream stream) {
  auto edtInit =
      GPU::Kernel::get(PROGRAM(voronoi), KERNEL_STR(edtInit)).setup2D(stream, (unsigned)width, (unsigned)height);
  return edtInit.enqueueWithKernelArgs(dst, buf, (unsigned)width, (unsigned)height, mask, otherMask);
}

Status makeMaskErect(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> blackResult,
                     GPU::Buffer<uint32_t> whiteResult, const PanoRegion& region, bool hWrap,
                     float maxTransitionDistance, float power, GPU::Stream stream) {
  const auto buildMaskVariant =
      (hWrap ? KERNEL_STR(buildTransitionMask_Wrap_distSphere) : KERNEL_STR(buildTransitionMask_NoWrap_distSphere));

  auto buildMask =
      GPU::Kernel::get(PROGRAM(voronoi), buildMaskVariant).setup2D(stream, region.viewWidth, region.viewHeight);
  return buildMask.enqueueWithKernelArgs(dst, blackResult, whiteResult, region, maxTransitionDistance, power);
}

Status makeMaskEuclidean(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> blackResult,
                         GPU::Buffer<uint32_t> whiteResult, const PanoRegion& region, bool hWrap,
                         float maxTransitionDistance, float power, GPU::Stream stream) {
  const auto makeMaskVariant =
      (hWrap ? KERNEL_STR(buildTransitionMask_Wrap_distSqr) : KERNEL_STR(buildTransitionMask_NoWrap_distSqr));

  auto makeMask =
      GPU::Kernel::get(PROGRAM(voronoi), makeMaskVariant).setup2D(stream, region.viewWidth, region.viewHeight);
  return makeMask.enqueueWithKernelArgs(dst, blackResult, whiteResult, region, maxTransitionDistance, power);
}

Status extractEuclideanDist(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> whiteResult, std::size_t width,
                            std::size_t height, bool hWrap, float maxTransitionDistance, float power,
                            GPU::Stream stream) {
  const auto extractDistVariant =
      (hWrap ? KERNEL_STR(extractDistKernel_Wrap_distSqr) : KERNEL_STR(extractDistKernel_NoWrap_distSqr));

  auto extract =
      GPU::Kernel::get(PROGRAM(voronoi), extractDistVariant).setup2D(stream, (unsigned)width, (unsigned)height);
  return extract.enqueueWithKernelArgs(dst, whiteResult, (unsigned)width, (unsigned)height, maxTransitionDistance,
                                       power);
}

}  // namespace Core
}  // namespace VideoStitch
