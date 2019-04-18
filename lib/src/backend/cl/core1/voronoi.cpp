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
GPU::Buffer<uint32_t> distanceMap(GPU::Buffer<uint32_t> src, GPU::Buffer<uint32_t> work, std::size_t width,
                                  std::size_t height, bool hWrap, GPU::Stream stream) {
  GPU::Buffer<uint32_t> tmpSrc = src;
  GPU::Buffer<uint32_t> tmpDst = work;

  std::string voronoiComputeVariant;

  if (hWrap) {
    voronoiComputeVariant = KERNEL_STR(voronoiComputeWrap);
  } else {
    voronoiComputeVariant = KERNEL_STR(voronoiComputeNoWrap);
  }

  auto voronoiCompute =
      GPU::Kernel::get(PROGRAM(voronoi), voronoiComputeVariant).setup2D(stream, (unsigned)width, (unsigned)height);
  for (unsigned step = largestPowerOfTwoLessThan((unsigned)std::max(width, height)); step > 0; step /= 2) {
    const Status computeStatus =
        voronoiCompute.enqueueWithKernelArgs(tmpDst, tmpSrc, (unsigned)width, (unsigned)height, step);
    assert(computeStatus.ok());
    std::swap(tmpDst, tmpSrc);
  }

  return tmpSrc;
}

/**
 * Compute the generalized voronoi diagram of @a src.
 * @param dst Output buffer for the voronoi diagram. Only two values: 0 and 255.
 * @param src Source buffer containing a setup image (i.e. the i-th bit of a pixel represents the i-th input).
 * @param work A work buffer.
 * @param width Width of the previous buffers.
 * @param height Height of the previous buffers.
 * @param fromIdMask Bit mask of the first input (e.g. if 0x00000004, the first input will be input 2, starting at 0).
 * @param toIdMask Bit mask of the second input.
 * @param hWrap If true, we consider the buffer to wrap horizontally.
 * @param stream CUDA stream where to run the kernels.
 * @note This call is asynchronous.
 */
void voronoiCompute(unsigned char* /*dst*/, uint32_t* /*src*/, uint32_t* /*work*/, std::size_t /*width*/,
                    std::size_t /*height*/, uint32_t /*fromIdMask*/, uint32_t /*toIdMask*/, bool /*hWrap*/,
                    unsigned /*blockSize*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
}

/**
 * Compute the euclidian distance transform of src.
 * @param dst Output buffer for the voronoi diagram. Output values are in [0;255].
 * @param src Source buffer containing a setup image (i.e. the i-th bit of a pixel represents the i-th input).
 * @param work A work buffer. Twice the size of @a src.
 * @param width Width of the previous buffers.
 * @param height Height of the previous buffers.
 * @param fromIdMask Bit mask of the first input (e.g. if 0x00000004, the first input will be input 2, starting at 0).
 * @param toIdMask Bit mask of the second input.
 * @param hWrap If true, we consider the buffer to wrap horizontally.
 * @param maxTransitionDistance maximum width of the transition / overlap.
 * @param power parameter of the p-norm that's used to calculate the transition. Should be >= 2.0 to use at least L2.
 * Steeper transition with larger power.
 * @param stream CUDA stream where to run the kernels.
 * @note This call is asynchronous.
 */
Status edtCompute(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> src, GPU::Buffer<uint32_t> workBuffer1,
                  GPU::Buffer<uint32_t> workBuffer2, std::size_t width, std::size_t height, uint32_t fromIdMask,
                  uint32_t toIdMask, bool hWrap, int maxTransitionDistance, float power, GPU::Stream stream) {
  // TODO_OPENCL_IMPL merge this with CUDA code, create backend shared header, impl

  const auto blackWork = workBuffer1;
  const auto whiteWork = workBuffer2;

  // dim3 dimBlock2D(blockSize, blockSize, 1);
  // // FIXME: make sure this holds ?
  // assert((width % dimBlock2D.x) == 0);
  // assert((height % dimBlock2D.x) == 0);
  // dim3 dimGrid2D((unsigned)width / dimBlock2D.x, (unsigned)height / dimBlock2D.y, 1);

  auto edtInit =
      GPU::Kernel::get(PROGRAM(voronoi), KERNEL_STR(edtInit)).setup2D(stream, (unsigned)width, (unsigned)height);

  // Extract base distance maps.
  PROPAGATE_FAILURE_STATUS(
      edtInit.enqueueWithKernelArgs(blackWork, src, (unsigned)width, (unsigned)height, fromIdMask, toIdMask));
  PROPAGATE_FAILURE_STATUS(
      edtInit.enqueueWithKernelArgs(whiteWork, src, (unsigned)width, (unsigned)height, toIdMask, fromIdMask));

  // Process black.
  const auto blackResult = distanceMap(blackWork, src, width, height, hWrap, stream);

  const auto workBuffer = (blackResult == blackWork) ? src : blackWork;

  // Process white.
  const auto whiteResult = distanceMap(whiteWork, workBuffer, width, height, hWrap, stream);

  const auto edtMakeMaskVariant =
      (hWrap ? KERNEL_STR(edtMakeMaskKernel_extractDistWrap) : KERNEL_STR(edtMakeMaskKernel_extractDistNoWrap));

  auto edtMakeMask =
      GPU::Kernel::get(PROGRAM(voronoi), edtMakeMaskVariant).setup2D(stream, (unsigned)width, (unsigned)height);
  return edtMakeMask.enqueueWithKernelArgs(dst, blackResult, whiteResult, (unsigned)width, (unsigned)height,
                                           maxTransitionDistance, power);
}

/**
 * Compute the euclidian distance transform of src to src.
 * @param dst Output buffer for the voronoi diagram. Output values are in [0;255].
 * @param src Source buffer containing a setup image (i.e. the i-th bit of a pixel represents the i-th input).
 * @param width Width of the previous buffers.
 * @param height Height of the previous buffers.
 * @param fromIdMask Bit mask of the  input (e.g. if 0x00000004, the first input will be input 2, starting at 0).
 * @param stream CUDA stream where to run the kernels.
 * @note This call is asynchronous.
 */
Status edtReflexive(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> src, std::size_t width, std::size_t height,
                    uint32_t fromIdMask, GPU::Stream stream) {
  auto kernel2D = GPU::Kernel::get(PROGRAM(voronoi), KERNEL_STR(edtReflexiveKernel))
                      .setup2D(stream, (unsigned)width, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(dst, src, (unsigned)width, (unsigned)height, fromIdMask);
}

}  // namespace Core
}  // namespace VideoStitch
