// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/image/blur.hpp"

#include "../kernel.hpp"

namespace {
#include "blur.xxd"
}

INDIRECT_REGISTER_OPENCL_PROGRAM(blur, true);

#include "backend/common/image/blurdef.h"

namespace VideoStitch {
namespace Image {
/**
 * Blur @buf with a gaussian filter of radius @radius. @work must be at least as big as @buf.
 * @passes is the number of box filtering passes and must be even (performance reasons).
 */
Status gaussianBlur2D(GPU::Buffer<unsigned char> /*buf*/, GPU::Buffer<unsigned char> /*work*/, std::size_t /*width*/,
                      std::size_t /*height*/, unsigned /*radius*/, unsigned /*passes*/, bool /*wrap*/,
                      unsigned /*blockSize*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::Stitcher, ErrType::UnsupportedAction, "2D gaussian blur not implemented in OpenCL backend"};
}

/**
 * Blur @buf with a gaussian filter of radius @radius. @work must be at least as big as @buf.
 * @passes is the number of box filtering passes and must be even (performance reasons).
 */
template <typename T>
Status gaussianBlur2D(GPU::Buffer<T> /*dst*/, GPU::Buffer<const T> /*src*/, GPU::Buffer<T> /*work*/,
                      std::size_t /*width*/, std::size_t /*height*/, unsigned /*radius*/, unsigned /*passes*/,
                      bool /*wrap*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::Stitcher, ErrType::UnsupportedAction, "2D gaussian blur not implemented in OpenCL backend"};
}

/**
 * Blur @buf with a gaussian filter of radius @radius. @work must be at least as big as @buf.
 * @passes is the number of box filtering passes and must be even (performance reasons).
 */
template <>
Status gaussianBlur2D(GPU::Buffer<unsigned char> /*dst*/, GPU::Buffer<const unsigned char> /*src*/,
                      GPU::Buffer<unsigned char> /*work*/, std::size_t /*width*/, std::size_t /*height*/,
                      unsigned /*radius*/, unsigned /*passes*/, bool /*wrap*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::Stitcher, ErrType::UnsupportedAction, "2D gaussian blur not implemented in OpenCL backend"};
}

/**
 * Blur @buf with a gaussian filter of radius @radius. @work must be at least as big as @buf.
 * @passes is the number of box filtering passes and must be even (performance reasons).
 */
template <>
Status gaussianBlur2D(GPU::Buffer<float2> /*dst*/, GPU::Buffer<const float2> /*src*/, GPU::Buffer<float2> /*work*/,
                      std::size_t /*width*/, std::size_t /*height*/, unsigned /*radius*/, unsigned /*passes*/,
                      bool /*wrap*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::Stitcher, ErrType::UnsupportedAction, "2D gaussian blur not implemented in OpenCL backend"};
}

static size_t ceil(size_t v, size_t d) {
  const size_t res = v / d;
  const size_t group = res + (v - res * d > 0);  // add one if the remainder is nonzero
  return group * d;
}

Status gaussianBlur2DRGBA(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src, GPU::Buffer<uint32_t> work,
                          std::size_t width, std::size_t height, unsigned /*radius*/, unsigned /*passes*/, bool wrap,
                          GPU::Stream stream) {
  {
    auto kernel2D = GPU::Kernel::get(PROGRAM(blur), KERNEL_STR(convolutionRowsKernel))
                        .setup2D(stream, (unsigned)ceil(width, ROWS_RESULT_STEPS), (unsigned)height, ROWS_BLOCKDIM_X,
                                 ROWS_BLOCKDIM_Y);
    FAIL_RETURN(kernel2D.enqueueWithKernelArgs(work, src, (unsigned)width, (int)wrap));
  }
  {
    auto kernel2D = GPU::Kernel::get(PROGRAM(blur), KERNEL_STR(convolutionColumnsKernel))
                        .setup2D(stream, (unsigned)width, (unsigned)ceil(height, COLUMNS_RESULT_STEPS),
                                 COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
    FAIL_RETURN(kernel2D.enqueueWithKernelArgs(dst, work, (unsigned)height, (unsigned)width));
  }
  return Status::OK();
}

/**
 * Small-support optimized version.
 * In-place.
 */
Status gaussianBlur2DRGBASS(GPU::Buffer<uint32_t> /*buf*/, uint32_t* /*work*/, std::size_t /*width*/,
                            std::size_t /*height*/, unsigned /*radius*/, bool /*wrap*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::Stitcher, ErrType::UnsupportedAction,
          "2D gaussian blur not implemented in OpenCL backend for RGBA color space"};
}

/**
 * Specialized gaussian blur:
 *  - Applies to an RGBA210 formatted buffer, colors are blurred independently.
 *  - Output is written to dst.
 * Passes need not be even nor odd.
 */
Status gaussianBlur2DRGBA210(GPU::Buffer<uint32_t> /*dst*/, GPU::Buffer<const uint32_t> /*src*/,
                             GPU::Buffer<uint32_t> /*work*/, std::size_t /*width*/, std::size_t /*height*/,
                             unsigned /*radius*/, unsigned /*passes*/, bool /*wrap*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::Stitcher, ErrType::UnsupportedAction,
          "2D gaussian blur not implemented in OpenCL backend for RGBA210 color space"};
}

/**
 * Small-support optimized version.
 * In-place.
 */
Status gaussianBlur2DRGBA210SS(GPU::Buffer<uint32_t> /*buf*/, uint32_t* /*work*/, std::size_t /*width*/,
                               std::size_t /*height*/, unsigned /*radius*/, bool /*wrap*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::Stitcher, ErrType::UnsupportedAction,
          "2D gaussian blur not implemented in OpenCL backend for RGBA210SS color space"};
}

template <typename T>
Status boxBlur1DNoWrap(GPU::Buffer<T> /*dst*/, GPU::Buffer<const T> /*src*/, std::size_t /*width*/,
                       std::size_t /*height*/, unsigned /*radius*/, unsigned /*blockSize*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::Stitcher, ErrType::UnsupportedAction, "1D blur not implemented in OpenCL backend"};
}

Status boxBlur1DNoWrapRGBA210(GPU::Buffer<uint32_t> /*dst*/, GPU::Buffer<const uint32_t> /*src*/, std::size_t /*width*/,
                              std::size_t /*height*/, unsigned /*radius*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::Stitcher, ErrType::UnsupportedAction,
          "1D blur not implemented in OpenCL backend for RGBA210 color space"};
}

Status boxBlur1DWrapRGBA210(GPU::Buffer<uint32_t> /*dst*/, GPU::Buffer<const uint32_t> /*src*/, std::size_t /*width*/,
                            std::size_t /*height*/, unsigned /*radius*/, GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::Stitcher, ErrType::UnsupportedAction,
          "1D blur not implemented in OpenCL backend for RGBA210 color space"};
}

Status gaussianBlur1DRGBA210SS(GPU::Buffer<uint32_t> /*dst*/, GPU::Buffer<const uint32_t> /*src*/,
                               std::size_t /*width*/, std::size_t /*height*/, unsigned /*radius*/, bool /*wrap*/,
                               GPU::Stream /*stream*/) {
  // TODO_OPENCL_IMPL
  return {Origin::Stitcher, ErrType::UnsupportedAction,
          "1D blur not implemented in OpenCL backend for RGBA210SS color space"};
}

}  // namespace Image
}  // namespace VideoStitch
