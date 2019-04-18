// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef FILL_HPP_
#define FILL_HPP_

#include "gpu/stream.hpp"
#include <stdint.h>

namespace VideoStitch {
namespace Image {
/**
 * Dumbest kernel. Ever.
 */
Status fill(uint32_t *dst, int64_t width, int64_t height, int32_t color, GPU::Stream stream);
}  // namespace Image
}  // namespace VideoStitch

#endif
