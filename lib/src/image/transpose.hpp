// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef TRANSPOSE_HPP_
#define TRANSPOSE_HPP_

#include <gpu/stream.hpp>

namespace VideoStitch {
namespace Image {
template <typename T>
Status transpose(T *dst, const T *src, int64_t w, int64_t h, GPU::Stream &stream);
}
}  // namespace VideoStitch

#endif
