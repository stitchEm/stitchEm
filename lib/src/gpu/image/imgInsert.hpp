// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef IMGINSERT_HPP_
#define IMGINSERT_HPP_

#include <stdint.h>

#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"

namespace VideoStitch {
namespace Image {

Status imgInsertInto(GPU::Buffer<uint32_t> dst, std::size_t dstWidth, std::size_t dstHeight,
                     GPU::Buffer<const uint32_t> src, std::size_t srcWidth, std::size_t srcHeight, std::size_t offsetX,
                     std::size_t offsetY, GPU::Buffer<const unsigned char> mask, bool hWrap, bool vWrap,
                     GPU::Stream stream);

Status imgInsertInto10bit(GPU::Buffer<uint32_t> dst, std::size_t dstWidth, std::size_t dstHeight,
                          GPU::Buffer<const uint32_t> src, std::size_t srcWidth, std::size_t srcHeight,
                          std::size_t offsetX, std::size_t offsetY, GPU::Buffer<const unsigned char> mask, bool hWrap,
                          bool vWrap, GPU::Stream stream);

}  // namespace Image
}  // namespace VideoStitch

#endif
