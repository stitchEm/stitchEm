// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <gpu/buffer.hpp>
#include <gpu/stream.hpp>

namespace VideoStitch {
namespace Core {

/**
 * This kernel computes the OR of all pixels in each row, and puts the result in colHasImage
 */
Status vertOr(std::size_t croppedWidth, std::size_t croppedHeight, GPU::Buffer<const uint32_t> contrib,
              GPU::Buffer<uint32_t> colHasImage, GPU::Stream stream);

Status horizOr(std::size_t croppedWidth, std::size_t croppedHeight, GPU::Buffer<const uint32_t> contrib,
               GPU::Buffer<uint32_t> rowHasImage, GPU::Stream stream);

}  // namespace Core
}  // namespace VideoStitch
