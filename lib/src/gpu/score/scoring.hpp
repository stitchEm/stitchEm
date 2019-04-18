// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

namespace VideoStitch {
namespace Image {

Status splitNoBlendImageMergerChannel(GPU::Buffer<float> dest_r, GPU::Buffer<float> dest_g,
                                      GPU::Buffer<unsigned char> dest_b, GPU::Buffer<const uint32_t> source,
                                      const unsigned width, const unsigned height, GPU::Stream stream);

}
}  // namespace VideoStitch
