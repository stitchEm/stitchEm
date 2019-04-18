// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/2dBuffer.hpp"
#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"

namespace VideoStitch {
namespace Image {

/**
 * This kernel computes the RGB histograms of a RGBA8888 video frame.
 */
Status rgbHistogram(GPU::Buffer<const uint32_t> frame, int64_t width, int64_t height, GPU::Buffer<const uint32_t> rHist,
                    GPU::Buffer<const uint32_t> gHist, GPU::Buffer<const uint32_t> bHist, GPU::Stream stream);

/*
 * This kernel computes the relative distribution of all luma values in a grayscale video frame.
 */
Status lumaHistogram(GPU::Buffer2D frame, frameid_t frameId, GPU::Buffer<uint32_t> hist, GPU::Stream stream);
}  // namespace Image
}  // namespace VideoStitch
