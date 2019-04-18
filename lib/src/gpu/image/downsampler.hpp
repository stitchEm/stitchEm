// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/config.hpp"
#include "libvideostitch/frame.hpp"

#include "gpu/stream.hpp"

namespace VideoStitch {

namespace GPU {
class Surface;
class Buffer2D;
}  // namespace GPU

namespace Image {

Status downsample(PixelFormat fmt, GPU::Buffer2D* planesIn, GPU::Buffer2D* planesOut, GPU::Stream stream);

Status downsample(GPU::Buffer2D& in, GPU::Buffer2D& out, GPU::Stream stream);

Status downsampleRGBASurf2x(GPU::Surface& dst, const GPU::Surface& src, unsigned dstWidth, unsigned dstHeight,
                            GPU::Stream stream);

Status downsampleRGBA(int factor, GPU::Buffer2D& in, GPU::Buffer2D& out, GPU::Stream stream);
Status downsampleRGB(int factor, GPU::Buffer2D& in, GPU::Buffer2D& out, GPU::Stream stream);
Status downsampleYUV422(int factor, GPU::Buffer2D& in, GPU::Buffer2D& out, GPU::Stream stream);
Status downsampleYUV422P10(int factor, GPU::Buffer2D& yIn, GPU::Buffer2D& uIn, GPU::Buffer2D& vIn, GPU::Buffer2D& yOut,
                           GPU::Buffer2D& uOut, GPU::Buffer2D& vOut, GPU::Stream stream);
Status downsampleYV12(int factor, GPU::Buffer2D& yIn, GPU::Buffer2D& uIn, GPU::Buffer2D& vIn, GPU::Buffer2D& yOut,
                      GPU::Buffer2D& uOut, GPU::Buffer2D& vOut, GPU::Stream stream);
Status downsampleNV12(int factor, GPU::Buffer2D& yIn, GPU::Buffer2D& uvIn, GPU::Buffer2D& yOut, GPU::Buffer2D& uvOut,
                      GPU::Stream stream);

}  // namespace Image
}  // namespace VideoStitch
